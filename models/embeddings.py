"""
Embedding model loaders and query encoding.

Dense:  sentence-transformers/all-mpnet-base-v2 (768d cosine)
Sparse: Qdrant/bm25 via FastEmbed (two independent BM25 fields)

Both models are singletons -- loaded once and reused across calls.

PyTorch 2.8 notes
-----------------
Two issues surfaced with torch==2.8 + transformers>=4.50:

1. SentenceTransformer.__init__ calls self.to(device) after loading the
   underlying AutoModel, but transformers may leave some tensors on the
   meta device.  PyTorch 2.8 forbids copying meta tensors with .to(),
   so we bypass SentenceTransformer entirely and use AutoModel directly.

2. Streamlit's execution context can activate torch._dynamo / fake-tensor
   dispatch, routing operations through meta-tensor shape checks even
   during normal eager inference.  Setting TORCHDYNAMO_DISABLE=1 and
   calling torch._dynamo.config.disable = True before any model load
   prevents this.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable torch.compile / dynamo BEFORE any torch import so that
# Streamlit's execution environment cannot activate fake-tensor dispatch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from qdrant_client.models import SparseVector

from config import DENSE_MODEL_NAME, BM25_MODEL_NAME

_dense_model = None
_bm25_model = None


def _disable_dynamo() -> None:
    """Best-effort: turn off torch._dynamo via every available API."""
    try:
        import torch._dynamo as _dyn
        _dyn.config.disable = True
        _dyn.reset()
    except Exception:
        pass


def _mpnet_forward(model, enc: dict):
    """
    Isolated forward function decorated with torch.compiler.disable.

    Keeps the model call outside torch.compile / dynamo tracing so that
    PyTorch 2.8's fake-tensor dispatch cannot intercept it.
    """
    return model(**enc)


# Apply torch.compiler.disable as a decorator at definition time.
# Falls back gracefully if the API is unavailable (older torch).
try:
    import torch as _torch
    _mpnet_forward = _torch.compiler.disable(_mpnet_forward)
except Exception:
    pass


class _MpnetEncoder:
    """
    Thin mean-pool + L2-normalise wrapper around AutoModel.

    Uses AutoModel.from_pretrained directly (low_cpu_mem_usage=False so
    weights land on CPU immediately) and performs the same mean-pool +
    cosine-normalise encoding as all-mpnet-base-v2.  Wraps inference in
    torch.compiler.disable to prevent fake-tensor dispatch on PyTorch 2.8.
    """

    def __init__(self, model_name: str) -> None:
        _disable_dynamo()

        import torch
        from transformers import AutoTokenizer, AutoModel

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=False)
        self.model.eval()

    def encode(
        self,
        sentence: str,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        """Return a numpy (768,) embedding for the given sentence."""
        import torch.nn.functional as F

        enc = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with self._torch.no_grad():
            out = _mpnet_forward(self.model, enc)

        tok  = out.last_hidden_state                          # (1, seq, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()    # (1, seq, 1)
        vec  = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (1, 768)

        if normalize_embeddings:
            vec = F.normalize(vec, p=2, dim=1)

        return vec[0].cpu().numpy()

    def encode_batch(
        self,
        sentences: list,
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        """Return a numpy (N, 768) array for a list of sentences.

        Same mean-pool + L2-normalize as encode(), but tokenizes and runs the
        model in batches — ~10-20x faster than calling encode() per item, which
        matters a lot for ingest (tens of thousands of products).
        """
        import numpy as np
        import torch.nn.functional as F

        out_vecs = []
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start:start + batch_size]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            with self._torch.no_grad():
                out = _mpnet_forward(self.model, enc)
            tok  = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec  = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            if normalize_embeddings:
                vec = F.normalize(vec, p=2, dim=1)
            out_vecs.append(vec.cpu().numpy())
        if not out_vecs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.vstack(out_vecs)


def get_dense_model():
    """Return the dense encoder, loading it on first call."""
    global _dense_model
    if _dense_model is None:
        _dense_model = _MpnetEncoder(DENSE_MODEL_NAME)
    return _dense_model


def get_bm25_model():
    """Return the FastEmbed BM25 model, loading it on first call."""
    global _bm25_model
    if _bm25_model is None:
        from fastembed import SparseTextEmbedding
        _bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
    return _bm25_model


def encode_query(query: str) -> tuple:
    """
    Encode a query into three vectors used for hybrid search.

    Returns:
        dense_vec        -- list[float], 768d cosine-normalized embedding
        sparse_model_vec -- SparseVector, BM25 over model number variants
        sparse_desc_vec  -- SparseVector, BM25 over spec-normalized text
    """
    from data.normalizers import model_number_variants, spec_text_with_attributes, strip_model_number_prefix
    from data.synonyms import expand_synonyms

    dense_model = get_dense_model()
    bm25_model  = get_bm25_model()

    # Static trade-jargon synonym map (zero latency, no LLM call).
    # Bridges known MEP trade terms to catalog vocabulary for products that may
    # not yet have been enriched (e.g. a newly added source before next ingest).
    # For enriched products this is a safety net — the trade terms are already
    # baked into the product's extended_description by domain-aware enrichment,
    # so the dense and BM25 channels find them without query expansion anyway.
    # The LLM query expander was removed from the hot path: enrichment solved
    # the vocabulary gap at the index side, making a per-query LLM call redundant.
    expanded = expand_synonyms(query)

    # Dense embedding (on the jargon-expanded text)
    dense_vec = dense_model.encode(
        expanded, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    # Sparse: model-number field uses variant expansion. Uses the BARE query
    # (no synonyms — synonyms must not pollute part-number matching).
    # Strip ERP prefixes (p/n, cat#, sku, stk no.) first so noise tokens
    # ("cat", "p", "n") don't flood the BM25 pool with wrong products.
    bare_query = strip_model_number_prefix(query)
    model_text = model_number_variants(bare_query) or bare_query

    # Sparse: description field uses spec + dimension normalization plus
    # canonical attribute anchors (so lamp bases / NEMA classes match), on the
    # jargon-expanded query. Metric bridging is query-side only (documents stay
    # canonical at ingest).
    desc_text = spec_text_with_attributes(expanded, bridge_metric=True) or expanded

    sm_result = list(bm25_model.embed([model_text]))[0]
    sd_result = list(bm25_model.embed([desc_text]))[0]

    sparse_model_vec = SparseVector(
        indices=sm_result.indices.tolist(),
        values=sm_result.values.tolist(),
    )
    sparse_desc_vec = SparseVector(
        indices=sd_result.indices.tolist(),
        values=sd_result.values.tolist(),
    )

    return dense_vec, sparse_model_vec, sparse_desc_vec
