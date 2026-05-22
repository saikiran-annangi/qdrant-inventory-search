"""
Embedding model loaders and query encoding.

Dense:  sentence-transformers/all-mpnet-base-v2 (768d cosine)
Sparse: Qdrant/bm25 via FastEmbed (two independent BM25 fields)

Both models are singletons -- loaded once and reused across calls.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qdrant_client.models import SparseVector

from config import DENSE_MODEL_NAME, BM25_MODEL_NAME

_dense_model = None
_bm25_model = None


class _MpnetEncoder:
    """
    Lightweight mean-pool + L2-normalize wrapper around AutoModel.

    SentenceTransformer.__init__ calls self.to(device) after loading the
    underlying transformer.  On PyTorch >= 2.8 this raises
    ``NotImplementedError: Cannot copy out of meta tensor`` because the
    transformers library initialises some tensors on the ``meta`` device
    and PyTorch 2.8 no longer allows moving them with .to().

    Bypass: load via AutoModel directly with low_cpu_mem_usage=False so
    weights land on CPU immediately, then do the same mean-pool + cosine
    normalisation that all-mpnet-base-v2 uses.
    """

    def __init__(self, model_name: str) -> None:
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
        """Return a numpy array (768,) for the given sentence."""
        import torch.nn.functional as F

        enc = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            out = self.model(**enc)

        # Mean pooling masked by attention
        tok = out.last_hidden_state                          # (1, seq, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()   # (1, seq, 1)
        vec = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (1, 768)

        if normalize_embeddings:
            vec = F.normalize(vec, p=2, dim=1)

        return vec[0].cpu().numpy()


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
    from data.normalizers import model_number_variants, normalize_specs

    dense_model = get_dense_model()
    bm25_model = get_bm25_model()

    # Dense embedding
    dense_vec = dense_model.encode(
        query, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    # Sparse: model-number field uses variant expansion
    model_text = model_number_variants(query) or query

    # Sparse: description field uses spec normalization
    desc_text = normalize_specs(query) or query

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
