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


def get_dense_model():
    """Return the dense SentenceTransformer model, loading it on first call.

    model_kwargs={"low_cpu_mem_usage": False} prevents transformers from
    initializing weights on the 'meta' device, which causes a
    NotImplementedError when SentenceTransformer subsequently calls
    self.to("cpu") on PyTorch >= 2.8.
    """
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        _dense_model = SentenceTransformer(
            DENSE_MODEL_NAME,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": False},
        )
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
