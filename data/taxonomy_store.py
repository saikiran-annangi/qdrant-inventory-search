"""Open, self-growing taxonomy store — the single source of truth at runtime.

The curated controlled vocabulary in data/taxonomy.py (PRODUCT_TAXONOMY) is the
SEED, not a ceiling. In production, new source files bring product types that
don't fit any seed node. Rather than force them into the nearest wrong bucket or
leave them unclassified, ingestion MINTS a new (category, subcategory) — but
under strict controls so the vocabulary can't fragment the way the old
LLM-invented 1,473-node cache did:

  1. ONE store. Both ingestion and the query side read this same file, so any
     node ingestion creates is immediately query-reachable. No drift.
  2. Controlled creation. Before minting, names are normalized and the candidate
     is deduped against existing nodes — exact (case-insensitive) AND semantic
     (cosine vs existing node embeddings). A near-duplicate reuses the existing
     node instead of creating a new one.
  3. Provenance. Every node records seed|auto, product_count, so auto nodes can
     be reviewed/merged (scripts/taxonomy_review.py).

This module is deliberately MODEL-FREE: embeddings are passed in by the caller
(the ingest classifier already loads an encoder). The query side only needs the
label list, so it never has to load a model.

Files written (one writer = store.save(), so they can't drift):
  taxonomy_store.json   full nodes incl. embeddings + provenance (ingest side)
  taxonomy_labels.json  {domain: ["Category > Subcategory", ...]} (query side)
"""

from __future__ import annotations

import json
import os

import numpy as np

# Cosine ≥ this between a product and a node ⇒ confident match, assign existing.
DEFAULT_ASSIGN_THRESHOLD = 0.55
# Cosine ≥ this between a *new node name* and an existing node ⇒ same concept,
# reuse instead of creating (prevents "Cable Ties & Wraps" vs "Cable ties ...").
DEFAULT_DEDUP_THRESHOLD = 0.86


def _clean(s: str) -> str:
    """Light display normalization: trim + collapse internal whitespace."""
    return " ".join(str(s or "").split())


def _to_floats(embedding):
    """Coerce an embedding (numpy array / list of np.float32) to plain floats so
    it is JSON-serializable."""
    if embedding is None:
        return None
    return [float(x) for x in embedding]


def _key(domain: str, category: str, subcategory: str) -> tuple:
    """Case-insensitive identity of a node (for exact dedup)."""
    return (
        _clean(domain).casefold(),
        _clean(category).casefold(),
        _clean(subcategory).casefold(),
    )


def node_text(domain: str, category: str, subcategory: str) -> str:
    """Canonical text used to embed a node (matches the seed embedding format)."""
    return f"{domain} | {category} | {subcategory}"


class TaxonomyStore:
    """A growing, deduped taxonomy of nodes with embeddings and provenance."""

    def __init__(self, store_path: str, labels_path: str | None = None):
        self.store_path = store_path
        self.labels_path = labels_path or os.path.join(
            os.path.dirname(store_path), "taxonomy_labels.json"
        )
        self._nodes: list[dict] = []
        self._index: dict[tuple, dict] = {}
        # per-domain cached embedding matrix, invalidated on add
        self._domain_cache: dict[str, tuple] = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────

    def load(self) -> None:
        self._nodes, self._index, self._domain_cache = [], {}, {}
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                data = json.load(f)
            for n in data.get("nodes", []):
                self._register(n)

    def save(self) -> None:
        tmp = self.store_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "nodes": self._nodes}, f)
        os.replace(tmp, self.store_path)
        # Lightweight labels projection for the query side (no embeddings).
        labels = self.labels_by_domain()
        tmp2 = self.labels_path + ".tmp"
        with open(tmp2, "w") as f:
            json.dump(labels, f, indent=2)
        os.replace(tmp2, self.labels_path)

    def _register(self, node: dict) -> None:
        self._nodes.append(node)
        self._index[_key(node["domain"], node["category"], node["subcategory"])] = node
        self._domain_cache.pop(node["domain"], None)

    # ── queries ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._nodes)

    def labels_by_domain(self) -> dict:
        out: dict[str, set] = {}
        for n in self._nodes:
            out.setdefault(n["domain"], set()).add(f"{n['category']} > {n['subcategory']}")
        return {d: sorted(v) for d, v in out.items()}

    def labels_for_domain(self, domain: str) -> dict:
        """{category: [subcategory, ...]} for one domain — for the LLM namer."""
        out: dict[str, list] = {}
        for n in self._nodes:
            if n["domain"] == domain:
                out.setdefault(n["category"], []).append(n["subcategory"])
        return {c: sorted(set(subs)) for c, subs in out.items()}

    def get(self, domain: str, category: str, subcategory: str):
        return self._index.get(_key(domain, category, subcategory))

    def _domain_matrix(self, domain: str):
        """(nodes, matrix) for a domain; matrix is L2-normalized embeddings."""
        cached = self._domain_cache.get(domain)
        if cached is not None:
            return cached
        nodes = [n for n in self._nodes if n["domain"] == domain and n.get("embedding")]
        if not nodes:
            res = ([], None)
        else:
            mat = np.array([n["embedding"] for n in nodes], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            res = (nodes, mat / norms)
        self._domain_cache[domain] = res
        return res

    def match(self, vec, domain: str):
        """Best node in `domain` for a query/product vector → (node, cosine)."""
        top = self.topk(vec, domain, k=1)
        return top[0] if top else (None, -1.0)

    def topk(self, vec, domain: str, k: int = 3):
        """Top-k nodes in `domain` by cosine → list of (node, cosine)."""
        nodes, mat = self._domain_matrix(domain)
        if mat is None:
            return []
        v = np.asarray(vec, dtype=np.float32)
        nv = np.linalg.norm(v)
        if nv == 0:
            return []
        sims = mat @ (v / nv)
        order = np.argsort(sims)[::-1][:k]
        return [(nodes[i], float(sims[i])) for i in order]

    # ── mutation ─────────────────────────────────────────────────

    def seed_from(self, product_taxonomy: dict, embed_fn) -> int:
        """Add every curated node not already present (provenance='seed').

        embed_fn(text) -> list[float]. Batched embedding is the caller's job;
        here we call it per missing node (only on first build).
        """
        added = 0
        for domain, cats in product_taxonomy.items():
            for category, subs in cats.items():
                for subcategory in subs:
                    if self.get(domain, category, subcategory):
                        continue
                    emb = embed_fn(node_text(domain, category, subcategory))
                    self._register({
                        "domain": domain,
                        "category": _clean(category),
                        "subcategory": _clean(subcategory),
                        "provenance": "seed",
                        "product_count": 0,
                        "embedding": _to_floats(emb),
                    })
                    added += 1
        return added

    def add_node(
        self,
        domain: str,
        category: str,
        subcategory: str,
        name_embedding=None,
        provenance: str = "auto",
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ):
        """Return a canonical node for (domain, category, subcategory).

        Reuses an existing node when the name matches exactly (case-insensitive)
        or is semantically near an existing node (cosine ≥ dedup_threshold on the
        node-name embedding). Only mints a genuinely novel node. Returns
        (node, created: bool).
        """
        domain, category, subcategory = _clean(domain), _clean(category), _clean(subcategory)
        if not (domain and category and subcategory):
            return None, False

        exact = self.get(domain, category, subcategory)
        if exact is not None:
            return exact, False

        if name_embedding is not None:
            near, score = self.match(name_embedding, domain)
            if near is not None and score >= dedup_threshold:
                return near, False  # same concept, different words → reuse

        node = {
            "domain": domain,
            "category": category,
            "subcategory": subcategory,
            "provenance": provenance,
            "product_count": 0,
            "embedding": _to_floats(name_embedding),
        }
        self._register(node)
        return node, True

    def bump(self, node: dict, by: int = 1) -> None:
        if node is not None:
            node["product_count"] = node.get("product_count", 0) + by

    def merge(self, src: dict, dst: dict) -> int:
        """Fold node `src` into `dst`; returns src's product_count. Caller is
        responsible for re-labeling affected products (see taxonomy_review.py)."""
        moved = src.get("product_count", 0)
        self.bump(dst, moved)
        self._nodes = [n for n in self._nodes if n is not src]
        self._index.pop(_key(src["domain"], src["category"], src["subcategory"]), None)
        self._domain_cache.pop(src["domain"], None)
        return moved
