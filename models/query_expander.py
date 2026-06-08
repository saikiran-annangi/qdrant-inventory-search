"""LLM-powered query expansion — the permanent catch-all for unknown MEP trade terms.

The static synonym map in data/synonyms.py covers known jargon. But the vocabulary
gap between how tradespeople search and how products are described exists across ALL
MEP domains, and we don't know in advance which terms will fail next.

This module is the structural fix: when a query contains jargon the static map
doesn't cover, ask a fast LLM to expand it with relevant trade synonyms. The LLM
knows the MEP domain vocabulary across electrical, plumbing, and mechanical —
so a plumber searching "Fernco coupling" or an HVAC tech searching "VAV reheat
box" gets expanded automatically, without any human adding a new synonym entry.

Design:
  - Called ONLY when the static synonym map had no hits (not every query).
  - Results cached in memory (process lifetime) and on disk (query_expansion_cache.json).
  - Fast: single-line output, temperature=0, small model.
  - Graceful degradation: returns the original query on any failure.

The disk cache means each unique query pattern is expanded only once, ever.
Subsequent runs (restarts, new sessions) reuse cached expansions immediately.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_client = None
_mem_cache: dict[str, str] = {}   # query -> expanded (runtime)
_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "query_expansion_cache.json",
)


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


def _load_disk_cache() -> None:
    if _mem_cache:
        return
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                _mem_cache.update(json.load(f))
        except Exception:
            pass


def _save_disk_cache() -> None:
    try:
        tmp = _CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_mem_cache, f, indent=2)
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        pass


def is_enabled() -> bool:
    return _get_client() is not None


_PROMPT = """\
You are an MEP (mechanical/electrical/plumbing) trade synonym assistant for a \
distributor catalog search engine.

Given a product search query, add the trade jargon, field names, and synonyms \
that electricians, plumbers, HVAC technicians, or mechanical contractors use for \
the same product. Cover all relevant MEP trades — not just electrical.

Rules:
- Output ONLY the extra words to append (space-separated, lowercase).
- Do NOT repeat words already in the query.
- If the query is already clear technical language with no jargon gap, output nothing.
- Maximum 10 words.

Examples:
Query: fernco coupling          → flexible rubber coupling no-hub mission drain pipe
Query: shark bite fitting       → push fit push to connect speedfit plumbing connector
Query: p trap sink              → bottle trap u bend drain trap waste fitting
Query: vav box hvac             → variable air volume terminal unit reheat box
Query: fcu unit                 → fan coil unit hydronic terminal chilled water
Query: mcb breaker              → miniature circuit breaker rcbo overload protection
Query: dry connector            → bx flex armored non liquidtight
Query: gpo outlet               → power point socket receptacle general purpose outlet
Query: sweat fitting copper     → solder fitting soldered joint capillary fitting
Query: fernco 4 inch            → flexible coupling no-hub rubber band coupler drain

Query: {query}
→"""


def expand(query: str, timeout: float = 4.0) -> str:
    """Return query + LLM-suggested trade synonyms, or original query on failure.

    Cached: each unique query is only sent to the LLM once.
    """
    _load_disk_cache()

    key = query.strip().lower()
    if key in _mem_cache:
        cached = _mem_cache[key]
        return f"{query} {cached}".strip() if cached else query

    client = _get_client()
    if client is None:
        return query

    try:
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
            max_tokens=30,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        # Strip any leading "→" or punctuation the model adds
        raw = re.sub(r"^[→\-\s]+", "", raw).strip()
        # Only keep word-like tokens, drop anything that looks like a full sentence
        tokens = [t for t in raw.split() if re.match(r"^[a-z0-9/\-]+$", t)]
        expansion = " ".join(tokens[:10])

        _mem_cache[key] = expansion
        _save_disk_cache()

        return f"{query} {expansion}".strip() if expansion else query

    except Exception as exc:
        logger.debug("Query expansion failed for %r: %s", query, exc)
        _mem_cache[key] = ""   # cache the miss so we don't retry on every search
        _save_disk_cache()
        return query
