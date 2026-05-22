"""
Qdrant filter builder.

Translates keyword arguments into a Qdrant Filter object.
Returns None when no conditions are specified (no-op filter).
"""

from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range


def build_filter(
    source: str = None,
    has_stock: bool = None,
    manufacturer_name: str = None,
    product_category: str = None,
    min_cost_gte: float = None,
    min_cost_lte: float = None,
) -> Optional[Filter]:
    """
    Build a Qdrant Filter from optional keyword constraints.

    Args:
        source:           Match a specific distributor (e.g. 'guillevin_1').
        has_stock:        Filter by stock availability (True / False).
        manufacturer_name: Exact manufacturer match.
        product_category: Exact product category match.
        min_cost_gte:     Minimum cost >= value.
        min_cost_lte:     Minimum cost <= value.

    Returns:
        A Filter object, or None if no constraints were provided.
    """
    must = []

    if source:
        must.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if has_stock is not None:
        must.append(FieldCondition(key="has_stock", match=MatchValue(value=has_stock)))
    if manufacturer_name:
        must.append(FieldCondition(key="manufacturer_name", match=MatchValue(value=manufacturer_name)))
    if product_category:
        must.append(FieldCondition(key="product_category", match=MatchValue(value=product_category)))
    if min_cost_gte is not None or min_cost_lte is not None:
        range_params = {}
        if min_cost_gte is not None:
            range_params["gte"] = min_cost_gte
        if min_cost_lte is not None:
            range_params["lte"] = min_cost_lte
        must.append(FieldCondition(key="min_cost", range=Range(**range_params)))

    return Filter(must=must) if must else None
