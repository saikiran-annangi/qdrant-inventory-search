"""
Unified data loader for all 8 inventory sources.

Each loader:
  - Reads the raw CSV / XLSX from inventory_data/
  - Deduplicates within the source by internal_id
  - Aggregates per-branch rows into a locations[] array
  - Optionally calls Gemini to generate descriptions for sparse products
  - Returns a list of payload dicts ready for embedding and upsert

The _bm25_model and _bm25_desc fields are used only during ingestion
to generate BM25 sparse vectors and are stripped before storage.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR
from data.normalizers import (
    normalize_manufacturer,
    normalize_specs,
    model_number_variants,
    make_id,
    is_sparse_description,
)

# ---------------------------------------------------------------------------
# Gemini description generation
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print("  [WARN] GEMINI_API_KEY not set -- skipping LLM description generation")
                return None
            genai.configure(api_key=api_key)
            _gemini_client = genai.GenerativeModel("gemini-1.5-flash")
        except ImportError:
            print("  [WARN] google-generativeai not installed -- skipping LLM description generation")
            return None
    return _gemini_client


def generate_description(model_number: str, manufacturer: str, category: str) -> str:
    """Call Gemini to produce a 1-2 sentence product description."""
    client = _get_gemini_client()
    if client is None:
        return f"{manufacturer} {model_number}".strip()

    prompt = (
        f"Write a concise 1-2 sentence product description for an electrical/industrial product. "
        f"Manufacturer: {manufacturer}. Model: {model_number}. Category: {category}. "
        f"Include key specs if inferable from the model number. No marketing language."
    )
    try:
        response = client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  [WARN] Gemini error for {model_number}: {e}")
        return f"{manufacturer} {model_number}".strip()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_float(val):
    """Convert to float, handling currency strings like '$35.53'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _build_locations_from_rows(
    df: pd.DataFrame,
    id_col: str,
    loc_name_col: str,
    loc_id_col: str,
    qoh_col: str,
    qoo_col=None,
    cost_col=None,
    sell_col=None,
    reorder_col=None,
    priority_col=None,
) -> dict:
    """
    Group a multi-row-per-product DataFrame into a dict of locations arrays.

    Returns: {internal_id: [location_dict, ...]}
    """
    groups: dict = {}
    for _, row in df.iterrows():
        iid = str(row[id_col]).strip()
        qoh = _to_float(row[qoh_col]) or 0.0
        loc = {
            "location_name":     str(row[loc_name_col]).strip() if pd.notna(row[loc_name_col]) else "",
            "location_erp_id":   str(row[loc_id_col]).strip() if pd.notna(row[loc_id_col]) else "",
            "qoh":               int(qoh),
            "quantity_on_order": int(_to_float(row.get(qoo_col)) or 0) if qoo_col else 0,
            "cost":              _to_float(row.get(cost_col)) if cost_col else None,
            "sell_price":        _to_float(row.get(sell_col)) if sell_col else None,
            "reorder_decision":  bool(row[reorder_col]) if reorder_col and pd.notna(row.get(reorder_col)) else False,
            "stock_priority":    str(row[priority_col]).strip() if priority_col and pd.notna(row.get(priority_col)) else None,
            "in_stock":          qoh > 0,
        }
        groups.setdefault(iid, []).append(loc)
    return groups


def _rollup(locations: list) -> dict:
    """Compute aggregate fields (total QOH, cost range) from a locations array."""
    qohs  = [loc["qoh"] for loc in locations]
    costs = [loc["cost"] for loc in locations if loc.get("cost") is not None]
    sells = [loc["sell_price"] for loc in locations if loc.get("sell_price") is not None]
    total_qoh = sum(qohs)
    return {
        "has_stock":       total_qoh > 0,
        "total_qoh":       total_qoh,
        "location_count":  len(locations),
        "min_cost":        round(min(costs), 4) if costs else None,
        "max_cost":        round(max(costs), 4) if costs else None,
        "avg_sell_price":  round(sum(sells) / len(sells), 4) if sells else None,
    }


# ---------------------------------------------------------------------------
# Individual source loaders
# ---------------------------------------------------------------------------


def load_au_parspec() -> list:
    """AU Parspec CSV -- multiple rows per product (one row per branch)."""
    source = "au_parspec"
    path = os.path.join(DATA_DIR, "AU Parspec inventory load 10032026 SEND AB V 1 Test copy for demo.csv")
    df = pd.read_csv(path, low_memory=False)

    df["_internal_id"] = (
        df["Manufacturer Abbreviation"].fillna("").astype(str)
        + df["Model Number"].fillna("").astype(str)
    )

    loc_groups = _build_locations_from_rows(
        df, "_internal_id",
        loc_name_col="Stock Location Name",
        loc_id_col="Stock Location ERP ID",
        qoh_col="QOH",
        qoo_col="Quantity On Order",
        cost_col="Product Cost",
        sell_col="Product Sell Price",
        reorder_col="Reorder Decision",
        priority_col="Stock Priority",
    )

    deduped = df.drop_duplicates(subset=["_internal_id"]).copy()
    records = []

    for _, row in deduped.iterrows():
        iid       = str(row["_internal_id"]).strip()
        locations = loc_groups.get(iid, [])
        rollup    = _rollup(locations)
        desc      = str(row["Description"]).strip() if pd.notna(row["Description"]) else ""
        mfr       = normalize_manufacturer(row["Manufacturer Name"])
        model     = str(row["Model Number"]).strip() if pd.notna(row["Model Number"]) else ""
        cat       = str(row["Product Category"]).strip() if pd.notna(row["Product Category"]) else ""
        uom       = str(row["Selling UOM"]).strip() if pd.notna(row["Selling UOM"]) else ""

        if is_sparse_description(desc):
            desc = generate_description(model, mfr, cat)

        records.append({
            "id":                   make_id(source, iid),
            "source":               source,
            "internal_id":          iid,
            "model_number":         model,
            "description":          desc,
            "extended_description": None,
            "manufacturer_name":    mfr,
            "manufacturer_abbrev":  str(row["Manufacturer Abbreviation"]).strip() if pd.notna(row["Manufacturer Abbreviation"]) else "",
            "product_category":     cat,
            "uom":                  uom,
            "currency":             "AUD",
            **rollup,
            "locations":            locations,
            "_bm25_model":          model_number_variants(model),
            "_bm25_desc":           normalize_specs(f"{desc} {mfr} {cat}"),
        })
    return records


def load_burnaby_dc() -> list:
    """Burnaby DC Lighting Excel -- one product per row (pre-aggregated)."""
    source = "burnaby_dc"
    path = os.path.join(DATA_DIR, "Burnaby DC Lighting Inventory 9 17.xlsx")
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.reset_index(drop=True)
    df["_internal_id"] = df["Item"].astype(str).str.strip() + "_" + df.index.astype(str)

    records = []
    for _, row in df.iterrows():
        iid      = str(row["_internal_id"]).strip()
        qoh      = float(row["On Hand"]) if pd.notna(row["On Hand"]) else 0.0
        cost     = float(row["Low Repl Cost"]) if pd.notna(row["Low Repl Cost"]) else None
        desc_raw = str(row["Description"]).strip() if pd.notna(row["Description"]) else ""
        mfr      = normalize_manufacturer(row["Mfr"])
        parts    = desc_raw.split()
        model    = parts[-1] if parts else iid

        loc = {
            "location_name":     str(row["Branch name"]).strip() if pd.notna(row["Branch name"]) else "",
            "location_erp_id":   str(row["Supplier Name"]).strip() if pd.notna(row["Supplier Name"]) else "",
            "qoh":               int(qoh),
            "quantity_on_order": 0,
            "cost":              cost,
            "sell_price":        None,
            "reorder_decision":  False,
            "stock_priority":    None,
            "in_stock":          qoh > 0,
        }

        desc = generate_description(model, mfr, "Lighting") if is_sparse_description(desc_raw) else desc_raw

        records.append({
            "id":                   make_id(source, iid),
            "source":               source,
            "internal_id":          iid,
            "model_number":         model,
            "description":          desc,
            "extended_description": None,
            "manufacturer_name":    mfr,
            "manufacturer_abbrev":  "",
            "product_category":     "Lighting",
            "uom":                  str(row["UOM"]).strip() if pd.notna(row["UOM"]) else "EA",
            "has_stock":            qoh > 0,
            "total_qoh":            int(qoh),
            "location_count":       1,
            "min_cost":             cost,
            "max_cost":             cost,
            "avg_sell_price":       None,
            "currency":             "CAD",
            "locations":            [loc],
            "_bm25_model":          model_number_variants(model),
            "_bm25_desc":           normalize_specs(f"{desc} {mfr} Lighting"),
        })
    return records


def load_guillevin_1() -> list:
    """Guillevin 1 Excel -- multiple rows per item (one row per location)."""
    source = "guillevin_1"
    path = os.path.join(DATA_DIR, "Guillevin_inventory_data_1.xlsx")
    df = pd.read_excel(path)

    loc_groups = _build_locations_from_rows(
        df, "Item Id",
        loc_name_col="Location Location Name",
        loc_id_col="Location Id",
        qoh_col="Qty On Hand",
    )

    deduped = df.drop_duplicates(subset=["Item Id"]).copy()
    records = []

    for _, row in deduped.iterrows():
        iid       = str(row["Item Id"]).strip()
        locations = loc_groups.get(iid, [])
        rollup    = _rollup(locations)
        desc      = str(row["Item Desc"]).strip() if pd.notna(row["Item Desc"]) else ""
        model     = str(row["Supplier Part No"]).strip() if pd.notna(row["Supplier Part No"]) else iid
        mfr       = normalize_manufacturer(row["Name"])

        if is_sparse_description(desc):
            desc = generate_description(model, mfr, "")

        records.append({
            "id":                   make_id(source, iid),
            "source":               source,
            "internal_id":          iid,
            "model_number":         model,
            "description":          desc,
            "extended_description": None,
            "manufacturer_name":    mfr,
            "manufacturer_abbrev":  "",
            "product_category":     "",
            "uom":                  "EA",
            "currency":             "CAD",
            **rollup,
            "locations":            locations,
            "_bm25_model":          model_number_variants(model) + " " + model_number_variants(iid),
            "_bm25_desc":           normalize_specs(f"{desc} {mfr}"),
        })
    return records


def _load_standard_schema(
    source: str,
    path: str,
    currency: str,
    loc_name_col: str,
    loc_id_col: str,
    skip_first_row: bool = False,
    id_col: str = None,
    require_description: bool = False,
) -> list:
    """
    Generic loader for files sharing the Guillevin_2 / AU Parspec column schema.

    Columns expected: Stock Location Name, Stock Location ERP ID, QOH,
    Quantity On Order, UOM, Manufacturer Name, Manufacturer Abbreviation,
    Model Number, Product Category, Description, Product Cost, Product Sell Price.

    Args:
        id_col:              If set, use this column as internal_id instead of
                             Manufacturer Abbreviation + Model Number.
        require_description: Drop rows where Description is null (removes placeholder rows).
    """
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    if skip_first_row:
        df = df.iloc[1:].copy()

    if require_description:
        df = df[df["Description"].notna()].copy()

    # Normalize column name variants
    col_map = {
        "Stock Location ID":  "Stock Location ERP ID",
        "Stock Location Id":  "Stock Location ERP ID",
        "Quantity on Order":  "Quantity On Order",
        "Product Cost ":      "Product Cost",
        " Product Cost ":     "Product Cost",
    }
    df.rename(columns=col_map, inplace=True)
    loc_id_col   = col_map.get(loc_id_col, loc_id_col)
    loc_name_col = col_map.get(loc_name_col, loc_name_col)

    df["Manufacturer Abbreviation"] = df.get("Manufacturer Abbreviation", pd.Series("")).fillna("")
    df["Model Number"] = df.get("Model Number", pd.Series("")).fillna("").astype(str).str.strip()

    if id_col and id_col in df.columns:
        df["_internal_id"] = df[id_col].fillna("").astype(str).str.strip()
    else:
        df["_internal_id"] = (
            df["Manufacturer Abbreviation"].astype(str).str.strip()
            + df["Model Number"]
        )

    loc_groups = _build_locations_from_rows(
        df, "_internal_id",
        loc_name_col=loc_name_col,
        loc_id_col=loc_id_col,
        qoh_col="QOH",
        qoo_col="Quantity On Order" if "Quantity On Order" in df.columns else None,
        cost_col="Product Cost" if "Product Cost" in df.columns else None,
        sell_col="Product Sell Price" if "Product Sell Price" in df.columns else None,
        reorder_col="Reorder Decision" if "Reorder Decision" in df.columns else None,
        priority_col="Stock Priority" if "Stock Priority" in df.columns else None,
    )

    deduped = df.drop_duplicates(subset=["_internal_id"]).copy()
    records = []

    for _, row in deduped.iterrows():
        iid = str(row["_internal_id"]).strip()
        if not iid:
            continue

        locations = loc_groups.get(iid, [])
        rollup    = _rollup(locations)
        desc      = str(row.get("Description", "")).strip() if pd.notna(row.get("Description")) else ""
        model     = str(row["Model Number"]).strip()
        # When model_number is empty and the source uses a dedicated id column
        # (e.g. Standard Supply uses Product ERP Code), the ERP code IS the
        # model number buyers search for -- use it as a fallback.
        if not model and id_col:
            model = iid
        mfr       = normalize_manufacturer(row.get("Manufacturer Name", ""))
        cat       = str(row.get("Product Category", "")).strip() if pd.notna(row.get("Product Category")) else ""
        uom       = str(row.get("UOM", "EA")).strip() if pd.notna(row.get("UOM")) else "EA"
        mfr_abbrev = str(row.get("Manufacturer Abbreviation", "")).strip()

        if is_sparse_description(desc):
            desc = generate_description(model, mfr, cat)

        records.append({
            "id":                   make_id(source, iid),
            "source":               source,
            "internal_id":          iid,
            "model_number":         model,
            "description":          desc,
            "extended_description": None,
            "manufacturer_name":    mfr,
            "manufacturer_abbrev":  mfr_abbrev,
            "product_category":     cat,
            "uom":                  uom,
            "currency":             currency,
            **rollup,
            "locations":            locations,
            "_bm25_model":          model_number_variants(model),
            "_bm25_desc":           normalize_specs(f"{desc} {mfr} {cat}"),
        })
    return records


def load_guillevin_2() -> list:
    path = os.path.join(DATA_DIR, "Guillevin_inventory_data_2_utf8.csv")
    # The file contains 1M+ placeholder rows after the real data.
    # require_description=True drops all rows where Description is null.
    return _load_standard_schema(
        "guillevin_2", path, "CAD",
        loc_name_col="Stock Location Name",
        loc_id_col="Stock Location ERP ID",
        require_description=True,
    )


def load_inventory_sample() -> list:
    """INVENTORY SAMPLE.xlsx -- Mersen fuses, no location data."""
    source = "inventory_sample"
    path = os.path.join(DATA_DIR, "INVENTORY SAMPLE.xlsx")
    df = pd.read_excel(path)
    df.columns = ["manufacturer_name", "model_number", "description", "extended_description"]

    records = []
    for _, row in df.iterrows():
        model = str(row["model_number"]).strip() if pd.notna(row["model_number"]) else ""
        if not model or model.lower() == "nan":
            continue

        mfr  = normalize_manufacturer(row["manufacturer_name"])
        desc = str(row["description"]).strip() if pd.notna(row["description"]) else ""
        ext  = str(row["extended_description"]).strip() if pd.notna(row["extended_description"]) else None
        iid  = model

        if is_sparse_description(desc):
            desc = generate_description(model, mfr, "Fuses")

        locations = [{
            "location_name":     "Default",
            "location_erp_id":   "0",
            "qoh":               0,
            "quantity_on_order": 0,
            "cost":              None,
            "sell_price":        None,
            "reorder_decision":  False,
            "stock_priority":    None,
            "in_stock":          False,
        }]

        records.append({
            "id":                   make_id(source, iid),
            "source":               source,
            "internal_id":          iid,
            "model_number":         model,
            "description":          desc,
            "extended_description": ext,
            "manufacturer_name":    mfr,
            "manufacturer_abbrev":  "",
            "product_category":     "Fuses",
            "uom":                  "EA",
            "currency":             "USD",
            "has_stock":            False,
            "total_qoh":            0,
            "location_count":       1,
            "min_cost":             None,
            "max_cost":             None,
            "avg_sell_price":       None,
            "locations":            locations,
            "_bm25_model":          model_number_variants(model),
            "_bm25_desc":           normalize_specs(f"{desc} {mfr} Fuses {ext or ''}"),
        })
    return records


def load_plumbing() -> list:
    path = os.path.join(DATA_DIR, "Plumbing Inventory example.xlsx")
    return _load_standard_schema(
        "plumbing", path, "USD",
        loc_name_col="Stock Location Name",
        loc_id_col="Stock Location ERP ID",
        skip_first_row=True,
    )


def load_standard_supply() -> list:
    path = os.path.join(DATA_DIR, "Standard Supply_inventory_data_1.csv")
    # Model Number is null for most rows; use Product ERP Code as the ID.
    return _load_standard_schema(
        "standard_supply", path, "USD",
        loc_name_col="Stock Location Name",
        loc_id_col="Stock Location ID",
        id_col="Product ERP Code",
    )


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

LOADERS: dict = {
    "au_parspec":       load_au_parspec,
    "burnaby_dc":       load_burnaby_dc,
    "guillevin_1":      load_guillevin_1,
    "guillevin_2":      load_guillevin_2,
    "inventory_sample": load_inventory_sample,
    "plumbing":         load_plumbing,
    "standard_supply":  load_standard_supply,
}


def load_all(sources: list = None, verbose: bool = True) -> list:
    """
    Load all (or a subset of) inventory sources.

    Args:
        sources: List of source keys to load. Defaults to all known sources.
        verbose: Print per-source record counts.

    Returns:
        Flat list of payload dicts ready for embedding and upsert.
    """
    sources = sources or list(LOADERS.keys())
    all_records = []

    for src in sources:
        loader = LOADERS.get(src)
        if loader is None:
            if verbose:
                print(f"  [WARN] Unknown source '{src}' -- skipping")
            continue
        if verbose:
            print(f"  Loading {src}...", end=" ", flush=True)
        try:
            recs = loader()
            if verbose:
                print(f"{len(recs)} products")
            all_records.extend(recs)
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    if verbose:
        print(f"\nTotal products loaded: {len(all_records)}")
    return all_records
