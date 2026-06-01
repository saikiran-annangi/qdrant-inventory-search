"""
Inventory source loaders — shared library used by scripts/ingest.py and
the cache-building scripts (build_attributes_cache.py, build_taxonomy_cache.py).

Each loader reads one raw CSV/XLSX source, aggregates per-product rows into
a locations[] array, and returns a list of payload dicts. Three optional
JSON caches are merged in at load time if present:

    enrichment_cache.json   → extended_description (LLM-generated rich text)
    attributes_cache.json   → structured attributes {domain, explicit, inferred}
    taxonomy_cache.json     → taxonomy_domain / taxonomy_category / taxonomy_subcategory
"""

import hashlib
import json
import os
import uuid
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from data.normalizers import (
    normalize_manufacturer,
    normalize_specs,
    model_number_variants,
    make_id,
    is_sparse_description,
)

# ---------------------------------------------------------------------------
# Repo root and data directory
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = os.path.join(_REPO_ROOT, "inventory_data")

# ---------------------------------------------------------------------------
# Optional caches (built by scripts; gracefully absent on first run)
# ---------------------------------------------------------------------------

_enrichment_cache: dict = {}
_attributes_cache: dict = {}
_taxonomy_cache:   dict = {}
_caches_loaded = False


def _load_caches() -> None:
    global _enrichment_cache, _attributes_cache, _taxonomy_cache, _caches_loaded
    if _caches_loaded:
        return

    for attr, fname, label in [
        ("_enrichment_cache", "enrichment_cache.json",  "Enrichment"),
        ("_attributes_cache", "attributes_cache.json",  "Attributes"),
        ("_taxonomy_cache",   "taxonomy_cache.json",    "Taxonomy"),
    ]:
        path = os.path.join(_REPO_ROOT, fname)
        if os.path.exists(path):
            with open(path) as f:
                globals()[attr] = json.load(f)
            print(f"  [INFO] {label} cache loaded: {len(globals()[attr])} entries")

    _caches_loaded = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_float(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip().replace("$", "").replace(",", "")
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _aggregate(src, currency, df, *, id_col, mfr_col, model_col, desc_col,
               loc_name_col, loc_id_col, qoh_col,
               mfr_abbrev_col=None, cat_col=None, uom_col=None,
               qoo_col=None, cost_col=None, sell_col=None,
               reorder_col=None, priority_col=None, default_cat=""):
    records = []
    for iid, group in df.groupby(id_col, dropna=True):
        iid = str(iid).strip()
        if not iid or iid.lower() == "nan":
            continue
        first = group.iloc[0]
        locations = []
        for _, r in group.iterrows():
            qoh = int(_to_float(r.get(qoh_col)) or 0)
            locations.append({
                "location_name":     str(r.get(loc_name_col, "") or "").strip() if loc_name_col else "",
                "location_erp_id":   str(r.get(loc_id_col,   "") or "").strip() if loc_id_col   else "",
                "qoh":               qoh,
                "quantity_on_order": int(_to_float(r.get(qoo_col)) or 0) if qoo_col else 0,
                "cost":              _to_float(r.get(cost_col))    if cost_col    else None,
                "sell_price":        _to_float(r.get(sell_col))    if sell_col    else None,
                "reorder_decision":  (str(r.get(reorder_col, "")).strip().lower() == "yes") if reorder_col else False,
                "stock_priority":    (str(r.get(priority_col, "")).strip() or None) if priority_col else None,
                "in_stock":          qoh > 0,
            })

        qohs  = [l["qoh"]  for l in locations]
        costs = [l["cost"] for l in locations if l["cost"] is not None]
        sells = [l["sell_price"] for l in locations if l["sell_price"] is not None]
        total_qoh = sum(qohs)

        rec = {
            "id":                  make_id(src, iid),
            "source":              src,
            "internal_id":         iid,
            "description":         str(first.get(desc_col, "") or "").strip(),
            "manufacturer_name":   normalize_manufacturer(str(first.get(mfr_col, "") or "").strip()),
            "manufacturer_abbrev": str(first.get(mfr_abbrev_col, "") or "").strip() if mfr_abbrev_col else "",
            "model_number":        str(first.get(model_col, "") or "").strip(),
            "product_category":    (str(first.get(cat_col, "") or "").strip() if cat_col else "") or default_cat,
            "uom":                 (str(first.get(uom_col, "") or "").strip() or "EA") if uom_col else "EA",
            "currency":            currency,
            "has_stock":           total_qoh > 0,
            "total_qoh":           total_qoh,
            "location_count":      len(locations),
            "min_cost":            round(min(costs), 4) if costs else None,
            "max_cost":            round(max(costs), 4) if costs else None,
            "avg_sell_price":      round(sum(sells) / len(sells), 4) if sells else None,
            "locations":           locations,
        }
        records.append(rec)
    return records


def _attach_caches(records: list) -> list:
    """Attach enrichment, attributes, and taxonomy to each record from caches."""
    _load_caches()
    for rec in records:
        pid = rec["id"]
        rec["extended_description"] = _enrichment_cache.get(pid)
        rec["attributes"]           = _attributes_cache.get(pid, {"domain": "Unknown", "explicit": {}, "inferred": {}})
        tax = _taxonomy_cache.get(pid, {})
        rec["taxonomy_domain"]      = tax.get("taxonomy_domain",      "") or ""
        rec["taxonomy_category"]    = tax.get("taxonomy_category",    "") or ""
        rec["taxonomy_subcategory"] = tax.get("taxonomy_subcategory", "") or ""
    return records


# ---------------------------------------------------------------------------
# Individual source loaders
# ---------------------------------------------------------------------------

def load_au_parspec() -> list:
    df = pd.read_csv(
        os.path.join(DATA_DIR, "AU Parspec inventory load 10032026 SEND AB V 1 Test copy for demo.csv"),
        dtype=str, keep_default_na=False, low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate(
        "au_parspec", "AUD", df,
        id_col="Product ERP Code", mfr_col="Manufacturer Name",
        mfr_abbrev_col="Manufacturer Abbreviation",
        model_col="Model Number", desc_col="Description",
        cat_col="Product Category", uom_col="Selling UOM",
        loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
        qoh_col="QOH", qoo_col="Quantity On Order",
        cost_col="Product Cost", sell_col="Product Sell Price",
        reorder_col="Reorder Decision", priority_col="Stock Priority",
    )


def load_plumbing() -> list:
    df = pd.read_excel(os.path.join(DATA_DIR, "Plumbing Inventory example.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[1:].copy()
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate(
        "plumbing", "USD", df,
        id_col="Product ERP Code", mfr_col="Manufacturer Name",
        mfr_abbrev_col="Manufacturer Abbreviation",
        model_col="Model Number", desc_col="Description",
        cat_col="Product Category", uom_col="UOM",
        loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
        qoh_col="QOH", qoo_col="Quantity On Order",
        cost_col="Product Cost", sell_col="Product Sell Price",
        reorder_col="Reorder Decision", priority_col="Stock Priority",
    )


def load_standard_supply() -> list:
    df = pd.read_csv(
        os.path.join(DATA_DIR, "Standard Supply_inventory_data_1.csv"),
        dtype=str, keep_default_na=False, low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    if "Stock Location ID" in df.columns:
        df = df.rename(columns={"Stock Location ID": "Stock Location ERP ID"})
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate(
        "standard_supply", "USD", df,
        id_col="Product ERP Code", mfr_col="Manufacturer Name",
        mfr_abbrev_col="Manufacturer Abbreviation",
        model_col="Model Number", desc_col="Description",
        cat_col="Product Category", uom_col="UOM",
        loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
        qoh_col="QOH", qoo_col="Quantity on Order",
        cost_col="Product Cost", sell_col="Product Sell Price",
        reorder_col="Reorder Decision",
    )


def load_guillevin_2() -> list:
    df = pd.read_csv(
        os.path.join(DATA_DIR, "Guillevin_inventory_data_2_utf8.csv"),
        dtype=str, keep_default_na=False, low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    if "Stock Location Id" in df.columns:
        df = df.rename(columns={"Stock Location Id": "Stock Location ERP ID"})
    if "Product Cost " in df.columns:
        df = df.rename(columns={"Product Cost ": "Product Cost"})
    df = df[df["Description"].astype(str).str.strip() != ""]
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate(
        "guillevin_2", "CAD", df,
        id_col="Product ERP Code", mfr_col="Manufacturer Name",
        mfr_abbrev_col="Manufacturer Abbreviation",
        model_col="Model Number", desc_col="Description",
        cat_col="Product Category", uom_col="UOM",
        loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
        qoh_col="QOH", qoo_col="Quantity On Order",
        cost_col="Product Cost",
    )


def load_guillevin_1() -> list:
    df = pd.read_excel(os.path.join(DATA_DIR, "Guillevin_inventory_data_1.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Item Id"].astype(str).str.strip() != ""]
    return _aggregate(
        "guillevin_1", "CAD", df,
        id_col="Item Id", mfr_col="Name",
        model_col="Supplier Part No", desc_col="Item Desc",
        loc_name_col="Location Location Name", loc_id_col="Location Id",
        qoh_col="Qty On Hand",
    )


def load_burnaby_dc() -> list:
    df = pd.read_excel(os.path.join(DATA_DIR, "Burnaby DC Lighting Inventory 9 17.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.reset_index(drop=True)
    df["__id"] = df["Item"].astype(str).str.strip() + "_" + df.index.astype(str)
    records = []
    for _, r in df.iterrows():
        iid = r["__id"]
        if not iid or iid.startswith("_"):
            continue
        qoh  = int(_to_float(r.get("On Hand")) or 0)
        cost = _to_float(r.get("Low Repl Cost"))
        loc  = {
            "location_name":   str(r.get("Branch name",   "") or "").strip(),
            "location_erp_id": str(r.get("Supplier Name", "") or "").strip(),
            "qoh": qoh, "quantity_on_order": 0, "cost": cost,
            "sell_price": None, "reorder_decision": False,
            "stock_priority": None, "in_stock": qoh > 0,
        }
        records.append({
            "id":                make_id("burnaby_dc", iid),
            "source":            "burnaby_dc",
            "internal_id":       iid,
            "description":       str(r.get("Description", "") or "").strip(),
            "manufacturer_name": normalize_manufacturer(
                str(r.get("Mfr", "") or "").strip()
                or str(r.get("Supplier Name", "") or "").strip()
            ),
            "manufacturer_abbrev": "",
            "model_number":      str(r.get("Item", "") or "").strip(),
            "product_category":  "Lighting",
            "uom":               str(r.get("UOM", "EA") or "EA").strip(),
            "currency":          "CAD",
            "has_stock":         qoh > 0,
            "total_qoh":         qoh,
            "location_count":    1,
            "min_cost":          cost,
            "max_cost":          cost,
            "avg_sell_price":    None,
            "locations":         [loc],
        })
    return records


def load_inventory_sample() -> list:
    df = pd.read_excel(os.path.join(DATA_DIR, "INVENTORY SAMPLE.xlsx"), dtype=str)
    df.columns = ["manufacturer_name", "model_number", "description", "extended_description"]
    records = []
    for _, r in df.iterrows():
        model = str(r.get("model_number", "") or "").strip()
        if not model or model.lower() == "nan":
            continue
        mfr  = normalize_manufacturer(str(r.get("manufacturer_name", "") or "").strip())
        desc = str(r.get("description",           "") or "").strip()
        ext  = str(r.get("extended_description",  "") or "").strip()
        if ext.lower() == "nan":
            ext = ""
        loc = {
            "location_name": "Default", "location_erp_id": "0",
            "qoh": 0, "quantity_on_order": 0, "cost": None,
            "sell_price": None, "reorder_decision": False,
            "stock_priority": None, "in_stock": False,
        }
        records.append({
            "id":                  make_id("inventory_sample", model),
            "source":              "inventory_sample",
            "internal_id":         model,
            "description":         desc,
            "extended_description": ext or None,
            "manufacturer_name":   mfr,
            "manufacturer_abbrev": "",
            "model_number":        model,
            "product_category":    "Fuses",
            "uom":                 "EA",
            "currency":            "USD",
            "has_stock":           False,
            "total_qoh":           0,
            "location_count":      1,
            "min_cost":            None,
            "max_cost":            None,
            "avg_sell_price":      None,
            "locations":           [loc],
        })
    return records


_BRIGGS_MFR_MAP = {
    "GEB": "GERBER", "ZN": "ZURN",   "TS":   "T&S BRASS",
    "OLY": "OLYMPIA", "GED1": "GERBER",
    "CMI1": "GERBER", "CMI2": "GERBER", "CMI3": "GERBER",
}


def load_briggs_plumbing() -> list:
    df = pd.read_excel(os.path.join(DATA_DIR, "Plumbing Inventory_Briggs.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Prod"].astype(str).str.strip() != ""]
    records = []
    for _, r in df.iterrows():
        iid      = str(r["Prod"]).strip()
        desc     = str(r.get("Descrip 1", "") or "").strip()
        mfr_code = str(r.get("Prodline",   "") or "").strip()
        mfr_name = _BRIGGS_MFR_MAP.get(mfr_code, mfr_code)
        sell     = _to_float(r.get("Listprice"))
        loc = {
            "location_name": "Default", "location_erp_id": "0",
            "qoh": 0, "quantity_on_order": 0, "cost": None,
            "sell_price": sell, "reorder_decision": False,
            "stock_priority": None, "in_stock": False,
        }
        records.append({
            "id":                  make_id("briggs_plumbing", iid),
            "source":              "briggs_plumbing",
            "internal_id":         iid,
            "description":         desc,
            "manufacturer_name":   mfr_name,
            "manufacturer_abbrev": mfr_code,
            "model_number":        iid,
            "product_category":    "Plumbing",
            "uom":                 "EA",
            "currency":            "USD",
            "has_stock":           False,
            "total_qoh":           0,
            "location_count":      1,
            "min_cost":            None,
            "max_cost":            None,
            "avg_sell_price":      sell,
            "locations":           [loc],
        })
    return records


def load_plumbing_2() -> list:
    df = pd.read_excel(os.path.join(DATA_DIR, "Plumbing_Inventory_2.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Product Code"].astype(str).str.strip() != ""]
    records = []
    for _, r in df.iterrows():
        iid  = str(r["Product Code"]).strip()
        desc = str(r.get("Product Description", "") or "").strip()
        cost = _to_float(r.get("Cost"))
        sell = _to_float(r.get("List Price"))
        loc = {
            "location_name": "Default", "location_erp_id": "0",
            "qoh": 0, "quantity_on_order": 0, "cost": cost,
            "sell_price": sell, "reorder_decision": False,
            "stock_priority": None, "in_stock": False,
        }
        records.append({
            "id":                  make_id("plumbing_2", iid),
            "source":              "plumbing_2",
            "internal_id":         iid,
            "description":         desc,
            "manufacturer_name":   "",
            "manufacturer_abbrev": "",
            "model_number":        iid,
            "product_category":    "Plumbing",
            "uom":                 "EA",
            "currency":            "USD",
            "has_stock":           False,
            "total_qoh":           0,
            "location_count":      1,
            "min_cost":            cost,
            "max_cost":            cost,
            "avg_sell_price":      sell,
            "locations":           [loc],
        })
    return records


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

LOADERS = [
    load_au_parspec,
    load_briggs_plumbing,
    load_burnaby_dc,
    load_guillevin_1,
    load_guillevin_2,
    load_inventory_sample,
    load_plumbing,
    load_plumbing_2,
    load_standard_supply,
]


def load_all(verbose: bool = True, attach_caches: bool = True) -> list:
    """
    Load all inventory sources and optionally attach enrichment/taxonomy caches.

    Args:
        verbose:       Print per-source counts.
        attach_caches: If True, merge enrichment_cache, attributes_cache,
                       and taxonomy_cache into each record.

    Returns:
        Flat list of payload dicts ready for embedding and upsert.
    """
    records = []
    for fn in LOADERS:
        try:
            rs = fn()
            if verbose:
                print(f"  {fn.__name__:<24} {len(rs):>6} products")
            records.extend(rs)
        except FileNotFoundError:
            if verbose:
                print(f"  {fn.__name__:<24} SKIPPED (file not found)")
        except Exception as e:
            if verbose:
                print(f"  {fn.__name__:<24} ERROR: {e}")

    if verbose:
        print(f"\n  Total: {len(records)} products across {len(LOADERS)} sources")

    if attach_caches:
        _attach_caches(records)

    return records
