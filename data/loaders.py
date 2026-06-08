"""
Inventory source loaders.

Each loader reads one raw CSV/XLSX source, aggregates per-product rows into a
locations[] array, and returns a list of payload dicts. Two optional caches are
merged at load time when present:

    enrichment_cache.json   → extended_description (LLM-generated rich text)
    taxonomy_cache.json     → taxonomy_domain / taxonomy_category / taxonomy_subcategory
"""

import hashlib
import json
import os
import uuid
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from data.normalizers import normalize_manufacturer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = os.path.join(_REPO_ROOT, "inventory_data")

# ---------------------------------------------------------------------------
# Optional caches (gracefully absent on first run)
# ---------------------------------------------------------------------------

_enrichment_cache: dict = {}
_taxonomy_cache:   dict = {}
_caches_loaded = False


def _load_caches() -> None:
    global _enrichment_cache, _taxonomy_cache, _caches_loaded
    if _caches_loaded:
        return

    for attr, fname, label in [
        ("_enrichment_cache", "enrichment_cache.json",  "Enrichment"),
        ("_taxonomy_cache",   "taxonomy_cache.json",    "Taxonomy"),
    ]:
        path = os.path.join(_REPO_ROOT, fname)
        if os.path.exists(path):
            import sys
            with open(path) as f:
                globals()[attr] = json.load(f)
            print(f"  [INFO] {label} cache loaded: {len(globals()[attr])} entries",
                  file=sys.stderr)

    _caches_loaded = True


def _attach_caches(records: list) -> list:
    """Merge enrichment and taxonomy data into each record."""
    _load_caches()
    for rec in records:
        pid = rec["id"]
        rec["extended_description"] = _enrichment_cache.get(pid)
        tax = _taxonomy_cache.get(pid, {})
        rec["taxonomy_domain"]      = tax.get("taxonomy_domain",      "") or ""
        rec["taxonomy_category"]    = tax.get("taxonomy_category",    "") or ""
        rec["taxonomy_subcategory"] = tax.get("taxonomy_subcategory", "") or ""
    return records


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_id(src: str, iid: str) -> str:
    return str(uuid.UUID(hashlib.md5(f"{src}:{iid}".encode()).hexdigest()))


def _f(v):
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
            qoh = int(_f(r.get(qoh_col)) or 0)
            locations.append({
                "location_name":     str(r.get(loc_name_col, "") or "").strip() if loc_name_col else "",
                "location_erp_id":   str(r.get(loc_id_col,   "") or "").strip() if loc_id_col   else "",
                "qoh":               qoh,
                "quantity_on_order": int(_f(r.get(qoo_col)) or 0) if qoo_col else 0,
                "cost":              _f(r.get(cost_col))    if cost_col    else None,
                "sell_price":        _f(r.get(sell_col))    if sell_col    else None,
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


# ---------------------------------------------------------------------------
# Robust source-file reader
# ---------------------------------------------------------------------------
# Source files arrive inconsistently: the same logical source may be saved as
# .xlsx in one drop and .csv in the next, a file may be misnamed (an xlsx saved
# with a .csv extension), and CSVs come in mixed encodings (utf-8, cp1252,
# latin-1). Rather than hard-code one format per loader — which silently SKIPs a
# source the moment the extension changes — every loader reads through this
# helper. It resolves the file by either extension, detects the REAL format from
# the file's magic bytes (not the extension), and falls back across encodings.

_XLSX_MAGIC = b"PK\x03\x04"   # xlsx/xlsm are ZIP archives
_XLS_MAGIC  = b"\xd0\xcf\x11\xe0"  # legacy .xls (OLE2 compound document)


def _resolve_source_path(filename: str) -> str:
    """Return an existing path for `filename`, trying the other extension.

    Tries the name as given, then swaps .xlsx/.xls ↔ .csv. Returns the original
    candidate (so the caller raises a clear FileNotFoundError) if none exist.
    """
    cand = os.path.join(DATA_DIR, filename)
    if os.path.exists(cand):
        return cand
    base, ext = os.path.splitext(filename)
    alts = [".csv", ".xlsx", ".xls"]
    if ext.lower() in alts:
        alts.remove(ext.lower())
    for alt_ext in alts:
        alt = os.path.join(DATA_DIR, base + alt_ext)
        if os.path.exists(alt):
            return alt
    return cand


def _read_source(filename: str, **csv_kwargs):
    """Read an inventory source into a DataFrame, format- and encoding-agnostic.

    - Resolves the file by either extension (.xlsx/.xls/.csv).
    - Sniffs magic bytes so a misnamed file (e.g. an xlsx saved as .csv) is
      still parsed correctly.
    - For CSV, tries utf-8 → utf-8-sig → cp1252 → latin-1 (latin-1 decodes any
      byte, so reading always succeeds).

    `csv_kwargs` are the read_csv options the loader would normally pass
    (dtype, keep_default_na, low_memory); options unsupported by read_excel are
    dropped automatically when the file turns out to be a spreadsheet.
    """
    path = _resolve_source_path(filename)
    with open(path, "rb") as fh:
        head = fh.read(8)

    if head.startswith(_XLSX_MAGIC) or head.startswith(_XLS_MAGIC):
        # read_excel shares dtype/keep_default_na/header but not low_memory etc.
        excel_kwargs = {k: v for k, v in csv_kwargs.items()
                        if k in ("dtype", "keep_default_na", "header", "sheet_name")}
        return pd.read_excel(path, **excel_kwargs)

    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **csv_kwargs)
        except UnicodeDecodeError:
            continue
    # Should be unreachable (latin-1 never raises), but stay defensive.
    return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", **csv_kwargs)


# ---------------------------------------------------------------------------
# Individual source loaders
# ---------------------------------------------------------------------------

def load_au_parspec():
    df = _read_source(
        "AU Parspec inventory load 10032026 SEND AB V 1 Test copy for demo.csv",
        dtype=str, keep_default_na=False, low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate("au_parspec", "AUD", df,
                      id_col="Product ERP Code", mfr_col="Manufacturer Name",
                      mfr_abbrev_col="Manufacturer Abbreviation",
                      model_col="Model Number", desc_col="Description",
                      cat_col="Product Category", uom_col="Selling UOM",
                      loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
                      qoh_col="QOH", qoo_col="Quantity On Order",
                      cost_col="Product Cost", sell_col="Product Sell Price",
                      reorder_col="Reorder Decision", priority_col="Stock Priority")


def load_plumbing():
    df = _read_source("Plumbing Inventory example.xlsx", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[1:].copy()
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate("plumbing", "USD", df,
                      id_col="Product ERP Code", mfr_col="Manufacturer Name",
                      mfr_abbrev_col="Manufacturer Abbreviation",
                      model_col="Model Number", desc_col="Description",
                      cat_col="Product Category", uom_col="UOM",
                      loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
                      qoh_col="QOH", qoo_col="Quantity On Order",
                      cost_col="Product Cost", sell_col="Product Sell Price",
                      reorder_col="Reorder Decision", priority_col="Stock Priority")


def load_standard_supply():
    df = _read_source(
        "Standard Supply_inventory_data_1.csv",
        dtype=str, keep_default_na=False, low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    if "Stock Location ID" in df.columns:
        df = df.rename(columns={"Stock Location ID": "Stock Location ERP ID"})
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate("standard_supply", "USD", df,
                      id_col="Product ERP Code", mfr_col="Manufacturer Name",
                      mfr_abbrev_col="Manufacturer Abbreviation",
                      model_col="Model Number", desc_col="Description",
                      cat_col="Product Category", uom_col="UOM",
                      loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
                      qoh_col="QOH", qoo_col="Quantity on Order",
                      cost_col="Product Cost", sell_col="Product Sell Price",
                      reorder_col="Reorder Decision")


def load_guillevin_2():
    df = _read_source(
        "Guillevin_inventory_data_2.csv",
        dtype=str, keep_default_na=False, low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    if "Stock Location Id" in df.columns:
        df = df.rename(columns={"Stock Location Id": "Stock Location ERP ID"})
    if "Product Cost " in df.columns:
        df = df.rename(columns={"Product Cost ": "Product Cost"})
    df = df[df["Description"].astype(str).str.strip() != ""]
    df = df[df["Product ERP Code"].astype(str).str.strip() != ""]
    return _aggregate("guillevin_2", "CAD", df,
                      id_col="Product ERP Code", mfr_col="Manufacturer Name",
                      mfr_abbrev_col="Manufacturer Abbreviation",
                      model_col="Model Number", desc_col="Description",
                      cat_col="Product Category", uom_col="UOM",
                      loc_name_col="Stock Location Name", loc_id_col="Stock Location ERP ID",
                      qoh_col="QOH", qoo_col="Quantity On Order",
                      cost_col="Product Cost")


def load_guillevin_1():
    df = _read_source("Guillevin_inventory_data_1.xlsx", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Item Id"].astype(str).str.strip() != ""]
    return _aggregate("guillevin_1", "CAD", df,
                      id_col="Item Id", mfr_col="Name",
                      model_col="Supplier Part No", desc_col="Item Desc",
                      loc_name_col="Location Location Name", loc_id_col="Location Id",
                      qoh_col="Qty On Hand")


def load_burnaby_dc():
    df = _read_source("Burnaby DC Lighting Inventory 9 17.xlsx", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.reset_index(drop=True)
    df["__id"] = df["Item"].astype(str).str.strip() + "_" + df.index.astype(str)
    records = []
    for _, r in df.iterrows():
        iid = r["__id"]
        if not iid or iid.startswith("_"):
            continue
        qoh  = int(_f(r.get("On Hand")) or 0)
        cost = _f(r.get("Low Repl Cost"))
        loc  = {
            "location_name":     str(r.get("Branch name", "") or "").strip(),
            "location_erp_id":   str(r.get("Supplier Name", "") or "").strip(),
            "qoh": qoh, "quantity_on_order": 0, "cost": cost,
            "sell_price": None, "reorder_decision": False,
            "stock_priority": None, "in_stock": qoh > 0,
        }
        records.append({
            "id":                  make_id("burnaby_dc", iid),
            "source":              "burnaby_dc",
            "internal_id":         iid,
            "description":         str(r.get("Description", "") or "").strip(),
            "manufacturer_name":   normalize_manufacturer(
                                       str(r.get("Mfr", "") or "").strip()
                                       or str(r.get("Supplier Name", "") or "").strip()),
            "manufacturer_abbrev": "",
            "model_number":        str(r.get("Item", "") or "").strip(),
            "product_category":    "Lighting",
            "uom":                 str(r.get("UOM", "EA") or "EA").strip(),
            "currency":            "CAD",
            "has_stock":           qoh > 0, "total_qoh": qoh, "location_count": 1,
            "min_cost": cost, "max_cost": cost, "avg_sell_price": None,
            "locations": [loc],
        })
    return records


def load_inventory_sample():
    df = _read_source("INVENTORY SAMPLE.xlsx", dtype=str)
    df.columns = ["manufacturer_name", "model_number", "description", "extended_description"]
    records = []
    for _, r in df.iterrows():
        model = str(r.get("model_number", "") or "").strip()
        if not model or model.lower() == "nan":
            continue
        mfr  = normalize_manufacturer(str(r.get("manufacturer_name", "") or "").strip())
        desc = str(r.get("description", "") or "").strip()
        ext  = str(r.get("extended_description", "") or "").strip()
        if ext.lower() == "nan":
            ext = ""
        loc = {
            "location_name": "Default", "location_erp_id": "0", "qoh": 0,
            "quantity_on_order": 0, "cost": None, "sell_price": None,
            "reorder_decision": False, "stock_priority": None, "in_stock": False,
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
            "uom": "EA", "currency": "USD",
            "has_stock": False, "total_qoh": 0, "location_count": 1,
            "min_cost": None, "max_cost": None, "avg_sell_price": None,
            "locations": [loc],
        })
    return records


# Briggs Prodline (manufacturer code) -> full manufacturer name
_BRIGGS_MFR_MAP = {
    "GEB":  "GERBER",  "ZN":   "ZURN",   "TS":   "T&S BRASS",
    "OLY":  "OLYMPIA", "GED1": "GERBER",
    "CMI1": "GERBER",  "CMI2": "GERBER", "CMI3": "GERBER",
}


def load_briggs_plumbing():
    df = _read_source("Plumbing Inventory_Briggs.xlsx", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Prod"].astype(str).str.strip() != ""]
    records = []
    for _, r in df.iterrows():
        iid      = str(r["Prod"]).strip()
        desc     = str(r.get("Descrip 1", "") or "").strip()
        mfr_code = str(r.get("Prodline", "") or "").strip()
        mfr_name = _BRIGGS_MFR_MAP.get(mfr_code, mfr_code)
        sell     = _f(r.get("Listprice"))
        loc = {
            "location_name": "Default", "location_erp_id": "0",
            "qoh": 0, "quantity_on_order": 0, "cost": None,
            "sell_price": sell, "reorder_decision": False,
            "stock_priority": None, "in_stock": False,
        }
        records.append({
            "id":                  make_id("briggs_plumbing", iid),
            "source":              "briggs_plumbing",
            "internal_id":         iid, "description": desc,
            "manufacturer_name":   mfr_name, "manufacturer_abbrev": mfr_code,
            "model_number":        iid, "product_category": "Plumbing",
            "uom": "EA", "currency": "USD",
            "has_stock": False, "total_qoh": 0, "location_count": 1,
            "min_cost": None, "max_cost": None, "avg_sell_price": sell,
            "locations": [loc],
        })
    return records


def load_plumbing_2():
    df = _read_source("Plumbing_Inventory_2.xlsx", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Product Code"].astype(str).str.strip() != ""]
    records = []
    for _, r in df.iterrows():
        iid  = str(r["Product Code"]).strip()
        desc = str(r.get("Product Description", "") or "").strip()
        cost = _f(r.get("Cost"))
        sell = _f(r.get("List Price"))
        loc  = {
            "location_name": "Default", "location_erp_id": "0",
            "qoh": 0, "quantity_on_order": 0, "cost": cost,
            "sell_price": sell, "reorder_decision": False,
            "stock_priority": None, "in_stock": False,
        }
        records.append({
            "id":                  make_id("plumbing_2", iid),
            "source":              "plumbing_2",
            "internal_id":         iid, "description": desc,
            "manufacturer_name":   "", "manufacturer_abbrev": "",
            "model_number":        iid, "product_category": "Plumbing",
            "uom": "EA", "currency": "USD",
            "has_stock": False, "total_qoh": 0, "location_count": 1,
            "min_cost": cost, "max_cost": cost, "avg_sell_price": sell,
            "locations": [loc],
        })
    return records


# ---------------------------------------------------------------------------
# Public API
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
    Load all inventory sources and optionally attach enrichment / taxonomy caches.

    Args:
        verbose:       Print per-source counts.
        attach_caches: If True, merge enrichment_cache and taxonomy_cache into
                       each record (gracefully skipped when files are absent).

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
        except Exception as exc:
            if verbose:
                print(f"  {fn.__name__:<24} ERROR: {exc}")

    if verbose:
        print(f"\n  Total: {len(records)} products across {len(LOADERS)} sources")

    if attach_caches:
        _attach_caches(records)

    return records
