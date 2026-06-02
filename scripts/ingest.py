"""Universal/canonical re-ingest from the raw CSV/XLSX files into local_storage.

For every product, use the CSV's canonical ID column as internal_id:
  AU_Parspec, Plumbing, Standard_Supply, Guillevin_2 -> 'Product ERP Code'
  Guillevin_1                                       -> 'Item Id'
  Burnaby_DC                                        -> 'Item' + row index
  Inventory_Sample                                  -> 'Our Product' (model)

Use the CSV's raw Description as-is — no 'MFR MODEL' fallback overwrite.
Aggregate location rows per ID. One Qdrant point per unique product per source.
Re-embed dense (mpnet), sparse_model (BM25 over model variants), sparse_desc
(BM25 over normalize_specs(desc + mfr + cat)).

Wipes local_storage and rebuilds fresh."""

import os, sys, time, hashlib, uuid, shutil, warnings
warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
# Load .env so QDRANT_URL / QDRANT_API_KEY are available when targeting cloud
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_R, ".env"))
except ImportError:
    pass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, SparseVector, VectorParams, Distance,
    SparseVectorParams, SparseIndexParams, HnswConfigDiff, PayloadSchemaType,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
)
from data.normalizers import normalize_specs, normalize_manufacturer, model_number_variants
from models.embeddings import get_dense_model, get_bm25_model

# Source CSV/XLSX path: $INVENTORY_DATA env override, else repo-root/inventory_data/
INV   = os.environ.get("INVENTORY_DATA",
                       os.path.join(_R, "inventory_data"))
LOCAL = os.path.join(_R, "local_storage")
COL   = "inventory"


def _open_client():
    """Return (client, target_description). Target = cloud if QDRANT_URL set
    (production happy path), else embedded local store (dev override)."""
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key, check_compatibility=False,
                            timeout=120), f"cloud ({url})"
    # local fallback
    if os.path.isdir(LOCAL):
        shutil.rmtree(LOCAL)
    return QdrantClient(path=LOCAL), f"embedded local ({LOCAL})"


def _f(v):
    if v is None: return None
    if isinstance(v, float) and pd.isna(v): return None
    s = str(v).strip().replace("$", "").replace(",", "")
    if s == "" or s.lower() == "nan": return None
    try: return float(s)
    except: return None


def make_id(src, iid):
    return str(uuid.UUID(hashlib.md5(f"{src}:{iid}".encode()).hexdigest()))


def _aggregate(src, currency, df, *, id_col, mfr_col, model_col, desc_col,
               loc_name_col, loc_id_col, qoh_col,
               mfr_abbrev_col=None, cat_col=None, uom_col=None,
               qoo_col=None, cost_col=None, sell_col=None,
               reorder_col=None, priority_col=None, default_cat=""):
    records = []
    for iid, group in df.groupby(id_col, dropna=True):
        iid = str(iid).strip()
        if not iid or iid.lower() == "nan": continue
        first = group.iloc[0]
        locations = []
        for _, r in group.iterrows():
            qoh = int(_f(r.get(qoh_col)) or 0)
            locations.append({
                "location_name":  str(r.get(loc_name_col, "") or "").strip() if loc_name_col else "",
                "location_erp_id": str(r.get(loc_id_col, "") or "").strip() if loc_id_col else "",
                "qoh": qoh,
                "quantity_on_order": int(_f(r.get(qoo_col)) or 0) if qoo_col else 0,
                "cost": _f(r.get(cost_col)) if cost_col else None,
                "sell_price": _f(r.get(sell_col)) if sell_col else None,
                "reorder_decision": (str(r.get(reorder_col, "")).strip().lower() == "yes") if reorder_col else False,
                "stock_priority": (str(r.get(priority_col, "")).strip() or None) if priority_col else None,
                "in_stock": qoh > 0,
            })
        qohs  = [l["qoh"] for l in locations]
        costs = [l["cost"] for l in locations if l["cost"] is not None]
        sells = [l["sell_price"] for l in locations if l["sell_price"] is not None]
        total_qoh = sum(qohs)
        rec = {
            "id":   make_id(src, iid),
            "source": src,
            "internal_id":  iid,
            "description":  str(first.get(desc_col, "") or "").strip(),
            "manufacturer_name":  normalize_manufacturer(str(first.get(mfr_col, "") or "").strip()),
            "manufacturer_abbrev": str(first.get(mfr_abbrev_col, "") or "").strip() if mfr_abbrev_col else "",
            "model_number": str(first.get(model_col, "") or "").strip(),
            "product_category": (str(first.get(cat_col, "") or "").strip() if cat_col else "") or default_cat,
            "uom": (str(first.get(uom_col, "") or "").strip() or "EA") if uom_col else "EA",
            "currency": currency,
            "has_stock": total_qoh > 0,
            "total_qoh": total_qoh,
            "location_count": len(locations),
            "min_cost": round(min(costs), 4) if costs else None,
            "max_cost": round(max(costs), 4) if costs else None,
            "avg_sell_price": round(sum(sells)/len(sells), 4) if sells else None,
            "locations": locations,
        }
        records.append(rec)
    return records


def load_au_parspec():
    df = pd.read_csv(os.path.join(INV, "AU Parspec inventory load 10032026 SEND AB V 1 Test copy for demo.csv"),
                     dtype=str, keep_default_na=False, low_memory=False)
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
    df = pd.read_excel(os.path.join(INV, "Plumbing Inventory example.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[1:].copy()  # skip the doc/header row
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
    df = pd.read_csv(os.path.join(INV, "Standard Supply_inventory_data_1.csv"),
                     dtype=str, keep_default_na=False, low_memory=False)
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
    df = pd.read_csv(os.path.join(INV, "Guillevin_inventory_data_2_utf8.csv"),
                     dtype=str, keep_default_na=False, low_memory=False)
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
    df = pd.read_excel(os.path.join(INV, "Guillevin_inventory_data_1.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Item Id"].astype(str).str.strip() != ""]
    return _aggregate("guillevin_1", "CAD", df,
                      id_col="Item Id", mfr_col="Name",
                      model_col="Supplier Part No", desc_col="Item Desc",
                      loc_name_col="Location Location Name", loc_id_col="Location Id",
                      qoh_col="Qty On Hand")


def load_burnaby_dc():
    df = pd.read_excel(os.path.join(INV, "Burnaby DC Lighting Inventory 9 17.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.reset_index(drop=True)
    df["__id"] = df["Item"].astype(str).str.strip() + "_" + df.index.astype(str)
    records = []
    for _, r in df.iterrows():
        iid = r["__id"]
        if not iid or iid.startswith("_"): continue
        qoh = int(_f(r.get("On Hand")) or 0)
        cost = _f(r.get("Low Repl Cost"))
        loc = {"location_name": str(r.get("Branch name", "") or "").strip(),
               "location_erp_id": str(r.get("Supplier Name", "") or "").strip(),
               "qoh": qoh, "quantity_on_order": 0, "cost": cost,
               "sell_price": None, "reorder_decision": False,
               "stock_priority": None, "in_stock": qoh > 0}
        records.append({
            "id": make_id("burnaby_dc", iid), "source": "burnaby_dc", "internal_id": iid,
            "description": str(r.get("Description", "") or "").strip(),
            "manufacturer_name": normalize_manufacturer(str(r.get("Mfr", "") or "").strip() or str(r.get("Supplier Name", "") or "").strip()),
            "manufacturer_abbrev": "",
            "model_number": str(r.get("Item", "") or "").strip(),
            "product_category": "Lighting", "uom": str(r.get("UOM", "EA") or "EA").strip(),
            "currency": "CAD",
            "has_stock": qoh > 0, "total_qoh": qoh, "location_count": 1,
            "min_cost": cost, "max_cost": cost, "avg_sell_price": None,
            "locations": [loc],
        })
    return records


def load_inventory_sample():
    df = pd.read_excel(os.path.join(INV, "INVENTORY SAMPLE.xlsx"), dtype=str)
    df.columns = ["manufacturer_name", "model_number", "description", "extended_description"]
    records = []
    for _, r in df.iterrows():
        model = str(r.get("model_number", "") or "").strip()
        if not model or model.lower() == "nan": continue
        mfr = normalize_manufacturer(str(r.get("manufacturer_name", "") or "").strip())
        desc = str(r.get("description", "") or "").strip()
        ext = str(r.get("extended_description", "") or "").strip()
        if ext.lower() == "nan": ext = ""
        loc = {"location_name": "Default", "location_erp_id": "0", "qoh": 0,
               "quantity_on_order": 0, "cost": None, "sell_price": None,
               "reorder_decision": False, "stock_priority": None, "in_stock": False}
        records.append({
            "id": make_id("inventory_sample", model), "source": "inventory_sample",
            "internal_id": model, "description": desc,
            "extended_description": ext or None,
            "manufacturer_name": mfr, "manufacturer_abbrev": "",
            "model_number": model, "product_category": "Fuses",
            "uom": "EA", "currency": "USD",
            "has_stock": False, "total_qoh": 0, "location_count": 1,
            "min_cost": None, "max_cost": None, "avg_sell_price": None,
            "locations": [loc],
        })
    return records


# Briggs Prodline (manufacturer code) -> full manufacturer name. Triangulated
# from product descriptions per code: GEB -> Gerber fixtures, ZN -> Zurn
# vacuum breakers/RPZ, TS -> T&S Brass pre-rinse, OLY -> Olympia, GED1 ->
# Gerber Antioch/Northerly product lines, CMI1/2/3 -> Gerber Majestic/Delano/
# Oak Lawn (low confidence on CMI3 stainless sinks). Raw Prodline code is
# preserved in `manufacturer_abbrev` for audit / rollback.
_BRIGGS_MFR_MAP = {
    "GEB":  "GERBER",   "ZN":   "ZURN",       "TS":   "T&S BRASS",
    "OLY":  "OLYMPIA",  "GED1": "GERBER",
    "CMI1": "GERBER",   "CMI2": "GERBER",     "CMI3": "GERBER",
}


def load_briggs_plumbing():
    df = pd.read_excel(os.path.join(INV, "Plumbing Inventory_Briggs.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Prod"].astype(str).str.strip() != ""]
    records = []
    for _, r in df.iterrows():
        iid = str(r["Prod"]).strip()
        desc = str(r.get("Descrip 1", "") or "").strip()
        mfr_code = str(r.get("Prodline", "") or "").strip()
        mfr_name = _BRIGGS_MFR_MAP.get(mfr_code, mfr_code)  # fall back to raw code if unmapped
        sell = _f(r.get("Listprice"))
        loc = {"location_name": "Default", "location_erp_id": "0",
               "qoh": 0, "quantity_on_order": 0, "cost": None,
               "sell_price": sell, "reorder_decision": False,
               "stock_priority": None, "in_stock": False}
        records.append({
            "id": make_id("briggs_plumbing", iid), "source": "briggs_plumbing",
            "internal_id": iid, "description": desc,
            "manufacturer_name": mfr_name, "manufacturer_abbrev": mfr_code,
            "model_number": iid, "product_category": "Plumbing",
            "uom": "EA", "currency": "USD",
            "has_stock": False, "total_qoh": 0, "location_count": 1,
            "min_cost": None, "max_cost": None, "avg_sell_price": sell,
            "locations": [loc],
        })
    return records


def load_plumbing_2():
    df = pd.read_excel(os.path.join(INV, "Plumbing_Inventory_2.xlsx"), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Product Code"].astype(str).str.strip() != ""]
    records = []
    for _, r in df.iterrows():
        iid = str(r["Product Code"]).strip()
        desc = str(r.get("Product Description", "") or "").strip()
        cost = _f(r.get("Cost"))
        sell = _f(r.get("List Price"))
        loc = {"location_name": "Default", "location_erp_id": "0",
               "qoh": 0, "quantity_on_order": 0, "cost": cost,
               "sell_price": sell, "reorder_decision": False,
               "stock_priority": None, "in_stock": False}
        records.append({
            "id": make_id("plumbing_2", iid), "source": "plumbing_2",
            "internal_id": iid, "description": desc,
            "manufacturer_name": "", "manufacturer_abbrev": "",
            "model_number": iid, "product_category": "Plumbing",
            "uom": "EA", "currency": "USD",
            "has_stock": False, "total_qoh": 0, "location_count": 1,
            "min_cost": cost, "max_cost": cost, "avg_sell_price": sell,
            "locations": [loc],
        })
    return records


def main():
    print("Loading all source files…")
    records = []
    for fn in [load_au_parspec, load_briggs_plumbing, load_burnaby_dc,
               load_guillevin_1, load_guillevin_2, load_inventory_sample,
               load_plumbing, load_plumbing_2, load_standard_supply]:
        t = time.time(); rs = fn()
        print(f"  {fn.__name__:<24} {len(rs):>6} products  ({time.time()-t:.0f}s)")
        records.extend(rs)
    print(f"\nTotal unique products: {len(records)}\n")

    # Texts
    dense_texts, sm_texts, sd_texts = [], [], []
    for r in records:
        d = " ".join(x for x in [r.get("description") or "", r.get("manufacturer_name") or "",
                                   r.get("product_category") or "", r.get("model_number") or ""] if x).strip()
        dense_texts.append(d or r["internal_id"])
        sm = model_number_variants(r["model_number"]) or r["model_number"] or r["internal_id"]
        sm_texts.append(sm)
        sd = normalize_specs(" ".join(x for x in [r.get("description") or "", r.get("manufacturer_name") or "",
                                                    r.get("product_category") or ""] if x))
        sd_texts.append(sd or r.get("description") or r["internal_id"])

    dense_model = get_dense_model()
    bm25 = get_bm25_model()

    print("Embedding dense (mpnet, ~35k)…")
    t0 = time.time()
    dense_vecs = []
    for i, t in enumerate(dense_texts):
        dense_vecs.append(dense_model.encode(t, normalize_embeddings=True).tolist())
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(dense_texts)}  ({(i+1)/(time.time()-t0):.1f}/s, {time.time()-t0:.0f}s)")
    print(f"  Dense done in {time.time()-t0:.0f}s")

    print("Embedding sparse_model (BM25)…")
    sm_vecs = [SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
               for r in bm25.embed(sm_texts)]
    print("Embedding sparse_desc (BM25)…")
    sd_vecs = [SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
               for r in bm25.embed(sd_texts)]

    client, target = _open_client()
    print(f"\nTarget: {target}")

    # Re-create collection from scratch (wipes existing data in the target).
    try:
        client.delete_collection(COL)
        print(f"  deleted existing collection '{COL}'")
    except Exception:
        pass
    client.create_collection(
        collection_name=COL,
        # Dense vectors use int8 scalar quantization: the quantized copy is
        # pinned in RAM (always_ram) while the original float32 vectors live on
        # disk (on_disk=True). This is the cost lever -- it shrinks the in-RAM
        # working set ~4x so the cluster can stay on a smaller RAM tier as the
        # catalog scales. Rescore (enabled query-side in core/search.py) reads
        # the on-disk originals to recover accuracy. Binary quantization is NOT
        # used: mpnet is 768d, below the ~1024d threshold where binary holds up.
        vectors_config={"dense": VectorParams(
            size=768, distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, quantile=0.99, always_ram=True,
                )
            ),
            on_disk=True,
        )},
        sparse_vectors_config={
            "sparse_model": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            "sparse_desc":  SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    for fld, sch in [("source", PayloadSchemaType.KEYWORD),
                      ("manufacturer_name", PayloadSchemaType.KEYWORD),
                      ("product_category", PayloadSchemaType.KEYWORD),
                      ("currency", PayloadSchemaType.KEYWORD),
                      ("has_stock", PayloadSchemaType.BOOL)]:
        client.create_payload_index(COL, fld, field_schema=sch)

    print("Upserting…")
    BATCH = 256
    t0 = time.time()
    for i in range(0, len(records), BATCH):
        batch = records[i:i+BATCH]
        points = [PointStruct(
            id=r["id"],
            vector={"dense": dense_vecs[i+j],
                    "sparse_model": sm_vecs[i+j],
                    "sparse_desc":  sd_vecs[i+j]},
            payload={k: v for k, v in r.items() if k != "id"},
        ) for j, r in enumerate(batch)]
        client.upsert(COL, points=points, wait=(i + BATCH >= len(records)))
        print(f"\r  {min(i+BATCH, len(records))}/{len(records)} ", end="", flush=True)

    n = client.count(COL).count
    print(f"\n\nDone. Collection rebuilt with {n} points in {time.time()-t0:.0f}s.")
    client.close()


if __name__ == "__main__":
    main()
