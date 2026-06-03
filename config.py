"""
Central configuration for the inventory search system.
All constants, paths, and tuning parameters live here.
"""

import os

# Load .env file when present (OPENROUTER_API_KEY, QDRANT_URL, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", None)
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "")
COLLECTION_NAME   = "inventory"
DENSE_DIM         = 768  # all-mpnet-base-v2 output dimension

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

DENSE_MODEL_NAME    = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_MODEL_NAME     = "Qdrant/bm25"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw inventory CSV/XLSX files go here (gitignored)
DATA_DIR = os.path.join(REPO_ROOT, "inventory_data")

# Generated cache files — rebuild using scripts in scripts/
ENRICHMENT_CACHE_PATH    = os.path.join(REPO_ROOT, "enrichment_cache.json")
TAXONOMY_EMBEDDINGS_PATH = os.path.join(REPO_ROOT, "taxonomy_embeddings.json")
TAXONOMY_CACHE_PATH      = os.path.join(REPO_ROOT, "taxonomy_cache.json")

# ---------------------------------------------------------------------------
# Retrieval tuning: prefetch limits per query type
# ---------------------------------------------------------------------------
# dense        -- semantic embedding (all-mpnet, int8-quantized)
# sparse_model -- BM25 over model number variants
# sparse_desc  -- BM25 over normalized description + spec attribute anchors
# ---------------------------------------------------------------------------

PREFETCH_LIMITS = {
    # model_number: sparse_model is the primary signal. Dense kept small for
    # tokenization variants. sparse_desc excluded — product family siblings
    # share descriptions and would push incorrect siblings to rank 1.
    "model_number": {"dense": 10, "sparse_model": 80, "sparse_desc": 0},
    "technical":    {"dense": 50, "sparse_model": 50, "sparse_desc": 40},
    "descriptive":  {"dense": 20, "sparse_model": 50, "sparse_desc": 80},
    # Single fixed profile used when the classifier is bypassed. Balanced
    # across all three channels — A/B across two held-out 300-query eval sets
    # showed this matches/beats per-type routing while removing ~600ms latency
    # and the OpenRouter dependency.
    "default":      {"dense": 50, "sparse_model": 50, "sparse_desc": 40},
}

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------
# When USE_CLASSIFIER=False (the default), every query uses DEFAULT_PROFILE
# and no OpenRouter call is made. Set USE_CLASSIFIER=True to restore per-type
# routing via Gemini 2.5 Flash.
# ---------------------------------------------------------------------------

USE_CLASSIFIER  = False
DEFAULT_PROFILE = "default"

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

INGEST_BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Product taxonomy — domain → category → [subcategories]
# Used by models/query_taxonomy_llm.py (query side) and
# scripts/build_taxonomy_embeddings.py / build_taxonomy_from_descriptions.py
# (ingest side).
# ---------------------------------------------------------------------------

PRODUCT_TAXONOMY = {
    "Mechanical": {
        "Air handling units & packaged equipment": [
            "Rooftop units", "Indoor AHUs", "DOAS units",
            "Make-up air units", "Split systems",
        ],
        "Chillers": [
            "Air-cooled scroll", "Air-cooled screw",
            "Water-cooled centrifugal", "Water-cooled screw", "Absorption",
        ],
        "Boilers & heating plant": [
            "Condensing gas boilers", "Non-condensing boilers",
            "Electric boilers", "Steam boilers", "Heat exchangers",
        ],
        "Terminal units & zone equipment": [
            "VAV boxes", "Fan-coil units", "Unit heaters", "Cabinet heaters",
            "Chilled beams", "VRF indoor units", "Mini-splits",
        ],
        "Fans & air moving devices": [
            "Centrifugal fans", "Axial fans", "Inline fans", "Exhaust fans",
            "Ceiling fans", "Jet fans", "Energy recovery ventilators",
        ],
        "Pumps & hydronic accessories": [
            "Base-mounted end suction", "Inline circulators", "Vertical turbine",
            "Condensate pumps", "Expansion tanks", "Air separators",
        ],
        "Ductwork & air distribution accessories": [
            "Sheet metal duct", "Flex duct", "Fiberglass duct board",
            "Diffusers", "Registers/grilles", "Dampers", "Sound attenuators",
        ],
        "Piping, valves & hydronic specialties": [
            "Steel pipe", "Copper pipe", "PVC/CPVC", "Grooved fittings",
            "Gate/globe/ball valves", "Check valves", "Strainers", "Flow meters",
        ],
        "Fire suppression equipment": [
            "Sprinkler heads", "Fire pumps", "Standpipe valves",
            "Backflow preventers", "Clean agent systems", "Pre-action/deluge valves",
        ],
        "Fire detection & alarm devices": [
            "Smoke detectors", "Heat detectors", "Duct detectors", "Pull stations",
            "Notification appliances", "FACP", "Aspirating detection",
        ],
        "Controls, sensors & actuators": [
            "DDC controllers", "Temperature sensors", "Pressure transducers",
            "CO2/IAQ sensors", "Humidity sensors", "Control valves",
            "Damper actuators", "VFDs",
        ],
        "Vibration isolation & seismic restraints": [
            "Spring isolators", "Rubber mounts", "Inertia bases",
            "Seismic snubbers", "Flexible connectors", "Pipe restraints",
        ],
    },
    "Electrical": {
        "Switchgear, switchboards & panelboards": [
            "MV switchgear", "LV switchboards", "Panelboards", "MCCs", "Busway",
        ],
        "Transformers": [
            "Dry-type LV", "Medium voltage", "K-rated", "Buck-boost", "Isolation",
        ],
        "Generators & emergency power": [
            "Diesel generators", "Natural gas generators", "Bi-fuel",
            "Portable", "Paralleling switchgear",
        ],
        "UPS & battery systems": [
            "Online double-conversion", "Line-interactive",
            "Lithium-ion BESS", "Lead-acid VRLA", "Nickel-cadmium",
        ],
        "Luminaires & lighting controls": [
            "LED troffers", "Downlights", "Linear fixtures", "High-bay",
            "Exterior", "Emergency/exit", "Occupancy sensors", "Dimmers",
            "Lighting control panels",
        ],
        "Conductors, cable & raceways": [
            "Building wire (THHN/XHHW)", "MC cable", "MI cable",
            "Fire alarm cable", "Tray cable", "Conduit (EMT/RGS/PVC)",
            "Cable tray", "Wireway",
        ],
        "Low-voltage & structured cabling": [
            "Cat6/6A UTP/STP", "Fiber optic (SM/MM)", "Patch panels",
            "Racks/cabinets", "Wireless APs", "DAS/BDA",
        ],
        "Security, access control & AV": [
            "IP cameras", "Access control panels", "Card readers",
            "Intercoms", "Intrusion detection", "AV displays", "Speakers",
        ],
        "Grounding, lightning & surge protection": [
            "Ground rods", "Ground bars", "Grounding conductors",
            "Air terminals", "Down conductors", "SPDs",
        ],
        "Solar PV, EV charging & renewables": [
            "PV modules", "Inverters", "Racking", "Combiner boxes",
            "EV chargers (L2/DCFC)", "Wind turbines", "Energy storage",
        ],
    },
    "Plumbing": {
        "Plumbing fixtures & fittings": [
            "Water closets", "Lavatories", "Sinks", "Urinals", "Showers",
            "Drinking fountains", "Faucets", "Flush valves",
        ],
        "Domestic water heaters": [
            "Gas storage", "Gas tankless", "Electric storage",
            "Electric tankless", "Heat pump", "Solar thermal", "Semi-instantaneous",
        ],
        "Pumps (domestic, sewage, sump)": [
            "Booster pumps", "Recirculation pumps", "Sewage ejectors",
            "Sump pumps", "Grinder pumps", "Condensate pumps",
        ],
        "Pipe, fittings & valves — domestic water": [
            "Copper (Type K/L/M)", "PEX", "CPVC", "SS", "Fittings",
            "Ball/gate/check valves", "PRVs", "Backflow preventers", "TMVs",
        ],
        "Sanitary drainage pipe, fittings & accessories": [
            "Cast iron (hub/no-hub)", "PVC DWV", "ABS", "HDPE",
            "Cleanouts", "Floor drains", "Trench drains", "Interceptors",
        ],
        "Stormwater management": [
            "Roof drains", "Overflow drains", "Leaders/downspouts",
            "Siphonic systems", "Retention/detention", "Rainwater harvesting",
        ],
        "Natural gas piping & components": [
            "Black steel pipe", "CSST", "Gas valves", "Regulators",
            "Gas meters", "Flex connectors", "Seismic shutoffs",
        ],
        "Water treatment equipment": [
            "Water softeners", "RO systems", "UV disinfection",
            "Filtration", "Chemical treatment", "Deionization",
        ],
        "Medical gas systems": [
            "Oxygen", "Medical air", "Vacuum", "Nitrous oxide", "Nitrogen",
            "WAGD", "Zone valves", "Alarms", "Outlets",
        ],
        "Specialty plumbing": [
            "Pool pumps", "Pool filters", "Chemical feeders", "Air compressors",
            "Air dryers", "Irrigation controllers", "Greywater systems",
        ],
    },
}
