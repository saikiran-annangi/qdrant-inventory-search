"""Controlled-vocabulary product taxonomy — the SINGLE SOURCE OF TRUTH.

This module is the one place the taxonomy is defined. Everything else derives
from it:

  - scripts/build_taxonomy_embeddings.py      embeds ONLY these nodes
  - scripts/build_taxonomy_from_descriptions  classifies products into these nodes
  - models/query_taxonomy_llm.py              offers the query classifier ONLY these labels
  - core/search.py apply_taxonomy_boost        matches query label ↔ product label

Two structures:

  PRODUCT_TAXONOMY  {domain: {category: [subcategories]}}
      The clean 3-level vocabulary, grounded in the real distributor ERP
      categories (not an idealized project-spec taxonomy).

  CATEGORY_MAP      {erp_category_string: (domain, category, subcategory)}
      Deterministic map from a product's ERP `product_category` to a taxonomy
      node, so products that already carry a category are labeled WITHOUT
      embedding guesswork. Ambiguous business buckets (PREFAB, Premium
      Products, …) map to (None, None, None) → description-based fallback.

Domains: Electrical, Mechanical, Plumbing, Tools & Site.

Invariant (enforced by tests/test_taxonomy_consistency.py): every non-None
CATEGORY_MAP target must exist in PRODUCT_TAXONOMY. Keep them in sync.
"""

# ============================================================================
# PRODUCT_TAXONOMY
# ============================================================================

PRODUCT_TAXONOMY = {

    "Electrical": {

        # --- Circuit protection & power distribution ---
        "Circuit Protection & Distribution": [
            "MCBs & RCBOs",
            "RCDs & Safety Switches",
            "Main Switches & Isolators",
            "Fuses & Fuse Carriers",
            "Surge Protection & UPS",
            "Busbars & Chassis",
            "Switchboards & Distribution Boards",
            "Power Factor Correction",
        ],

        # --- Wiring devices ---
        "Wiring Devices": [
            "Light Switches & Mechanisms",
            "Power Outlets & GPOs",
            "Weatherproof Outlets & Switches",
            "Cover Plates & Wall Plates",
            "Dimmers & Fan Controllers",
            "Plugs & Sockets",
            "Industrial Plugs & Sockets",
            "Mounting Blocks & Brackets",
        ],

        # --- Conduit & conduit fittings ---
        "Conduit & Fittings": [
            "Rigid PVC Conduit",
            "Corrugated & Flexible Conduit",
            "Metal & Industrial Conduit",
            "Conduit Fittings & Adaptors",
            "Conduit Bends",
            "Conduit Saddles & Clips",
            "Junction Boxes",
        ],

        # --- Cable & wire ---
        "Cable & Wire": [
            "Building & Earth Wire",
            "TPS & Flat Cable",
            "XLPE & Power Cable",
            "Flexible & Cord Cable",
            "Control & Instrumentation Cable",
            "Fire Rated Cable",
            "Catenary & Aerial Cable",
            "Specialty & Automotive Cable",
        ],

        # --- Cable accessories ---
        "Cable Accessories": [
            "Cable Lugs & Links",
            "Terminals & Terminal Blocks",
            "Cable Glands",
            "Heatshrink & Coldshrink",
            "Cable Ties & Clips",
            "Cable Marking & Identification",
            "Cable Jointing & Connectors",
            "Tapes & Insulating Materials",
        ],

        # --- Cable management / containment ---
        "Cable Management": [
            "Cable Tray",
            "Cable Ladder",
            "Channel & Strut",
            "Cable Duct & Trunking",
            "Cable Cover & Protection",
            "Threaded Rod & Fixings",
        ],

        # --- Enclosures & boxes ---
        "Enclosures & Boxes": [
            "Metal Enclosures",
            "Polymer & PVC Enclosures",
            "Special & Custom Enclosures",
            "Wall & Mounting Boxes",
            "Enclosure Accessories",
        ],

        # --- Lighting luminaires ---
        "Lighting": [
            "LED Downlights",
            "LED Panels & Troffers",
            "LED Battens",
            "LED Floodlights",
            "LED Highbay & Lowbay",
            "LED Strip & Extrusion",
            "Exterior & Wall Lighting",
            "Spotlights & Track Lighting",
            "Oyster & Bulkhead Lights",
        ],

        # --- Emergency & exit lighting ---
        "Emergency Lighting": [
            "Emergency Exit Lights",
            "Emergency Battens & Bulkheads",
            "Emergency Battery Packs & Spares",
            "Emergency Test Devices",
        ],

        # --- Lamps & lampholders ---
        "Lamps & Lampholders": [
            "LED Lamps",
            "Fluorescent Lamps",
            "Halogen Lamps",
            "HID Lamps",
            "Incandescent & Specialty Lamps",
            "Lampholders & Starters",
            "Drivers & Ballasts",
        ],

        # --- Fans & ventilation ---
        "Fans & Ventilation": [
            "Exhaust Fans",
            "Ceiling Sweep Fans",
            "Wall & Inline Fans",
            "Fan Light & Heat Units",
            "Fan Controllers & Speed Control",
            "Louvres, Grills & Mounting",
        ],

        # --- Data, comms & AV ---
        "Data & Communications": [
            "Copper Data Cable",
            "Fibre Optic Cable & Leads",
            "Patch Leads & Patch Panels",
            "Data Outlets & Wall Plates",
            "Racks, Cabinets & Accessories",
            "Telephone & Intercom",
            "Audio Visual & Antennas",
            "Networking Hubs & Switches",
        ],

        # --- Control & automation ---
        "Control & Automation": [
            "Contactors & Overloads",
            "Relays & Timers",
            "Push Buttons & Indicators",
            "Motor Control & Soft Starters",
            "Variable Speed Drives",
            "Signalling Devices",
            "Smart & Home Automation",
        ],

        # --- Sensors & detection ---
        "Sensors & Detection": [
            "PIR & Motion Sensors",
            "Smoke & Fire Detectors",
            "Proximity & Special Sensors",
            "Sunset & Daylight Switches",
            "Cameras & Security Kits",
        ],

        # --- Metering & monitoring ---
        "Metering & Monitoring": [
            "Power Meters & Monitoring",
            "Metering Equipment",
            "Distribution Frames & Boxes",
        ],

        # --- Earthing & grounding ---
        "Earthing & Grounding": [
            "Earth Grounding Equipment",
            "Earth Bars & Links",
        ],

        # --- Solar & EV ---
        "Solar & EV": [
            "Photovoltaic Panels",
            "Inverters",
            "Solar Cable & Connectors",
            "DC Isolators",
            "Solar Batteries & Storage",
            "EV Charging Stations",
        ],

        # --- Power supplies & transformers ---
        "Power Supplies & Transformers": [
            "Transformers & Power Supplies",
            "Batteries & Chargers",
            "Capacitors & Chokes",
        ],

        # --- Electric heating ---
        "Electric Heating": [
            "Floor & Panel Heating",
            "Radiant & Space Heating",
            "Heat Lamps",
        ],

        # --- Underground & MV distribution ---
        "Network & Underground": [
            "Underground Distribution Equipment",
            "Pillars & Pits",
            "Bushings & Insulators",
        ],
    },

    "Mechanical": {

        # --- Air conditioning ---
        "Air Conditioning": [
            "Split System Air Conditioning",
            "Air Conditioning Accessories",
        ],

        # --- Ducting (mechanical air) ---
        "Ducting & Ventilation": [
            "Metal Duct",
            "Flexible Duct & Hose",
        ],

        # --- Appliances / whitegoods ---
        "Appliances": [
            "Cooking Appliances & Ranges",
            "Hand Dryers",
            "Catering Equipment",
        ],
    },

    "Plumbing": {

        # --- Hot water ---
        "Hot Water": [
            "Hot Water Elements",
            "Water Heating & Boiling Units",
            "Water Coolers",
        ],

        # --- Pipe & hose ---
        "Pipe & Hose": [
            "Flexible Hose & Fittings",
            "Hose Clamps & Straps",
        ],

        # --- General plumbing fittings ---
        "Plumbing Fittings": [
            "Valves & Connectors",
            "Pipe Fittings & Adaptors",
        ],
    },

    "Tools & Site": {

        # --- Drilling & cutting ---
        "Drilling & Cutting": [
            "Drill Bits & Drilling",
            "Holesaws",
            "Saws & Blades",
            "Cutting & Grinding Wheels",
            "Bending Springs",
        ],

        # --- Hand tools ---
        "Hand Tools": [
            "Cutters & Pliers",
            "Screwdrivers",
            "Crimp & Cable Tie Tools",
            "Stripping Tools",
            "Spanners & Wrenches",
            "Hex Bits & Keys",
            "Knives & Files",
            "Socket Sets",
        ],

        # --- Power tools ---
        "Power Tools": [
            "Power Tools",
            "Power Tool Batteries & Chargers",
            "Soldering & Heat Tools",
            "Caulking & Glue Guns",
        ],

        # --- Test & measurement ---
        "Test & Measurement": [
            "Multimeters & Clamp Meters",
            "Insulation & RCD Testers",
            "Voltage Detectors & Indicators",
            "Measuring & Thermometers",
            "Fibre & Communication Testers",
            "Test Leads",
        ],

        # --- Safety & PPE ---
        "Safety & PPE": [
            "Gloves & Hand Protection",
            "Eye & Face Protection",
            "Hearing Protection",
            "Respirators & Masks",
            "Lockout Equipment",
            "Worksite Safety",
            "Fire Blankets & Barriers",
        ],

        # --- Marking & labelling ---
        "Marking & Labelling": [
            "Labels & Signs",
            "Marking & Paint",
            "Labelling Systems",
            "Warning Tape",
            "Appliance Test Tags",
        ],

        # --- Adhesives & chemicals ---
        "Adhesives & Chemicals": [
            "Silicone & Sealants",
            "Adhesives",
            "Chemical Sprays & Aerosols",
            "Lubricants",
            "Fire Sealants & Barriers",
        ],

        # --- Fasteners & fixings ---
        "Fasteners & Fixings": [
            "Fasteners & Fixings",
            "Washers & Lock Nuts",
            "Straps & Clips",
            "Mounting Brackets & Systems",
        ],

        # --- Access & storage ---
        "Access & Storage": [
            "Ladders",
            "Storage Solutions & Tool Boxes",
            "Cases",
            "Cable Pulling Equipment",
        ],
    },
}


# ============================================================================
# CATEGORY_MAP — exact ERP category string → (domain, category, subcategory)
# (None, None, None) => ambiguous business bucket → description fallback
# ============================================================================

CATEGORY_MAP = {

    # ----- Ambiguous / non-product business buckets (fallback) -----
    "PREFAB": (None, None, None),
    "Premium Products": (None, None, None),
    "EQGCOM": (None, None, None),
    "Indent Industry Automation": (None, None, None),
    "Other Building Installation": (None, None, None),
    "Other Cable": (None, None, None),
    "Other Conduit": (None, None, None),
    "Other Audio Visual": (None, None, None),
    "Other Energy Conversion & Storage": (None, None, None),
    "Solar Other": (None, None, None),
    "Orange Circular": (None, None, None),
    "Medical": (None, None, None),

    # ----- Electrical: Circuit Protection & Distribution -----
    "Circuit Protection & Distribution - Res + Com": ("Electrical", "Circuit Protection & Distribution", "MCBs & RCBOs"),
    "Circuit Protection & Distribution - Industrial": ("Electrical", "Circuit Protection & Distribution", "MCBs & RCBOs"),
    "Main Switches & Isolators": ("Electrical", "Circuit Protection & Distribution", "Main Switches & Isolators"),
    "Isolators": ("Electrical", "Circuit Protection & Distribution", "Main Switches & Isolators"),
    "Fuses & Carriers": ("Electrical", "Circuit Protection & Distribution", "Fuses & Fuse Carriers"),
    "Surge Arrestors & Diverters": ("Electrical", "Circuit Protection & Distribution", "Surge Protection & UPS"),
    "UPS & Protection": ("Electrical", "Circuit Protection & Distribution", "Surge Protection & UPS"),
    "Busbar": ("Electrical", "Circuit Protection & Distribution", "Busbars & Chassis"),
    "Power Factor Correction": ("Electrical", "Circuit Protection & Distribution", "Power Factor Correction"),
    "Switches Rotary": ("Electrical", "Circuit Protection & Distribution", "Main Switches & Isolators"),

    # ----- Electrical: Wiring Devices -----
    "Domestic Switches": ("Electrical", "Wiring Devices", "Light Switches & Mechanisms"),
    "Switch Mechanisms": ("Electrical", "Wiring Devices", "Light Switches & Mechanisms"),
    "Comb Switch & Sockets": ("Electrical", "Wiring Devices", "Light Switches & Mechanisms"),
    "Switched Sockets": ("Electrical", "Wiring Devices", "Power Outlets & GPOs"),
    "Domestic Outlets GPOs": ("Electrical", "Wiring Devices", "Power Outlets & GPOs"),
    "Weatherproof Outlets & Switches": ("Electrical", "Wiring Devices", "Weatherproof Outlets & Switches"),
    "Cover Plates": ("Electrical", "Wiring Devices", "Cover Plates & Wall Plates"),
    "Wall Plates": ("Electrical", "Wiring Devices", "Cover Plates & Wall Plates"),
    "Plate & Surround": ("Electrical", "Wiring Devices", "Cover Plates & Wall Plates"),
    "Dimmers & Fan Controllers": ("Electrical", "Wiring Devices", "Dimmers & Fan Controllers"),
    "Dimming & Switch Control": ("Electrical", "Wiring Devices", "Dimmers & Fan Controllers"),
    "Plugs & Sockets": ("Electrical", "Wiring Devices", "Plugs & Sockets"),
    "Plugs": ("Electrical", "Wiring Devices", "Plugs & Sockets"),
    "Plug Tops & Bases": ("Electrical", "Wiring Devices", "Plugs & Sockets"),
    "Plugs and Sockets - Metal": ("Electrical", "Wiring Devices", "Industrial Plugs & Sockets"),
    "Extension Leads": ("Electrical", "Wiring Devices", "Plugs & Sockets"),
    "Power Boards": ("Electrical", "Wiring Devices", "Power Outlets & GPOs"),
    "Mounting Blocks": ("Electrical", "Wiring Devices", "Mounting Blocks & Brackets"),

    # ----- Electrical: Conduit & Fittings -----
    "PVC Rigid Conduit": ("Electrical", "Conduit & Fittings", "Rigid PVC Conduit"),
    "PVC Rigid Conduit Fittings": ("Electrical", "Conduit & Fittings", "Conduit Fittings & Adaptors"),
    "PVC Corrugated Conduit": ("Electrical", "Conduit & Fittings", "Corrugated & Flexible Conduit"),
    "PVC Corrugated Fittings": ("Electrical", "Conduit & Fittings", "Conduit Fittings & Adaptors"),
    "Metal Conduit": ("Electrical", "Conduit & Fittings", "Metal & Industrial Conduit"),
    "Metal Conduit Fittings": ("Electrical", "Conduit & Fittings", "Conduit Fittings & Adaptors"),
    "Industrial Conduit": ("Electrical", "Conduit & Fittings", "Metal & Industrial Conduit"),
    "Industrial Conduit Fittings": ("Electrical", "Conduit & Fittings", "Conduit Fittings & Adaptors"),
    "Communication Conduit": ("Electrical", "Conduit & Fittings", "Metal & Industrial Conduit"),
    "Conduit Fittings": ("Electrical", "Conduit & Fittings", "Conduit Fittings & Adaptors"),
    "Conduit Bend": ("Electrical", "Conduit & Fittings", "Conduit Bends"),
    "Conduit Saddles": ("Electrical", "Conduit & Fittings", "Conduit Saddles & Clips"),
    "Conduit Fixing Accessories": ("Electrical", "Conduit & Fittings", "Conduit Saddles & Clips"),
    "Junction Boxes": ("Electrical", "Conduit & Fittings", "Junction Boxes"),
    "Adaptors & Reducers": ("Electrical", "Conduit & Fittings", "Conduit Fittings & Adaptors"),
    "Bushing": ("Electrical", "Network & Underground", "Bushings & Insulators"),

    # ----- Electrical: Cable & Wire -----
    "Building Wire / Earth Wire <=16Mm": ("Electrical", "Cable & Wire", "Building & Earth Wire"),
    "Flexible Building Wire": ("Electrical", "Cable & Wire", "Building & Earth Wire"),
    "TPS Twin": ("Electrical", "Cable & Wire", "TPS & Flat Cable"),
    "TPS Twin & Earth": ("Electrical", "Cable & Wire", "TPS & Flat Cable"),
    "SDI Single Double Insulated": ("Electrical", "Cable & Wire", "Building & Earth Wire"),
    "XLPE Single Core Cable": ("Electrical", "Cable & Wire", "XLPE & Power Cable"),
    "XLPE Aluminium Cable": ("Electrical", "Cable & Wire", "XLPE & Power Cable"),
    "Flexible Ordinary Duty": ("Electrical", "Cable & Wire", "Flexible & Cord Cable"),
    "Flexible Heavy Duty": ("Electrical", "Cable & Wire", "Flexible & Cord Cable"),
    "Welding Flex": ("Electrical", "Cable & Wire", "Flexible & Cord Cable"),
    "Cable Figure 8": ("Electrical", "Cable & Wire", "Flexible & Cord Cable"),
    "Instrumentation Cable": ("Electrical", "Cable & Wire", "Control & Instrumentation Cable"),
    "Instrumentation Control Cable": ("Electrical", "Cable & Wire", "Control & Instrumentation Cable"),
    "Control Cable SWA": ("Electrical", "Cable & Wire", "Control & Instrumentation Cable"),
    "Cable Fire Rated": ("Electrical", "Cable & Wire", "Fire Rated Cable"),
    "Cable Fire Signal": ("Electrical", "Cable & Wire", "Fire Rated Cable"),
    "Catenary Wire": ("Electrical", "Cable & Wire", "Catenary & Aerial Cable"),
    "Cable Automotive": ("Electrical", "Cable & Wire", "Specialty & Automotive Cable"),
    "Audio Cable": ("Electrical", "Cable & Wire", "Specialty & Automotive Cable"),
    "Solar Cable": ("Electrical", "Solar & EV", "Solar Cable & Connectors"),

    # ----- Electrical: Cable Accessories -----
    "Lugs & Links": ("Electrical", "Cable Accessories", "Cable Lugs & Links"),
    "Terminals": ("Electrical", "Cable Accessories", "Terminals & Terminal Blocks"),
    "Din Rail Mounted Terminals": ("Electrical", "Cable Accessories", "Terminals & Terminal Blocks"),
    "Din Rail Terminals and Accessories": ("Electrical", "Cable Accessories", "Terminals & Terminal Blocks"),
    "Strip Connectors": ("Electrical", "Cable Accessories", "Terminals & Terminal Blocks"),
    "Connectors": ("Electrical", "Cable Accessories", "Cable Jointing & Connectors"),
    "Cable Jointing Kits": ("Electrical", "Cable Accessories", "Cable Jointing & Connectors"),
    "Disconnect Modules": ("Electrical", "Cable Accessories", "Terminals & Terminal Blocks"),
    "Cable Glands": ("Electrical", "Cable Accessories", "Cable Glands"),
    "Cable Glands Metal": ("Electrical", "Cable Accessories", "Cable Glands"),
    "Cable Glands Nylon": ("Electrical", "Cable Accessories", "Cable Glands"),
    "Cable Gland Shrouds": ("Electrical", "Cable Accessories", "Cable Glands"),
    "Heatshrink": ("Electrical", "Cable Accessories", "Heatshrink & Coldshrink"),
    "Coldshrink": ("Electrical", "Cable Accessories", "Heatshrink & Coldshrink"),
    "Spiral Wrap": ("Electrical", "Cable Accessories", "Heatshrink & Coldshrink"),
    "Cable Ties": ("Electrical", "Cable Accessories", "Cable Ties & Clips"),
    "Cable Clips": ("Electrical", "Cable Accessories", "Cable Ties & Clips"),
    "Cable Marking Systems": ("Electrical", "Cable Accessories", "Cable Marking & Identification"),
    "Cable Security": ("Electrical", "Cable Accessories", "Cable Ties & Clips"),
    "Tapes": ("Electrical", "Cable Accessories", "Tapes & Insulating Materials"),
    "Insulating Materials": ("Electrical", "Cable Accessories", "Tapes & Insulating Materials"),

    # ----- Electrical: Cable Management -----
    "Cable Tray": ("Electrical", "Cable Management", "Cable Tray"),
    "Cable Tray Accessories": ("Electrical", "Cable Management", "Cable Tray"),
    "Cable Ladder": ("Electrical", "Cable Management", "Cable Ladder"),
    "Cable Ladder Accessories": ("Electrical", "Cable Management", "Cable Ladder"),
    "Channel / Strut": ("Electrical", "Cable Management", "Channel & Strut"),
    "Channel / Strut Accessories": ("Electrical", "Cable Management", "Channel & Strut"),
    "PVC Duct": ("Electrical", "Cable Management", "Cable Duct & Trunking"),
    "Slotted Duct PVC": ("Electrical", "Cable Management", "Cable Duct & Trunking"),
    "Skirting Duct": ("Electrical", "Cable Management", "Cable Duct & Trunking"),
    "Cable Cover": ("Electrical", "Cable Management", "Cable Cover & Protection"),
    "Cable Management": ("Electrical", "Cable Management", "Cable Cover & Protection"),
    "Threaded Rod": ("Electrical", "Cable Management", "Threaded Rod & Fixings"),

    # ----- Electrical: Enclosures & Boxes -----
    "Enclosures Metal": ("Electrical", "Enclosures & Boxes", "Metal Enclosures"),
    "Enclosures and Mounting - Metal": ("Electrical", "Enclosures & Boxes", "Metal Enclosures"),
    "Polymer Enclosures": ("Electrical", "Enclosures & Boxes", "Polymer & PVC Enclosures"),
    "Enclosures PVC/Polyester": ("Electrical", "Enclosures & Boxes", "Polymer & PVC Enclosures"),
    "Special Enclosures (SS/Custom/Mining)": ("Electrical", "Enclosures & Boxes", "Special & Custom Enclosures"),
    "Wall Boxes": ("Electrical", "Enclosures & Boxes", "Wall & Mounting Boxes"),
    "Terminal Boxes": ("Electrical", "Enclosures & Boxes", "Wall & Mounting Boxes"),
    "Enclosure Accessories": ("Electrical", "Enclosures & Boxes", "Enclosure Accessories"),

    # ----- Electrical: Lighting -----
    "LED D/Light Recessed": ("Electrical", "Lighting", "LED Downlights"),
    "LED Panel": ("Electrical", "Lighting", "LED Panels & Troffers"),
    "Troffer Diffuser & Access": ("Electrical", "Lighting", "LED Panels & Troffers"),
    "LED Batten Diffused": ("Electrical", "Lighting", "LED Battens"),
    "LED Weatherproof Batten": ("Electrical", "Lighting", "LED Battens"),
    "Batten Diffused Fluoro": ("Electrical", "Lighting", "LED Battens"),
    "LED Floodlight Domestic": ("Electrical", "Lighting", "LED Floodlights"),
    "LED Floodlight Commercial": ("Electrical", "Lighting", "LED Floodlights"),
    "LED Floodlight Industrial": ("Electrical", "Lighting", "LED Floodlights"),
    "Portable Floodlights": ("Electrical", "Lighting", "LED Floodlights"),
    "LED High/Low Bay": ("Electrical", "Lighting", "LED Highbay & Lowbay"),
    "LED High/Low Bay IP65": ("Electrical", "Lighting", "LED Highbay & Lowbay"),
    "LED Strip Lighting": ("Electrical", "Lighting", "LED Strip & Extrusion"),
    "LED Strip Extrusion": ("Electrical", "Lighting", "LED Strip & Extrusion"),
    "LED Wall Light Exterior": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "LED Wall Lighting": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "LED Garden Lighting": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "Garden Lighting": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "Street Light LED": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "LED Security / Sensor Lighting": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "Security / Sensor Lighting": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "LED Spotlight & Track": ("Electrical", "Lighting", "Spotlights & Track Lighting"),
    "Spotlight & Track": ("Electrical", "Lighting", "Spotlights & Track Lighting"),
    "LED Shoplight": ("Electrical", "Lighting", "Spotlights & Track Lighting"),
    "LED Oyster": ("Electrical", "Lighting", "Oyster & Bulkhead Lights"),
    "LED Bunker Wall/Ceiling": ("Electrical", "Lighting", "Oyster & Bulkhead Lights"),
    "LED Bulkhead": ("Electrical", "Lighting", "Oyster & Bulkhead Lights"),
    "Pendant Lighting": ("Electrical", "Lighting", "Spotlights & Track Lighting"),
    "Project Luminaires": ("Electrical", "Lighting", "LED Panels & Troffers"),
    "Smart Lighting Luminaires": ("Electrical", "Lighting", "LED Panels & Troffers"),
    "LED Hazardous Area": ("Electrical", "Lighting", "LED Highbay & Lowbay"),
    "Gimbles & Adaptor Plates": ("Electrical", "Lighting", "LED Downlights"),

    # ----- Electrical: Emergency Lighting -----
    "Emergency Exit LED": ("Electrical", "Emergency Lighting", "Emergency Exit Lights"),
    "Exit Diffuser & Spares": ("Electrical", "Emergency Lighting", "Emergency Exit Lights"),
    "Emergency Batten LED": ("Electrical", "Emergency Lighting", "Emergency Battens & Bulkheads"),
    "Emergency Recessed LED": ("Electrical", "Emergency Lighting", "Emergency Battens & Bulkheads"),
    "Emergency Surface Mount LED": ("Electrical", "Emergency Lighting", "Emergency Battens & Bulkheads"),
    "Emergency Bunker/Oyster LED": ("Electrical", "Emergency Lighting", "Emergency Battens & Bulkheads"),
    "Emergency Flood LED": ("Electrical", "Emergency Lighting", "Emergency Battens & Bulkheads"),
    "Emergency Battery Packs": ("Electrical", "Emergency Lighting", "Emergency Battery Packs & Spares"),
    "Emergency Test Devices": ("Electrical", "Emergency Lighting", "Emergency Test Devices"),

    # ----- Electrical: Lamps & Lampholders -----
    "LED Lamps GLS B14 B22 E14 E27": ("Electrical", "Lamps & Lampholders", "LED Lamps"),
    "LED Lamps Linear": ("Electrical", "Lamps & Lampholders", "LED Lamps"),
    "LED Lamps Reflector": ("Electrical", "Lamps & Lampholders", "LED Lamps"),
    "Fluorescent Lamps Linear T5": ("Electrical", "Lamps & Lampholders", "Fluorescent Lamps"),
    "Fluorescent Lamps Linear T8": ("Electrical", "Lamps & Lampholders", "Fluorescent Lamps"),
    "Fluorescent Lamps Circular": ("Electrical", "Lamps & Lampholders", "Fluorescent Lamps"),
    "CFL Non Integrated Ballast": ("Electrical", "Lamps & Lampholders", "Fluorescent Lamps"),
    "Halogen Reflector & Capsule": ("Electrical", "Lamps & Lampholders", "Halogen Lamps"),
    "Halogen Lamps Linear": ("Electrical", "Lamps & Lampholders", "Halogen Lamps"),
    "Halogen GLS B14 B22 E14 E27": ("Electrical", "Lamps & Lampholders", "Halogen Lamps"),
    "Hid Metal Halide": ("Electrical", "Lamps & Lampholders", "HID Lamps"),
    "HID Sodium Vapour": ("Electrical", "Lamps & Lampholders", "HID Lamps"),
    "Incandescent Pilot & Indicator": ("Electrical", "Lamps & Lampholders", "Incandescent & Specialty Lamps"),
    "Automotive Lamps": ("Electrical", "Lamps & Lampholders", "Incandescent & Specialty Lamps"),
    "Heat Lamps": ("Electrical", "Electric Heating", "Heat Lamps"),
    "Lamp Holders": ("Electrical", "Lamps & Lampholders", "Lampholders & Starters"),
    "Batten & Lamp Holder": ("Electrical", "Lamps & Lampholders", "Lampholders & Starters"),
    "Fluorescent Starters": ("Electrical", "Lamps & Lampholders", "Lampholders & Starters"),
    "Driver LED": ("Electrical", "Lamps & Lampholders", "Drivers & Ballasts"),
    "Ballasts & Control Electronics": ("Electrical", "Lamps & Lampholders", "Drivers & Ballasts"),

    # ----- Electrical: Fans & Ventilation -----
    "Fan Exhaust": ("Electrical", "Fans & Ventilation", "Exhaust Fans"),
    "Fan Ceiling Sweep AC": ("Electrical", "Fans & Ventilation", "Ceiling Sweep Fans"),
    "Fan Ceiling Sweep DC": ("Electrical", "Fans & Ventilation", "Ceiling Sweep Fans"),
    "Fan Wall": ("Electrical", "Fans & Ventilation", "Wall & Inline Fans"),
    "Fan Light Heat Units": ("Electrical", "Fans & Ventilation", "Fan Light & Heat Units"),
    "Fan Heating": ("Electrical", "Fans & Ventilation", "Fan Light & Heat Units"),
    "Fan Speed Controllers": ("Electrical", "Fans & Ventilation", "Fan Controllers & Speed Control"),
    "Fan Louvres & Grills": ("Electrical", "Fans & Ventilation", "Louvres, Grills & Mounting"),
    "Fan Mounting Plynths": ("Electrical", "Fans & Ventilation", "Louvres, Grills & Mounting"),
    "Rangehood": ("Electrical", "Fans & Ventilation", "Exhaust Fans"),

    # ----- Electrical: Data & Communications -----
    "Cable Data Copper CAT5 & CAT6": ("Electrical", "Data & Communications", "Copper Data Cable"),
    "Cable Data & Communication": ("Electrical", "Data & Communications", "Copper Data Cable"),
    "Cable Coax": ("Electrical", "Data & Communications", "Copper Data Cable"),
    "Cable Telephone": ("Electrical", "Data & Communications", "Telephone & Intercom"),
    "Cable Fibre Optic": ("Electrical", "Data & Communications", "Fibre Optic Cable & Leads"),
    "Fibre Patch Leads": ("Electrical", "Data & Communications", "Fibre Optic Cable & Leads"),
    "Fibre Plugs": ("Electrical", "Data & Communications", "Fibre Optic Cable & Leads"),
    "Fibre Data Outlets": ("Electrical", "Data & Communications", "Data Outlets & Wall Plates"),
    "Fibre Gland Plates": ("Electrical", "Data & Communications", "Racks, Cabinets & Accessories"),
    "Fibre Patch Panels": ("Electrical", "Data & Communications", "Patch Leads & Patch Panels"),
    "Patch Leads": ("Electrical", "Data & Communications", "Patch Leads & Patch Panels"),
    "Patch Panels": ("Electrical", "Data & Communications", "Patch Leads & Patch Panels"),
    "Modular Plugs": ("Electrical", "Data & Communications", "Patch Leads & Patch Panels"),
    "Data Outlets": ("Electrical", "Data & Communications", "Data Outlets & Wall Plates"),
    "Rack Cabinets": ("Electrical", "Data & Communications", "Racks, Cabinets & Accessories"),
    "Rack Cabinet Accessories": ("Electrical", "Data & Communications", "Racks, Cabinets & Accessories"),
    "Telephone Accessories": ("Electrical", "Data & Communications", "Telephone & Intercom"),
    "Intercoms & Door Bells": ("Electrical", "Data & Communications", "Telephone & Intercom"),
    "Audio Visual Leads": ("Electrical", "Data & Communications", "Audio Visual & Antennas"),
    "Amplifiers": ("Electrical", "Data & Communications", "Audio Visual & Antennas"),
    "Splitters": ("Electrical", "Data & Communications", "Audio Visual & Antennas"),
    "Splitters & Amplifiers": ("Electrical", "Data & Communications", "Audio Visual & Antennas"),
    "Antennas": ("Electrical", "Data & Communications", "Audio Visual & Antennas"),
    "Satellite Dish": ("Electrical", "Data & Communications", "Audio Visual & Antennas"),
    "Set Top Box": ("Electrical", "Data & Communications", "Audio Visual & Antennas"),
    "Hubs & Switches": ("Electrical", "Data & Communications", "Networking Hubs & Switches"),
    "Computer Accessories": ("Electrical", "Data & Communications", "Networking Hubs & Switches"),

    # ----- Electrical: Control & Automation -----
    "Contactors & Overloads": ("Electrical", "Control & Automation", "Contactors & Overloads"),
    "Contactors": ("Electrical", "Control & Automation", "Contactors & Overloads"),
    "Relays & Accessories": ("Electrical", "Control & Automation", "Relays & Timers"),
    "Relays Timing": ("Electrical", "Control & Automation", "Relays & Timers"),
    "Timers": ("Electrical", "Control & Automation", "Relays & Timers"),
    "Plug In Timers": ("Electrical", "Control & Automation", "Relays & Timers"),
    "Push Buttons and Indicators": ("Electrical", "Control & Automation", "Push Buttons & Indicators"),
    "Motor Control & Accessories": ("Electrical", "Control & Automation", "Motor Control & Soft Starters"),
    "Soft Starter Star Delta & DOL": ("Electrical", "Control & Automation", "Motor Control & Soft Starters"),
    "Variable Speed Drives": ("Electrical", "Control & Automation", "Variable Speed Drives"),
    "Signalling Devices": ("Electrical", "Control & Automation", "Signalling Devices"),
    "C-Bus": ("Electrical", "Control & Automation", "Smart & Home Automation"),
    "Home Automation": ("Electrical", "Control & Automation", "Smart & Home Automation"),

    # ----- Electrical: Sensors & Detection -----
    "PIR & Motion Sensors": ("Electrical", "Sensors & Detection", "PIR & Motion Sensors"),
    "Motion Detectors": ("Electrical", "Sensors & Detection", "PIR & Motion Sensors"),
    "Smoke Detectors": ("Electrical", "Sensors & Detection", "Smoke & Fire Detectors"),
    "Proximity Sensors": ("Electrical", "Sensors & Detection", "Proximity & Special Sensors"),
    "Special Sensors & Accessories": ("Electrical", "Sensors & Detection", "Proximity & Special Sensors"),
    "Pressure Level & Limit": ("Electrical", "Sensors & Detection", "Proximity & Special Sensors"),
    "Sunset Switches": ("Electrical", "Sensors & Detection", "Sunset & Daylight Switches"),
    "Camera Kits": ("Electrical", "Sensors & Detection", "Cameras & Security Kits"),

    # ----- Electrical: Metering & Monitoring -----
    "Power Measuring & Monitoring": ("Electrical", "Metering & Monitoring", "Power Meters & Monitoring"),
    "Metering": ("Electrical", "Metering & Monitoring", "Metering Equipment"),
    "Distribution Frames & Boxes": ("Electrical", "Metering & Monitoring", "Distribution Frames & Boxes"),

    # ----- Electrical: Earthing & Grounding -----
    "Earth Grounding": ("Electrical", "Earthing & Grounding", "Earth Grounding Equipment"),

    # ----- Electrical: Solar & EV -----
    "Photovoltaic Panels": ("Electrical", "Solar & EV", "Photovoltaic Panels"),
    "Inverters": ("Electrical", "Solar & EV", "Inverters"),
    "MC4 Connectors": ("Electrical", "Solar & EV", "Solar Cable & Connectors"),
    "DC Isolators": ("Electrical", "Solar & EV", "DC Isolators"),
    "Solar Battery": ("Electrical", "Solar & EV", "Solar Batteries & Storage"),
    "EV Charging Station": ("Electrical", "Solar & EV", "EV Charging Stations"),

    # ----- Electrical: Power Supplies & Transformers -----
    "Power Suppliers & Transformers": ("Electrical", "Power Supplies & Transformers", "Transformers & Power Supplies"),
    "Selv Transformers": ("Electrical", "Power Supplies & Transformers", "Transformers & Power Supplies"),
    "Batteries & Chargers": ("Electrical", "Power Supplies & Transformers", "Batteries & Chargers"),
    "Capacitors & Chokes": ("Electrical", "Power Supplies & Transformers", "Capacitors & Chokes"),

    # ----- Electrical: Electric Heating -----
    "Floor Heating": ("Electrical", "Electric Heating", "Floor & Panel Heating"),
    "Panel Heating": ("Electrical", "Electric Heating", "Floor & Panel Heating"),
    "Radiant Heating": ("Electrical", "Electric Heating", "Radiant & Space Heating"),

    # ----- Electrical: Network & Underground -----
    "Underground Distribution Equipment": ("Electrical", "Network & Underground", "Underground Distribution Equipment"),

    # ----- Mechanical: Air Conditioning -----
    "Airconditioning Split System": ("Mechanical", "Air Conditioning", "Split System Air Conditioning"),
    "Airconditioning Accessories": ("Mechanical", "Air Conditioning", "Air Conditioning Accessories"),

    # ----- Mechanical: Ducting & Ventilation -----
    "Metal Duct": ("Mechanical", "Ducting & Ventilation", "Metal Duct"),
    "PVC Flexible Hose": ("Mechanical", "Ducting & Ventilation", "Flexible Duct & Hose"),
    "PVC Flexible Hose Fittings": ("Mechanical", "Ducting & Ventilation", "Flexible Duct & Hose"),

    # ----- Mechanical: Appliances -----
    "Freestanding Cookers": ("Mechanical", "Appliances", "Cooking Appliances & Ranges"),
    "Electric Range Spares": ("Mechanical", "Appliances", "Cooking Appliances & Ranges"),
    "Hand Dryers": ("Mechanical", "Appliances", "Hand Dryers"),
    "Catering Equipment": ("Mechanical", "Appliances", "Catering Equipment"),

    # ----- Plumbing: Hot Water -----
    "Elements Hot Water": ("Plumbing", "Hot Water", "Hot Water Elements"),
    "Water Heating & Boiling": ("Plumbing", "Hot Water", "Water Heating & Boiling Units"),
    "Water Coolers": ("Plumbing", "Hot Water", "Water Coolers"),
    "Thermostats": ("Plumbing", "Hot Water", "Water Heating & Boiling Units"),

    # ----- Plumbing: Pipe & Hose -----
    "Hose Clamps": ("Plumbing", "Pipe & Hose", "Hose Clamps & Straps"),

    # ----- Tools & Site: Drilling & Cutting -----
    "Drilling": ("Tools & Site", "Drilling & Cutting", "Drill Bits & Drilling"),
    "Holesaws": ("Tools & Site", "Drilling & Cutting", "Holesaws"),
    "Saws & Blades": ("Tools & Site", "Drilling & Cutting", "Saws & Blades"),
    "Cutting & Grinding Wheels": ("Tools & Site", "Drilling & Cutting", "Cutting & Grinding Wheels"),
    "Bending Springs": ("Tools & Site", "Drilling & Cutting", "Bending Springs"),

    # ----- Tools & Site: Hand Tools -----
    "Cutters & Pliers": ("Tools & Site", "Hand Tools", "Cutters & Pliers"),
    "Screwdrivers": ("Tools & Site", "Hand Tools", "Screwdrivers"),
    "Crimp Tools": ("Tools & Site", "Hand Tools", "Crimp & Cable Tie Tools"),
    "Cable Tie Tools": ("Tools & Site", "Hand Tools", "Crimp & Cable Tie Tools"),
    "Stripping Tools": ("Tools & Site", "Hand Tools", "Stripping Tools"),
    "Adjustable Wrenches": ("Tools & Site", "Hand Tools", "Spanners & Wrenches"),
    "Spanners": ("Tools & Site", "Hand Tools", "Spanners & Wrenches"),
    "Hex Bits & Keys": ("Tools & Site", "Hand Tools", "Hex Bits & Keys"),
    "Knives": ("Tools & Site", "Hand Tools", "Knives & Files"),
    "Files": ("Tools & Site", "Hand Tools", "Knives & Files"),
    "Socket Sets": ("Tools & Site", "Hand Tools", "Socket Sets"),
    "Micro Tools": ("Tools & Site", "Hand Tools", "Cutters & Pliers"),
    "Communication Tools": ("Tools & Site", "Hand Tools", "Crimp & Cable Tie Tools"),

    # ----- Tools & Site: Power Tools -----
    "Power Tools": ("Tools & Site", "Power Tools", "Power Tools"),
    "Power Tool Batteries & Chargers": ("Tools & Site", "Power Tools", "Power Tool Batteries & Chargers"),
    "Soldering Equipment": ("Tools & Site", "Power Tools", "Soldering & Heat Tools"),
    "Solder": ("Tools & Site", "Power Tools", "Soldering & Heat Tools"),
    "Heat Guns": ("Tools & Site", "Power Tools", "Soldering & Heat Tools"),
    "Caulking Guns": ("Tools & Site", "Power Tools", "Caulking & Glue Guns"),

    # ----- Tools & Site: Test & Measurement -----
    "Multimeters": ("Tools & Site", "Test & Measurement", "Multimeters & Clamp Meters"),
    "Clamp Meters": ("Tools & Site", "Test & Measurement", "Multimeters & Clamp Meters"),
    "Insulation Testers": ("Tools & Site", "Test & Measurement", "Insulation & RCD Testers"),
    "RCD & ELCB Testers": ("Tools & Site", "Test & Measurement", "Insulation & RCD Testers"),
    "Continuity & Voltage Testers": ("Tools & Site", "Test & Measurement", "Voltage Detectors & Indicators"),
    "Voltage Indicators & Detectors": ("Tools & Site", "Test & Measurement", "Voltage Detectors & Indicators"),
    "Measuring Tools": ("Tools & Site", "Test & Measurement", "Measuring & Thermometers"),
    "Thermometers": ("Tools & Site", "Test & Measurement", "Measuring & Thermometers"),
    "Fibre Testers": ("Tools & Site", "Test & Measurement", "Fibre & Communication Testers"),
    "Test Leads": ("Tools & Site", "Test & Measurement", "Test Leads"),

    # ----- Tools & Site: Safety & PPE -----
    "Gloves": ("Tools & Site", "Safety & PPE", "Gloves & Hand Protection"),
    "Protective Eyeware": ("Tools & Site", "Safety & PPE", "Eye & Face Protection"),
    "Hearing Protection": ("Tools & Site", "Safety & PPE", "Hearing Protection"),
    "Respirators & Face Masks": ("Tools & Site", "Safety & PPE", "Respirators & Masks"),
    "Lockout Equipment": ("Tools & Site", "Safety & PPE", "Lockout Equipment"),
    "Worksite Safety": ("Tools & Site", "Safety & PPE", "Worksite Safety"),
    "Fire Blankets": ("Tools & Site", "Safety & PPE", "Fire Blankets & Barriers"),

    # ----- Tools & Site: Marking & Labelling -----
    "Labels & Signs": ("Tools & Site", "Marking & Labelling", "Labels & Signs"),
    "Marking & Paint": ("Tools & Site", "Marking & Labelling", "Marking & Paint"),
    "Portable Labelling Systems": ("Tools & Site", "Marking & Labelling", "Labelling Systems"),
    "Warning Tape": ("Tools & Site", "Marking & Labelling", "Warning Tape"),
    "Appliance Test Tags": ("Tools & Site", "Marking & Labelling", "Appliance Test Tags"),

    # ----- Tools & Site: Adhesives & Chemicals -----
    "Silicone & Sealants": ("Tools & Site", "Adhesives & Chemicals", "Silicone & Sealants"),
    "Adhesives": ("Tools & Site", "Adhesives & Chemicals", "Adhesives"),
    "Chemical Sprays": ("Tools & Site", "Adhesives & Chemicals", "Chemical Sprays & Aerosols"),
    "Aerosols": ("Tools & Site", "Adhesives & Chemicals", "Chemical Sprays & Aerosols"),
    "Lubricants": ("Tools & Site", "Adhesives & Chemicals", "Lubricants"),
    "Fire Sealants & Barriers": ("Tools & Site", "Adhesives & Chemicals", "Fire Sealants & Barriers"),

    # ----- Tools & Site: Fasteners & Fixings -----
    "Fasteners & Fixings": ("Tools & Site", "Fasteners & Fixings", "Fasteners & Fixings"),
    "Washers": ("Tools & Site", "Fasteners & Fixings", "Washers & Lock Nuts"),
    "Lock Nuts": ("Tools & Site", "Fasteners & Fixings", "Washers & Lock Nuts"),
    "Locknuts": ("Tools & Site", "Fasteners & Fixings", "Washers & Lock Nuts"),
    "Straps": ("Tools & Site", "Fasteners & Fixings", "Straps & Clips"),
    "Jack Chain": ("Tools & Site", "Fasteners & Fixings", "Straps & Clips"),
    "Girder Clips": ("Tools & Site", "Fasteners & Fixings", "Straps & Clips"),
    "Flashings": ("Tools & Site", "Fasteners & Fixings", "Straps & Clips"),
    "Mounting Systems": ("Tools & Site", "Fasteners & Fixings", "Mounting Brackets & Systems"),
    "Mounting Brackets": ("Tools & Site", "Fasteners & Fixings", "Mounting Brackets & Systems"),
    "Mounting Brackets & Accessories": ("Tools & Site", "Fasteners & Fixings", "Mounting Brackets & Systems"),
    "Wall Brackets": ("Tools & Site", "Fasteners & Fixings", "Mounting Brackets & Systems"),
    "Roof Mounting Products": ("Tools & Site", "Fasteners & Fixings", "Mounting Brackets & Systems"),

    # ----- Tools & Site: Access & Storage -----
    "Ladders": ("Tools & Site", "Access & Storage", "Ladders"),
    "Storage Solutions": ("Tools & Site", "Access & Storage", "Storage Solutions & Tool Boxes"),
    "Tool Boxes": ("Tools & Site", "Access & Storage", "Storage Solutions & Tool Boxes"),
    "Cases": ("Tools & Site", "Access & Storage", "Cases"),
    "Cable Pulling Equipment": ("Tools & Site", "Access & Storage", "Cable Pulling Equipment"),

    # ----- Additional / punctuation & ERP-variant categories -----
    "Soft Starter, Star Delta & DOL": ("Electrical", "Control & Automation", "Motor Control & Soft Starters"),
    "Pressure, Level & Limit": ("Electrical", "Sensors & Detection", "Proximity & Special Sensors"),
    "Surge &amp; Lightning Protection": ("Electrical", "Circuit Protection & Distribution", "Surge Protection & UPS"),
    "Surge & Lightning Protection": ("Electrical", "Circuit Protection & Distribution", "Surge Protection & UPS"),
    "Comb Switch & Sockets - Metal": ("Electrical", "Wiring Devices", "Light Switches & Mechanisms"),
    "Plugs and Sockets - Mining": ("Electrical", "Wiring Devices", "Industrial Plugs & Sockets"),
    "Plugs and Connectors": ("Electrical", "Wiring Devices", "Industrial Plugs & Sockets"),
    "Industrial Connectors": ("Electrical", "Wiring Devices", "Industrial Plugs & Sockets"),
    "MC4 Pre Assembled Leads": ("Electrical", "Solar & EV", "Solar Cable & Connectors"),
    "Rubber Multicore Cables": ("Electrical", "Cable & Wire", "Flexible & Cord Cable"),
    "Rubber 3.3Kv Cables": ("Electrical", "Cable & Wire", "XLPE & Power Cable"),
    "High Temperature Cables": ("Electrical", "Cable & Wire", "Specialty & Automotive Cable"),
    "Cable Low Voltage": ("Electrical", "Cable & Wire", "Specialty & Automotive Cable"),
    "Braded Cables & Leads": ("Electrical", "Cable & Wire", "Flexible & Cord Cable"),
    "Other Security Systems": ("Electrical", "Sensors & Detection", "Cameras & Security Kits"),
    "Other Datacommunications": ("Electrical", "Data & Communications", "Copper Data Cable"),
    "Other Power Distribution": ("Electrical", "Circuit Protection & Distribution", "Switchboards & Distribution Boards"),
    "Other Luminaires": ("Electrical", "Lighting", "LED Panels & Troffers"),
    "Other Ventilation": ("Electrical", "Fans & Ventilation", "Exhaust Fans"),
    "Other Conduit": ("Electrical", "Conduit & Fittings", "Conduit Fittings & Adaptors"),
    "Torches": ("Electrical", "Lighting", "Exterior & Wall Lighting"),
    "Power Analysers": ("Tools & Site", "Test & Measurement", "Multimeters & Clamp Meters"),
    "Knee Protectors": ("Tools & Site", "Safety & PPE", "Worksite Safety"),
    "Head & Face Protection": ("Tools & Site", "Safety & PPE", "Eye & Face Protection"),
    "Hand Tools": ("Tools & Site", "Hand Tools", "Cutters & Pliers"),
    "Hand Tools Not Data": ("Tools & Site", "Hand Tools", "Cutters & Pliers"),
}


# ---------------------------------------------------------------------------
# Punctuation/escape-tolerant lookup. ERP category strings vary in trivial
# ways (commas, HTML-escaped "&amp;", extra spaces). Normalizing the lookup
# means those variants resolve to the same node instead of silently falling
# through to description classification — and future variants self-heal.
# ---------------------------------------------------------------------------
import re as _re


def _norm_cat(s: str) -> str:
    s = (s or "").replace("&amp;", "&")
    s = s.lower().replace(",", " ").replace("/", " ")
    return " ".join(s.split())


_NORM_CATEGORY_MAP: dict = {}
for _k, _v in CATEGORY_MAP.items():
    _NORM_CATEGORY_MAP.setdefault(_norm_cat(_k), _v)


def lookup_category(erp_category):
    """Return (domain, category, subcategory) for an ERP category string.

    Tries an exact match first, then a punctuation/whitespace-normalized match.
    Returns (None, None, None) for unknown or ambiguous-bucket categories, which
    the caller treats as "use description-based classification".
    """
    if not erp_category:
        return (None, None, None)
    erp_category = str(erp_category).strip()
    if erp_category in CATEGORY_MAP:
        return CATEGORY_MAP[erp_category]
    return _NORM_CATEGORY_MAP.get(_norm_cat(erp_category), (None, None, None))
