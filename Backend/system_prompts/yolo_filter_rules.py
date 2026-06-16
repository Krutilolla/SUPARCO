import numpy as np

DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "expressway-service-area", "expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium",
    "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"
]

CLASS_SYNONYMS = {
    # Airplane variations
    "airplane": ["airplane", "plane", "planes", "aircraft", "aircrafts", "jet", "jets", 
                 "aeroplane", "aeroplanes", "airliner", "airliners", "flight", "flights",
                 "fighter", "fighters", "bomber", "bombers", "aviation"],
    
    # Airport variations
    "airport": ["airport", "airports", "airfield", "airfields", "aerodrome", "aerodromes",
                "runway", "runways", "terminal", "terminals", "tarmac", "helipad", "heliport"],
    
    # Baseball field variations
    "baseballfield": ["baseballfield", "baseball field", "baseball", "baseball diamond",
                      "diamond", "ballpark", "ball park", "softball field", "softball"],
    
    # Basketball court variations
    "basketballcourt": ["basketballcourt", "basketball court", "basketball", "court",
                        "hoops", "b-ball court"],
    
    # Bridge variations
    "bridge": ["bridge", "bridges", "overpass", "flyover", "viaduct", "crossing",
               "footbridge", "pedestrian bridge", "road bridge"],
    
    # Chimney variations
    "chimney": ["chimney", "chimneys", "smokestack", "smokestacks", "stack", "stacks",
                "flue", "industrial chimney", "exhaust", "cooling tower"],
    
    # Dam variations
    "dam": ["dam", "dams", "reservoir", "reservoirs", "barrage", "weir", "levee",
            "hydroelectric", "hydro dam", "water dam"],
    
    # Expressway service area variations
    "expressway-service-area": ["expressway-service-area", "service area", "rest area",
                                 "rest stop", "truck stop", "highway rest", "motorway services",
                                 "service station", "travel plaza", "rest station"],
    
    # Expressway toll station variations
    "expressway-toll-station": ["expressway-toll-station", "toll station", "toll booth",
                                 "toll gate", "toll plaza", "tollway", "toll road",
                                 "toll barrier", "payment booth", "toll collection"],
    
    # Golf field variations
    "golffield": ["golffield", "golf field", "golf course", "golf", "fairway", "fairways",
                  "green", "greens", "golf club", "driving range", "putting green", "links"],
    
    # Ground track field variations
    "groundtrackfield": ["groundtrackfield", "ground track field", "track field", "track",
                          "running track", "athletic track", "athletics field", "oval",
                          "race track", "sports track", "stadium track", "athletic field"],
    
    # Harbor variations
    "harbor": ["harbor", "harbors", "harbour", "harbours", "port", "ports", "dock", "docks",
               "marina", "marinas", "pier", "piers", "wharf", "wharves", "quay", "jetty",
               "seaport", "waterfront", "anchorage", "mooring", "berth"],
    
    # Overpass variations
    "overpass": ["overpass", "overpasses", "flyover", "flyovers", "elevated road",
                 "elevated highway", "interchange", "junction", "cloverleaf",
                 "highway bridge", "road bridge", "crossing"],
    
    # Ship variations
    "ship": ["ship", "ships", "boat", "boats", "vessel", "vessels", "cargo ship",
             "container ship", "tanker", "tankers", "freighter", "freighters",
             "barge", "barges", "ferry", "ferries", "yacht", "yachts", "cruiser",
             "warship", "naval vessel", "fishing boat", "trawler", "sailboat"],
    
    # Stadium variations
    "stadium": ["stadium", "stadiums", "arena", "arenas", "sports complex",
                "football stadium", "soccer stadium", "athletic stadium",
                "amphitheater", "coliseum", "field", "sports field", "pitch"],
    
    # Storage tank variations
    "storagetank": ["storagetank", "storage tank", "storage tanks", "tank", "tanks",
                    "oil tank", "fuel tank", "water tank", "silo", "silos",
                    "reservoir tank", "container", "cistern", "holding tank",
                    "petroleum tank", "industrial tank", "cylindrical tank"],
    
    # Tennis court variations
    "tenniscourt": ["tenniscourt", "tennis court", "tennis courts", "tennis",
                    "tennis field", "racquet court", "badminton court"],
    
    # Train station variations
    "trainstation": ["trainstation", "train station", "railway station", "rail station",
                     "railroad station", "depot", "terminal", "train terminal",
                     "metro station", "subway station", "station", "platform",
                     "railway", "railroad", "train yard", "rail yard"],
    
    # Vehicle variations
    "vehicle": ["vehicle", "vehicles", "car", "cars", "automobile", "automobiles",
                "truck", "trucks", "van", "vans", "bus", "buses", "suv", "suvs",
                "pickup", "pickups", "lorry", "lorries", "motor vehicle",
                "transportation", "auto", "autos", "sedan", "hatchback", "jeep"],
    
    # Windmill variations
    "windmill": ["windmill", "windmills", "wind turbine", "wind turbines", "turbine",
                 "turbines", "wind farm", "wind generator", "wind power",
                 "wind energy", "aerogenerator", "pinwheel"],
}

# Reverse mapping: synonym → class
SYNONYM_TO_CLASS = {}
for cls, synonyms in CLASS_SYNONYMS.items():
    for syn in synonyms:
        SYNONYM_TO_CLASS[syn.lower()] = cls

# ============================================================
# SPATIAL REGION DEFINITIONS
# ============================================================
SPATIAL_REGIONS = {
    # 9-grid regions
    "top left": {"x": (0, 0.33), "y": (0, 0.33)},
    "top center": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "top middle": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "top right": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "upper left": {"x": (0, 0.33), "y": (0, 0.33)},
    "upper center": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "upper middle": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "upper right": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "center left": {"x": (0, 0.33), "y": (0.33, 0.66)},
    "center": {"x": (0.33, 0.66), "y": (0.33, 0.66)},
    "middle": {"x": (0.33, 0.66), "y": (0.33, 0.66)},
    "center right": {"x": (0.66, 1.0), "y": (0.33, 0.66)},
    "bottom left": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "bottom center": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "bottom middle": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "bottom right": {"x": (0.66, 1.0), "y": (0.66, 1.0)},
    "lower left": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "lower center": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "lower middle": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "lower right": {"x": (0.66, 1.0), "y": (0.66, 1.0)},


    "top-left": {"x": (0, 0.33), "y": (0, 0.33)},
    "top-center": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "top-middle": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "top-right": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "upper-left": {"x": (0, 0.33), "y": (0, 0.33)},
    "upper-center": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "upper-middle": {"x": (0.33, 0.66), "y": (0, 0.33)},
    "upper-right": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "center-left": {"x": (0, 0.33), "y": (0.33, 0.66)},
    "central": {"x": (0.33, 0.66), "y": (0.33, 0.66)},
    "center-right": {"x": (0.66, 1.0), "y": (0.33, 0.66)},
    "bottom-left": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "bottom-center": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "bottom-middle": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "bottom-right": {"x": (0.66, 1.0), "y": (0.66, 1.0)},
    "lower-left": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "lower-center": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "lower-middle": {"x": (0.33, 0.66), "y": (0.66, 1.0)},
    "lower-right": {"x": (0.66, 1.0), "y": (0.66, 1.0)},
    
    # Half regions
    "top": {"x": (0, 1.0), "y": (0, 0.5)},
    "bottom": {"x": (0, 1.0), "y": (0.5, 1.0)},

    "upper": {"x": (0, 1.0), "y": (0, 0.5)},
    "lower": {"x": (0, 1.0), "y": (0.5, 1.0)},
    
    "left": {"x": (0, 0.5), "y": (0, 1.0)},
    "right": {"x": (0.5, 1.0), "y": (0, 1.0)},
    
    # Alternative phrasings
    "upper left": {"x": (0, 0.33), "y": (0, 0.33)},
    "upper right": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "lower left": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "lower right": {"x": (0.66, 1.0), "y": (0.66, 1.0)},
    "middle": {"x": (0.33, 0.66), "y": (0.33, 0.66)},

    "upper-left": {"x": (0, 0.33), "y": (0, 0.33)},
    "upper-right": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "lower-left": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "lower-right": {"x": (0.66, 1.0), "y": (0.66, 1.0)},
    
    # Corner phrases
    "top left corner": {"x": (0, 0.33), "y": (0, 0.33)},
    "top right corner": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "bottom left corner": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "bottom right corner": {"x": (0.66, 1.0), "y": (0.66, 1.0)},

    "top-left corner": {"x": (0, 0.33), "y": (0, 0.33)},
    "top-right corner": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "bottom-left corner": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "bottom-right corner": {"x": (0.66, 1.0), "y": (0.66, 1.0)},
    
    "upper-left corner": {"x": (0, 0.33), "y": (0, 0.33)},
    "upper-right corner": {"x": (0.66, 1.0), "y": (0, 0.33)},
    "lower-left corner": {"x": (0, 0.33), "y": (0.66, 1.0)},
    "lower-right corner": {"x": (0.66, 1.0), "y": (0.66, 1.0)},
}

# Spatial synonyms for parsing
SPATIAL_SYNONYMS = {
    "upper": "top",
    "lower": "bottom",
    "middle": "center",
    "central": "center",
    "centrally": "center",
    "northwest": "top left",
    "northeast": "top right",
    "southwest": "bottom left",
    "southeast": "bottom right",
}

# Size-based filters with synonyms
SIZE_FILTERS = {
    "largest": ["largest", "biggest", "huge", "massive", "enormous", "giant", "big", "large"],
    "smallest": ["smallest", "tiniest", "tiny", "little", "mini", "miniature", "small"],
    "longest": ["longest", "lengthiest"],
    "shortest": ["shortest"],
    "widest": ["widest", "broadest"],
    "narrowest": ["narrowest", "thinnest"],
}

# Flatten size filters for lookup
SIZE_KEYWORD_TO_FILTER = {}
for filter_name, keywords in SIZE_FILTERS.items():
    for kw in keywords:
        SIZE_KEYWORD_TO_FILTER[kw] = filter_name

# Ordinal filters
ORDINAL_FILTERS = ["first", "second", "third", "fourth", "fifth",
                   "1st", "2nd", "3rd", "4th", "5th",
                   "leftmost", "left-most", "left edge", "left-edge", "far left", "far-left", "left side", "left-side",
                   "rightmost", "right-most", "right edge", "right-edge", "far right", "far-right", "right side", "right-side",
                   "topmost", "top most", "top-most", "uppermost", "upper most", "top edge", "top-edge", "upper edge", "upper-edge", "upper-most",
                   "bottommost", "bottom most", "bottom-most", "lowermost", "lower most", "bottom edge", "bottom-edge", "lower edge", "lower-edge", "lower-most",
                   "nearest", "farthest", "closest", "furthest"]

# ============================================================
# COLOR DETECTION SYSTEM (24 colors)
# ============================================================
COLOR_HSV_RANGES = {
    # Reds (wrap around 0 and 180) - RELAXED
    "red": [
        {"lower": np.array([0, 40, 30]), "upper": np.array([12, 255, 255])},
        {"lower": np.array([165, 40, 30]), "upper": np.array([180, 255, 255])}
    ],
    "crimson": [
        {"lower": np.array([0, 60, 60]), "upper": np.array([12, 255, 220])},
        {"lower": np.array([165, 60, 60]), "upper": np.array([180, 255, 220])}
    ],
    "maroon": [
        # Dark reds - very relaxed: low saturation OK, very low value OK
        {"lower": np.array([0, 30, 15]), "upper": np.array([15, 255, 120])},
        {"lower": np.array([160, 30, 15]), "upper": np.array([180, 255, 120])}
    ],
    
    # Oranges - RELAXED
    "orange": [{"lower": np.array([8, 60, 60]), "upper": np.array([28, 255, 255])}],
    
    # Yellows - RELAXED
    "yellow": [{"lower": np.array([18, 60, 60]), "upper": np.array([38, 255, 255])}],
    "gold": [{"lower": np.array([12, 60, 60]), "upper": np.array([32, 255, 230])}],
    "cream": [{"lower": np.array([12, 15, 180]), "upper": np.array([38, 100, 255])}],
    "ivory": [{"lower": np.array([12, 5, 200]), "upper": np.array([38, 60, 255])}],
    
    # Greens - RELAXED
    "green": [{"lower": np.array([32, 30, 30]), "upper": np.array([88, 255, 255])}],
    "teal": [{"lower": np.array([75, 30, 30]), "upper": np.array([100, 255, 255])}],
    
    # Cyans - RELAXED
    "cyan": [{"lower": np.array([80, 50, 50]), "upper": np.array([105, 255, 255])}],
    
    # Blues - RELAXED
    "blue": [{"lower": np.array([95, 30, 30]), "upper": np.array([135, 255, 255])}],
    "indigo": [{"lower": np.array([95, 50, 30]), "upper": np.array([125, 255, 180])}],
    "navy": [{"lower": np.array([95, 50, 10]), "upper": np.array([135, 255, 120])}],
    
    # Purples/Violets - RELAXED
    "purple": [{"lower": np.array([120, 30, 30]), "upper": np.array([160, 255, 255])}],
    "violet": [{"lower": np.array([125, 30, 60]), "upper": np.array([155, 255, 255])}],
    "lavender": [{"lower": np.array([125, 10, 160]), "upper": np.array([155, 100, 255])}],
    "magenta": [{"lower": np.array([135, 50, 50]), "upper": np.array([175, 255, 255])}],
    
    # Pinks - RELAXED
    "pink": [
        {"lower": np.array([145, 20, 120]), "upper": np.array([180, 180, 255])},
        {"lower": np.array([0, 20, 120]), "upper": np.array([12, 180, 255])}
    ],
    
    # Browns - RELAXED (important for satellite imagery)
    "brown": [{"lower": np.array([3, 30, 20]), "upper": np.array([25, 220, 180])}],
    "tan": [{"lower": np.array([8, 20, 120]), "upper": np.array([28, 120, 240])}],
    "beige": [{"lower": np.array([12, 10, 160]), "upper": np.array([35, 80, 250])}],
    
    # Grays (low saturation) - RELAXED
    "gray": [{"lower": np.array([0, 0, 40]), "upper": np.array([180, 40, 210])}],
    "grey": [{"lower": np.array([0, 0, 40]), "upper": np.array([180, 40, 210])}],
    "silver": [{"lower": np.array([0, 0, 130]), "upper": np.array([180, 30, 230])}],
    "charcoal": [{"lower": np.array([0, 0, 20]), "upper": np.array([180, 40, 100])}],
    
    # Black (very low value) - RELAXED
    "black": [{"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 60])}],
    "dark": [{"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 85])}],
    
    # White (high value, low saturation) - RELAXED
    "white": [{"lower": np.array([0, 0, 180]), "upper": np.array([180, 40, 255])}],
    "light": [{"lower": np.array([0, 0, 160]), "upper": np.array([180, 50, 255])}],
}

BASE_COLORS = set(COLOR_HSV_RANGES.keys())

COLORS = list(COLOR_HSV_RANGES.keys())
BASE_SIZES=list(SIZE_FILTERS.keys())
BASE_SPATIAL_REGIONS=list(SPATIAL_REGIONS.keys())