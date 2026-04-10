from flask import Flask, request, jsonify
import pandas as pd
import os
import re

app = Flask(__name__)

# ===== Load dataset =====
df = pd.read_csv("adidas_usa.csv")
df.columns = df.columns.str.strip().str.lower()

# Make sure these columns are usable
for col in ["name", "brand", "color", "category", "description", "breadcrumbs", "availability", "images"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

for col in ["selling_price", "original_price", "average_rating", "reviews_count"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ===== Intent names from Dialogflow =====
INTENT_PRODUCT_SEARCH = "Product Search"
INTENT_LIST_CATEGORIES = "List Categories"
INTENT_SHOW_MORE = "Show More"
INTENT_HELP = "help"
INTENT_GOODBYE = "goodbye"
INTENT_NEGATIVE = "Negative Intent"
INTENT_WELCOME = "Default Welcome Intent"
INTENT_COMPARE = "Compare Products"
INTENT_PRODUCT_DETAIL = "Product Detail"
INTENT_AVAILABILITY = "Check Availability"
INTENT_SELECT_PRODUCT = "Select Product"   # user types "1", "2", "3" to pick from list
INTENT_GENDER_FILTER = "Gender Filter"     # user says "men" / "women" / "kids"

PAGE_SIZE = 3
SESSION_CACHE = {}

# ===== Wishlist stored per session (in-memory) =====
# Structure: { session_id: [ {product row dict}, ... ] }
WISHLIST_CACHE = {}

# ── Patterns for "back to results" ──
BACK_TO_RESULTS_PATTERN = re.compile(
    r"(?:back\s+to\s+results?|go\s+back|previous\s+results?|back)",
    re.IGNORECASE,
)

# ── Patterns for wishlist actions ──
WISHLIST_ADD_PATTERN = re.compile(
    r"\b(?:save\s+(?:this|it)|add\s+to\s+(?:wishlist|favorites?|saved)|wishlist|favourite|bookmark\s+this)\b",
    re.IGNORECASE,
)
WISHLIST_VIEW_PATTERN = re.compile(
    r"\b(?:show\s+(?:my\s+)?(?:wishlist|saved|favorites?)|my\s+wishlist|saved\s+items?|view\s+wishlist)\b",
    re.IGNORECASE,
)
WISHLIST_CLEAR_PATTERN = re.compile(
    r"\b(?:clear\s+wishlist|remove\s+all|empty\s+wishlist)\b",
    re.IGNORECASE,
)

# ── Pattern for "surprise me" ──
SURPRISE_PATTERN = re.compile(
    r"\b(?:surprise\s+me|random\s+pick|pick\s+(?:something|one)\s+for\s+me|recommend\s+(?:something|one)|just\s+pick\s+one)\b",
    re.IGNORECASE,
)

# ===== Stop words: never count these as meaningful product-name tokens =====
GENERIC_WORDS = {
    # footwear
    "shoes", "shoe", "sneakers", "sneaker", "boots", "boot", "sandals", "sandal",
    "slides", "slide", "footwear", "trainers", "trainer", "kicks",
    # clothing
    "hoodie", "hoodies", "jacket", "jackets", "shirt", "shirts", "shorts", "pants",
    "clothing", "clothes", "wear", "apparel", "tee", "tees", "pullover",
    "sweatshirt", "windbreaker", "windbreakers", "outerwear", "leggings", "tights",
    "dress", "jersey", "sweater", "top", "outfit",
    # accessories
    "socks", "sock", "gloves", "glove", "cap", "caps", "hat", "hats",
    "beanie", "beanies", "bag", "bags", "backpack", "backpacks",
    "accessories", "accessory", "gear",
    # generic query words
    "and", "or", "the", "a", "an", "for", "me", "please", "show", "find",
    "get", "want", "need", "some", "any", "under", "below", "above", "around",
    "with", "in", "on", "at", "of", "to", "my", "i", "like", "give",
}

# ===== Entity synonym map =====
# Maps every user-facing synonym → canonical dataset search term.
# Covers all Dialogflow entity entries so nothing gets missed.
# Structure: { "user synonym": "canonical_term_to_search" }
ENTITY_SYNONYM_MAP = {
    # ── Footwear ──
    "footwear": "shoes",
    "shoes": "shoes",
    "shoe": "shoes",
    "sneakers": "shoes",
    "sneaker": "shoes",
    "trainers": "shoes",
    "trainer": "shoes",
    "running shoes": "shoes",
    "sport shoes": "shoes",
    "kicks": "shoes",
    "joggers": "shoes",       # in footwear context
    "slides": "slides",
    # ── Hoodie ──
    "hoodie": "hoodie",
    "hoodies": "hoodie",
    "sweatshirt": "hoodie",
    "sweatshort": "hoodie",
    "pullover": "hoodie",
    "zip up hoodie": "hoodie",
    "zip hoodie": "hoodie",
    # ── T-shirt ──
    "t shirt": "t-shirt",
    "t-shirt": "t-shirt",
    "tshirt": "t-shirt",
    "tee": "t-shirt",
    "tees": "t-shirt",
    "shirt": "shirt",
    "shirts": "shirt",
    "short sleeve shirt": "shirt",
    "clothes": "clothing",
    # ── Jacket ──
    "jacket": "jacket",
    "jackets": "jacket",
    "coat": "jacket",
    "coats": "jacket",
    "windbreaker": "windbreaker",
    "windbreakers": "windbreaker",
    "outerwear": "jacket",
    "outerwears": "jacket",
    # ── Pants ──
    "pants": "pants",
    "trousers": "pants",
    "joggers pants": "pants",
    "track pants": "pants",
    "sweatpants": "pants",
    "trouser": "pants",
    "jogger": "pants",
    "track pant": "pants",
    "sweatpant": "pants",
    # ── Shorts ──
    "shorts": "shorts",
    "sport shorts": "shorts",
    "running shorts": "shorts",
    # ── Socks ──
    "socks": "socks",
    "sock": "socks",
    "ankle socks": "socks",
    "sports socks": "socks",
    "crew socks": "socks",
    # ── Bag ──
    "bag": "bag",
    "bags": "bag",
    "backpack": "backpack",
    "backpacks": "backpack",
    "gym bag": "bag",
    "gym bags": "bag",
    "duffel bag": "duffel",
    "duffel bags": "duffel",
    "duffle bags": "duffel",
    "sack": "bag",
    "sacks": "bag",
    "gym sacks": "bag",
    "gym sack": "bag",
    # ── Cap / Hat ──
    "cap": "cap",
    "caps": "cap",
    "hat": "hat",
    "hats": "hat",
    "beanie": "beanie",
    "beanies": "beanie",
    "headwear": "cap",
    "headwears": "cap",
    # ── Gloves ──
    "gloves": "gloves",
    "glove": "gloves",
    # ── Ball ──
    "ball": "ball",
    "balls": "ball",
    "football": "ball",
    "soccer ball": "ball",
    "basketball ball": "ball",
    # ── Accessories ──
    "accessories": "accessories",
    "accessory": "accessories",
    "gear": "accessories",
    # ── Activity subcategories ──
    "running": "running",
    "run": "running",
    "jog": "running",
    "jogging": "running",
    "training": "training",
    "gym": "training",
    "workout": "training",
    "exercise": "training",
    "soccer": "soccer",
    "football boots": "soccer",
    "cleats": "soccer",
    "cleat": "soccer",
    "golf": "golf",
    "basketball": "basketball",
    "climbing": "climbing",
    "cycling": "cycling",
    "bike": "cycling",
    "hiking": "hiking",
    "hike": "hiking",
    "trail": "hiking",
    "casual": "casual",
    "lifestyle": "casual",
    "everyday": "casual",
}

# Reverse lookup: canonical → list of synonyms (for subcategory searching)
SUBCATEGORY_MAP = {
    "running": ["running", "run", "jog", "jogging"],
    "casual": ["casual", "lifestyle", "originals", "everyday"],
    "training": ["training", "gym", "workout", "exercise"],
    "soccer": ["soccer", "football", "cleat", "cleats"],
    "golf": ["golf"],
    "basketball": ["basketball"],
    "climbing": ["climbing", "hiangle", "kestrel"],
    "cycling": ["cycling", "bike", "mountain bike"],
    "slides": ["slide", "adilette", "sandal"],
    "sandals": ["sandal", "slide", "adilette"],
    "hiking": ["hiking", "hike", "trail"],
}


def resolve_entity_synonyms(query_text):
    """
    Scans query_text for any synonym in ENTITY_SYNONYM_MAP and returns
    a list of canonical search terms to filter on.
    Longest phrase matches take priority over single words.
    """
    if not query_text:
        return []
    text = query_text.lower().strip()
    found = {}
    # Check multi-word phrases first (longest first to avoid partial matches)
    sorted_synonyms = sorted(ENTITY_SYNONYM_MAP.keys(), key=len, reverse=True)
    for synonym in sorted_synonyms:
        pattern = r"\b" + re.escape(synonym) + r"\b"
        if re.search(pattern, text):
            canonical = ENTITY_SYNONYM_MAP[synonym]
            found[canonical] = True
    return list(found.keys())

# ===== Preference synonyms =====
# Normalizes messy user preference language into canonical sorts
PREFERENCE_SYNONYMS = {
    "cheap": ["cheap", "cheapest", "affordable", "budget", "inexpensive",
              "low price", "low-price", "low cost", "not expensive",
              "not too expensive", "not that expensive", "budget friendly",
              "budget-friendly", "value"],
    "best": ["best", "top rated", "top-rated", "highest rated", "highest-rated",
             "best rated", "best-rated", "good", "quality", "recommended",
             "most reviewed", "popular"],
    "expensive": ["expensive", "premium", "luxury", "high end", "high-end",
                  "most expensive", "priciest", "top price"],
    "discount": ["discount", "deal", "sale", "offer", "most savings",
                 "biggest discount", "on sale"],
}


# ===== Build a dynamic term index from the CSV at startup =====
def build_csv_term_index(dataframe):
    """
    Returns a set of lowercase single-word and two-word phrases that appear
    in product names/descriptions and are meaningful item-type descriptors.
    """
    text_cols = ["name", "description", "category"]

    noise = {
        "and", "or", "the", "a", "an", "for", "me", "please", "show", "find",
        "get", "want", "need", "some", "any", "under", "below", "above", "around",
        "with", "in", "on", "at", "of", "to", "my", "i", "like", "give",
        "size", "plus", "pairs", "set", "pack", "new", "best", "great", "good",
        "sport", "adidas", "color", "fit", "designed", "move", "future", "icons",
        "logo", "graphic", "classic", "essentials", "brand", "love", "repeat",
        "high", "low", "mid", "full", "zip", "slim", "loose", "cropped", "crop",
        "long", "short", "woven", "knit", "stretch", "print", "printed", "stripe",
        "stripes", "letter", "crew", "waist", "neck", "sleeve", "performance",
        "cushioned", "ultralight", "allover", "comfort", "soft", "primegreen",
        "primeblue", "aeroready", "heat", "rdy", "super", "ultra", "x", "k",
        "l", "w", "n", "no", "vs", "3", "2", "6", "7", "8", "20", "21",
    }

    term_set = set()

    for col in text_cols:
        if col not in dataframe.columns:
            continue
        for text in dataframe[col].dropna().astype(str):
            words = re.findall(r"[a-z]+", text.lower())
            for w in words:
                if len(w) >= 3 and w not in noise:
                    term_set.add(w)
            # Capture two-word descriptive phrases (e.g. "tank top", "swim shorts")
            for i in range(len(words) - 1):
                bigram = words[i] + " " + words[i + 1]
                if words[i] not in noise and words[i + 1] not in noise and len(bigram) >= 6:
                    term_set.add(bigram)

    return term_set


# Build the index once at startup
CSV_TERM_INDEX = build_csv_term_index(df)


def extract_query_item_terms(query_text, dataframe):
    """
    Scans the user query against the CSV term index and returns every
    item-type term that (a) appears in the query AND (b) actually matches
    at least one product in name, description, or category.
    """
    if not query_text:
        return []

    text = query_text.lower()
    found_terms = []

    for term in CSV_TERM_INDEX:
        pattern = r"\b" + re.escape(term) + r"\b"
        if not re.search(pattern, text):
            continue

        for col in ["name", "description", "category"]:
            if col not in dataframe.columns:
                continue
            if dataframe[col].str.contains(re.escape(term), case=False, na=False).any():
                found_terms.append(term)
                break

    # Remove shorter terms that are substrings of longer matched phrases
    found_terms.sort(key=len, reverse=True)
    deduped = []
    for term in found_terms:
        if not any(term in longer for longer in deduped):
            deduped.append(term)

    return deduped


def extract_terms_from_query_text(query_text, dataframe):
    """
    CSV-backed fallback search: when Dialogflow passes no useful parameters,
    this function independently searches product names AND descriptions using
    every meaningful token in the user query.
    Returns a boolean mask over `dataframe` rows that are relevant.
    """
    if not query_text:
        return pd.Series(False, index=dataframe.index)

    text = query_text.lower()

    stop = {
        "and", "or", "the", "a", "an", "for", "me", "please", "show", "find",
        "get", "want", "need", "some", "any", "under", "below", "above", "around",
        "with", "in", "on", "at", "of", "to", "my", "i", "like", "give", "both",
        "also", "can", "you", "do", "have", "what", "which", "is", "are", "just",
    }
    tokens = [w for w in re.findall(r"[a-z]+", text) if w not in stop and len(w) >= 3]

    if not tokens:
        return pd.Series(False, index=dataframe.index)

    combined = pd.Series(False, index=dataframe.index)
    for token in tokens:
        for col in ["name", "description"]:
            if col in dataframe.columns:
                combined |= dataframe[col].str.contains(re.escape(token), case=False, na=False)

    return combined


def detect_products_from_text(query_text, df):
    """
    Detects SPECIFIC named products (e.g. 'Superstar Shoes', 'ZX 1K Boost')
    by matching meaningful (non-generic) keywords from each unique product
    name against the query. Requires ALL non-generic words to match.
    """
    if not query_text:
        return []

    text = query_text.lower()
    matched = []

    if "name" not in df.columns:
        return matched

    for name in df["name"].dropna().unique():
        name_lower = str(name).lower()
        words = [w for w in name_lower.split() if w not in GENERIC_WORDS]

        if not words:
            continue

        match_count = sum(1 for w in words if w in text)

        if match_count == len(words):
            matched.append(name)

    return list(set(matched))


# ===== NEW: Normalize preference from raw query text =====
def detect_preference_from_text(query_text):
    """
    Extracts a canonical preference label (cheap / best / expensive / discount)
    from messy natural language.

    Handles conflicting signals like 'cheap but good' or 'best cheap':
    - If BOTH 'cheap' AND 'best' appear → prefer 'best' (quality wins ambiguity)
    - If 'cheap' AND 'expensive' both appear → prefer 'cheap'
    - Explicit 'not expensive' / 'affordable' → cheap
    - 'highly rated' / 'top rated' → best
    """
    if not query_text:
        return None

    text = query_text.lower()
    found = {}

    for canonical, synonyms in PREFERENCE_SYNONYMS.items():
        for syn in synonyms:
            if re.search(r"\b" + re.escape(syn) + r"\b", text):
                found[canonical] = True
                break

    if not found:
        return None

    # Conflict resolution
    if "best" in found and "cheap" in found:
        return "best"  # quality over price when both mentioned
    if "expensive" in found and "cheap" in found:
        return "cheap"  # assume they want affordable if both mentioned
    if "best" in found and "expensive" in found:
        return "best"  # "expensive but highly rated" → sort by rating

    # Single signal
    for pref in ["best", "cheap", "expensive", "discount"]:
        if pref in found:
            return pref

    return None


# ===== NEW: Detect subcategory from query =====
def detect_subcategory_from_text(query_text):
    """
    Detects shoe/product subcategory keywords in the query.
    Returns a list of keywords to filter on, or [].
    E.g. "running shoes" → ["running"]
         "basketball shoes" → ["basketball"]
    """
    if not query_text:
        return []

    text = query_text.lower()
    found = []

    for subcat, keywords in SUBCATEGORY_MAP.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                found.extend(keywords)
                break

    return list(set(found))


# ===== NEW: Detect top-level category from query =====
def detect_category_from_text(query_text):
    """
    Returns the most likely top-level category string for a query.
    Uses the full ENTITY_SYNONYM_MAP so every Dialogflow synonym is covered.
    E.g. "tshirt" → "Clothing", "windbreaker" → "Clothing", "backpack" → "Accessories"
    """
    if not query_text:
        return None

    text = query_text.lower()

    # Canonical terms that map to each top-level dataset category
    shoe_canonicals = {
        "shoes", "slides", "boots", "sandals", "running", "basketball",
        "soccer", "golf", "climbing", "cycling", "hiking", "casual", "training",
    }
    clothing_canonicals = {
        "hoodie", "shirt", "t-shirt", "jacket", "windbreaker", "pants",
        "shorts", "clothing", "dress", "leggings", "jersey",
    }
    accessory_canonicals = {
        "accessories", "bag", "backpack", "duffel", "cap", "hat",
        "beanie", "socks", "gloves", "ball",
    }

    counts = {"Shoes": 0, "Clothing": 0, "Accessories": 0}

    resolved = resolve_entity_synonyms(text)
    for canonical in resolved:
        if canonical in shoe_canonicals:
            counts["Shoes"] += 1
        elif canonical in clothing_canonicals:
            counts["Clothing"] += 1
        elif canonical in accessory_canonicals:
            counts["Accessories"] += 1

    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return None
    return best


# ===== NEW: Multi-segment query parser =====
def parse_multi_segment_query(query_text):
    """
    Detects queries asking for MULTIPLE product types with potentially
    different colors/prices per segment.

    Handles patterns like:
      "pink hoodies and burgundy duffel bags under 500"
      "black shoes and white t-shirts"

    Returns a list of segment dicts:
      [{"color": "pink", "product": "hoodies", "max_price": None},
       {"color": "burgundy", "product": "duffel bags", "max_price": 500}]

    Returns [] if no multi-segment pattern is detected.
    """
    if not query_text:
        return []

    text = query_text.lower()

    # Skip comparison queries — they use 'and' but are NOT multi-segment
    if is_comparison_query(text):
        return []

    # Split on conjunctions
    raw_segments = re.split(r"\band\b", text)

    if len(raw_segments) < 2:
        return []

    # Build color list dynamically from dataset
    colors_in_dataset = set()
    if "color" in df.columns:
        for c in df["color"].dropna().unique():
            for word in re.findall(r"[a-z]+", c.lower()):
                if len(word) >= 3:
                    colors_in_dataset.add(word)

    segments = []
    global_price = _extract_global_price(text)

    for seg in raw_segments:
        seg = seg.strip()
        color = None
        product = None
        max_price = None

        # Extract price from this segment or fall back to global price
        price_match = re.search(
            r"(?:under|below|less\s+than|up\s+to|max(?:imum)?|<)\s*\$?(\d+(?:\.\d+)?)"
            r"|(?:between\s+\$?(\d+(?:\.\d+)?)\s+and\s+\$?(\d+(?:\.\d+)?))"
            r"|\$?(\d+(?:\.\d+)?)\s*(?:or\s+)?(?:less|below|max)",
            seg
        )
        if price_match:
            nums = [float(x) for x in price_match.groups() if x is not None]
            max_price = max(nums) if nums else None
        elif global_price is not None:
            max_price = global_price

        # Extract color
        for color_word in colors_in_dataset:
            if re.search(r"\b" + re.escape(color_word) + r"\b", seg):
                color = color_word
                break

        # Extract product terms — try entity synonym map first, then CSV index
        entity_terms = resolve_entity_synonyms(seg)
        if entity_terms:
            product = entity_terms[0]
        else:
            item_terms = extract_query_item_terms(seg, df)
            if item_terms:
                product = item_terms[0]  # Take longest/most specific match

        if product:
            segments.append({
                "color": color,
                "product": product,
                "max_price": max_price,
                "raw": seg,
            })

    return segments if len(segments) >= 2 else []


def _extract_global_price(text):
    """Extract a single trailing max price from text, e.g. 'under 500'."""
    m = re.search(
        r"(?:under|below|less\s+than|up\s+to|max(?:imum)?|<)\s*\$?(\d+(?:\.\d+)?)"
        r"|\$?(\d+(?:\.\d+)?)\s*(?:or\s+)?(?:less|below|max)",
        text
    )
    if m:
        nums = [float(x) for x in m.groups() if x is not None]
        return max(nums) if nums else None
    return None


def search_segment(color=None, product=None, max_price=None, preference=None, gender=None):
    """
    Run a focused search for a single color+product+price+gender segment.
    Returns a sorted DataFrame.
    """
    results = df.copy()

    if color and "color" in results.columns:
        results = results[results["color"].str.contains(re.escape(color), case=False, na=False)]

    if product:
        product_mask = pd.Series(False, index=results.index)
        for col in ["name", "category", "description"]:
            if col in results.columns:
                product_mask |= results[col].str.contains(re.escape(product), case=False, na=False)
        results = results[product_mask]

    if max_price is not None and "selling_price" in results.columns:
        results = results[results["selling_price"].notna()]
        results = results[results["selling_price"] <= max_price]

    # Apply gender filter per segment
    if gender:
        results = apply_gender_filter(results, gender)

    if "original_price" in results.columns and "selling_price" in results.columns:
        results["discount"] = results["original_price"] - results["selling_price"]
    else:
        results["discount"] = 0

    results = results.sort_values(
        ["average_rating", "reviews_count"], ascending=False
    ).reset_index(drop=True)

    return results


# ===== NEW: Product comparison =====
def compare_products(query_text):
    """
    Detects two product model names in the query and returns a formatted
    side-by-side comparison including description, price, rating.

    Supports patterns like:
      "compare ultraboost and runfalcon"
      "ultraboost vs runfalcon"
      "difference between nmd and superstar"
    """
    text = query_text.lower()

    # Strip leading intent words before parsing for product terms
    clean = re.sub(
        r"^(?:compare|comparison(?:\s+between)?|difference\s+between|compare\s+between)\s+",
        "", text
    ).strip()

    # Patterns for comparison queries
    vs_pattern = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", clean)
    and_pattern = re.search(r"(.+?)\s+(?:and|with)\s+(.+)", clean)

    term_a, term_b = None, None

    if vs_pattern:
        term_a = vs_pattern.group(1).strip()
        term_b = vs_pattern.group(2).strip()
    elif and_pattern:
        term_a = and_pattern.group(1).strip()
        term_b = and_pattern.group(2).strip()

    if not term_a or not term_b:
        return None

    def find_best_match(term):
        mask = pd.Series(False, index=df.index)
        for col in ["name", "description"]:
            if col in df.columns:
                mask |= df[col].str.contains(re.escape(term), case=False, na=False)
        matched = df[mask]
        if matched.empty:
            return None
        return matched.sort_values(
            ["average_rating", "reviews_count"], ascending=False
        ).iloc[0]

    product_a = find_best_match(term_a)
    product_b = find_best_match(term_b)

    if product_a is None and product_b is None:
        return f"❌ Could not find products matching '{term_a}' or '{term_b}'."
    if product_a is None:
        return f"❌ Could not find a product matching '{term_a}'."
    if product_b is None:
        return f"❌ Could not find a product matching '{term_b}'."

    def _s(val, prefix="", suffix="", decimals=None):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        if decimals is not None:
            return f"{prefix}{float(val):.{decimals}f}{suffix}"
        return f"{prefix}{val}{suffix}"

    def _stars(rating):
        try:
            n = round(float(rating))
            return "⭐" * n + "☆" * (5 - n)
        except Exception:
            return ""

    def _stock(avail):
        a = str(avail).lower()
        return "✅ In Stock" if ("in stock" in a or a in ("true","1","yes","available")) else "❌ Out of Stock"

    def _discount(orig, sell):
        try:
            o, s = float(orig), float(sell)
            if o > s > 0:
                pct = int(round((o - s) / o * 100))
                return f"🏷️ {pct}% off"
            return ""
        except Exception:
            return ""

    def product_card(p, label):
        name   = p.get("name", "Unknown")
        cat    = _s(p.get("category"))
        color  = _s(p.get("color"))
        price  = _s(p.get("selling_price"), "$", "", 0)
        orig   = _s(p.get("original_price"), "$", "", 0)
        rating = _s(p.get("average_rating"), "", "/5", 1)
        reviews= _s(p.get("reviews_count"))
        avail  = _stock(p.get("availability", ""))
        disc   = _discount(p.get("original_price"), p.get("selling_price"))
        stars  = _stars(p.get("average_rating"))
        gender = infer_gender_from_row(p if isinstance(p, dict) else p.to_dict())
        desc   = _truncate(str(p.get("description", "N/A")), 220)

        price_line = f"💰 {price}"
        if disc:
            price_line += f"  {disc}  (was {orig})"

        card = (
            f"{label}\n"
            f"👟 {name}\n"
            f"\n"
            f"{avail}\n"
            f"🏷️ {cat}   🎨 {color}   👤 {gender}\n"
            f"{price_line}\n"
            f"{stars} {rating}  💬 {reviews} reviews\n"
            f"\n"
            f"📝 {desc}"
        )
        return card

    divider = "═" * 30
    winner_note = ""
    try:
        r_a = float(product_a.get("average_rating") or 0)
        r_b = float(product_b.get("average_rating") or 0)
        p_a = float(product_a.get("selling_price") or 0)
        p_b = float(product_b.get("selling_price") or 0)
        name_a = product_a.get("name", term_a)
        name_b = product_b.get("name", term_b)
        if r_a > r_b:
            winner_note = f"\n🏆 Better rated: {name_a}"
        elif r_b > r_a:
            winner_note = f"\n🏆 Better rated: {name_b}"
        if p_a and p_b:
            if p_a < p_b:
                winner_note += f"\n💸 Better value: {name_a}"
            elif p_b < p_a:
                winner_note += f"\n💸 Better value: {name_b}"
    except Exception:
        pass

    card_a = product_card(product_a, "🅰️ Product A")
    card_b = product_card(product_b, "🅱️ Product B")

    result = (
        f"📊 Comparison\n"
        f"{divider}\n"
        f"{card_a}\n"
        f"{divider}\n"
        f"{card_b}\n"
        f"{divider}"
        f"{winner_note}"
    )
    return result


def _truncate(text, max_len=300):
    """Truncate text to max_len characters, appending '...' if cut."""
    if not text or text == "nan":
        return "N/A"
    return text[:max_len] + ("..." if len(text) > max_len else "")


def get_product_image(row):
    """
    Extracts the first (main) product image URL from the images column.
    Images are stored as tilde-separated URLs.
    Returns None if no image available.
    """
    raw = row.get("images", "") if isinstance(row, dict) else str(row)
    if not raw or raw in ("nan", "None", ""):
        return None
    first = raw.split("~")[0].strip()
    return first if first.startswith("http") else None


# ===== NEW: Detect comparison intent from free text =====
def is_comparison_query(query_text):
    """Returns True if the query looks like a product comparison request."""
    text = query_text.lower()
    triggers = [
        r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b",
        r"\bdifference between\b", r"\bcompare between\b",
    ]
    for t in triggers:
        if re.search(t, text):
            return True
    return False


# ===== Infer gender from product row =====
def infer_gender_from_row(row):
    """
    Infers gender label from breadcrumbs, name, category, description.
    Returns 'Women', 'Men', 'Kids', or 'Unisex'.
    """
    text = " ".join([
        str(row.get("breadcrumbs", "")),
        str(row.get("name", "")),
        str(row.get("category", "")),
        str(row.get("description", "")),
    ]).lower()

    if re.search(r"\bwomen|\bwomens|\bladies|\bfemale|\bher\b|\bgirl", text):
        return "Women"
    if re.search(r"\bmen\b|\bmens\b|\bmale|\bhis\b|\bguy|\bboy\b", text):
        return "Men"
    if re.search(r"\bkids|\bjunior|\byouth|\bchild", text):
        return "Kids"
    return "Unisex"


# ===== Rich product detail card =====
def build_product_detail_card(row):
    """
    Builds a rich, Streamlit-style product detail card from a product row dict.
    """
    def _s(val, prefix="", suffix="", decimals=None):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        if decimals is not None:
            return f"{prefix}{float(val):.{decimals}f}{suffix}"
        return f"{prefix}{val}{suffix}"

    name     = row.get("name", "Unknown")
    category = _s(row.get("category"))
    color    = _s(row.get("color"))
    price    = _s(row.get("selling_price"), "$", "", 0)
    orig     = _s(row.get("original_price"), "$", "", 0)
    rating   = _s(row.get("average_rating"), "", "/5", 1)
    reviews  = _s(row.get("reviews_count"))
    avail    = str(row.get("availability", "N/A"))
    gender   = infer_gender_from_row(row)
    desc     = _truncate(str(row.get("description", "N/A")), 500)

    # Availability badge
    in_stock = "in stock" in avail.lower() or avail.lower() in ("true", "1", "yes", "available")
    stock_badge = "✅ In Stock" if in_stock else "❌ Out of Stock"

    # Discount badge
    try:
        orig_f = float(row.get("original_price", 0) or 0)
        sell_f = float(row.get("selling_price", 0) or 0)
        if orig_f > sell_f > 0:
            pct = int(round((orig_f - sell_f) / orig_f * 100))
            discount_line = f"\n🏷️ Discount    : {pct}% off  (was {orig})"
        else:
            discount_line = ""
    except Exception:
        discount_line = ""

    # Star rendering
    try:
        stars = round(float(rating.replace("/5", "")))
        star_str = "⭐" * stars + "☆" * (5 - stars)
    except Exception:
        star_str = ""

    image_url = get_product_image(row)

    card = (
        f"🛍️ {name}\n"
        f"\n"
        f"{stock_badge}\n"
        f"\n"
        f"💰 Price      : {price}{discount_line}\n"
        f"🏷️ Category   : {category}   🎨 Color: {color}\n"
        f"👤 Gender     : {gender}\n"
        f"{star_str} Rating: {rating}\n"
        f"💬 Reviews    : {reviews}\n"
        f"\n"
        f"📝 Description:\n"
        f"{desc}"
    )
    return card, image_url


# ===== NEW: Product detail / info =====
def get_product_detail(query_text):
    """
    Returns full detail for a single named product found in the query.
    Used when user asks 'tell me about X' or 'what is X'.
    """
    text = query_text.lower()
    matched = detect_products_from_text(text, df)
    if not matched:
        if "name" in df.columns:
            for name in df["name"].dropna().unique():
                if str(name).lower() in text:
                    matched = [name]
                    break

    if not matched:
        return None

    target = matched[0]
    mask = df["name"].str.contains(re.escape(target), case=False, na=False)
    row = df[mask].sort_values(["average_rating", "reviews_count"], ascending=False).iloc[0]
    card_text, image_url = build_product_detail_card(row.to_dict())
    return card_text, image_url


# ===== NEW: Suggest similar products =====
def suggest_similar(product_name, exclude_names=None, top_n=3):
    """
    Given a product name, suggests similar products from the same category.
    Optionally excludes a list of already-shown names.
    """
    if "name" not in df.columns or "category" not in df.columns:
        return []

    mask = df["name"].str.contains(re.escape(product_name), case=False, na=False)
    if not mask.any():
        return []

    category = df[mask].iloc[0].get("category", None)
    if not category or category == "nan":
        return []

    similar = df[df["category"].str.contains(re.escape(str(category)), case=False, na=False)]

    if exclude_names:
        for exc in exclude_names:
            similar = similar[~similar["name"].str.contains(re.escape(exc), case=False, na=False)]

    similar = similar[~mask]
    similar = similar.sort_values(["average_rating", "reviews_count"], ascending=False)
    return similar.head(top_n)


def get_param(params, *names):
    """Return the first non-empty parameter value from a list of possible names."""
    for name in names:
        value = params.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def parse_price_range(value):
    """
    Accepts things like:
    - 100
    - 'under 100'
    - '100-200'
    - 'below 150'
    - 'above 80' / 'over 80'
    - 'between 50 and 150'
    Returns: (min_price, max_price)
    """
    if value is None:
        return None, None

    if isinstance(value, (int, float)):
        return None, float(value)

    text = str(value).lower().replace(",", "")
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]

    if not numbers:
        return None, None

    if len(numbers) == 1:
        if any(k in text for k in ["under", "below", "less than", "up to", "max", "maximum"]):
            return None, numbers[0]
        if any(k in text for k in ["above", "over", "more than", "min", "minimum", "at least"]):
            return numbers[0], None
        return numbers[0], numbers[0]

    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


# ===== NEW: Parse price constraints directly from free text =====
def parse_price_from_text(text):
    """
    Extract (min_price, max_price) directly from raw query text.
    Handles: 'under 100', 'below 50', 'above 80', 'between 50 and 150',
             'shoes above 80', 'over 200', '50 to 150', 'less than 120',
             'around 80' (treat as ±20% band)
    Returns (min_price, max_price) — either may be None.
    """
    if not text:
        return None, None

    t = text.lower().replace(",", "")

    # between X and Y
    m = re.search(r"between\s+\$?(\d+(?:\.\d+)?)\s+and\s+\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))

    # X to Y
    m = re.search(r"\$?(\d+(?:\.\d+)?)\s+to\s+\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))

    # under / below / less than / up to / max
    m = re.search(
        r"(?:under|below|less\s+than|up\s+to|max(?:imum)?|no\s+more\s+than|<)\s*\$?(\d+(?:\.\d+)?)", t
    )
    if m:
        return None, float(m.group(1))

    # X or less / X or below
    m = re.search(r"\$?(\d+(?:\.\d+)?)\s+or\s+(?:less|below)", t)
    if m:
        return None, float(m.group(1))

    # above / over / more than / at least / min
    m = re.search(
        r"(?:above|over|more\s+than|at\s+least|min(?:imum)?|>)\s*\$?(\d+(?:\.\d+)?)", t
    )
    if m:
        return float(m.group(1)), None

    # around / approximately (±20%)
    m = re.search(r"(?:around|approximately|about|~)\s*\$?(\d+(?:\.\d+)?)", t)
    if m:
        mid = float(m.group(1))
        return mid * 0.8, mid * 1.2

    return None, None


# ===== NEW: Deduplicate repeated price/color/word tokens in query =====
def sanitize_query(query_text):
    """
    Cleans up malformed queries:
    - 'black black shoes' → 'black shoes'
    - 'shoes under under 100' → 'shoes under 100'
    - 'more more more more' → 'show more'
    - quick-reply chip labels → canonical search terms
    - '1.', '#2', 'select 3' → '1', '2', '3'
    Returns cleaned string.
    """
    if not query_text:
        return query_text

    text = query_text.strip()

    # Strip emoji and map chip labels (case-insensitive, strip leading emoji)
    stripped = re.sub(r"[^\w\s]", "", text).strip().lower()
    chip_map = {
        "women": "women",
        "men": "men",
        "kids": "kids",
        "shoes": "shoes",
        "clothing": "clothing",
        "accessories": "accessories",
        "categories": "show all categories",
        "search again": "help",
        "show more": "show more",
        # "back to results" intentionally NOT here — handled by BACK_TO_RESULTS_PATTERN
        "view wishlist": "show my wishlist",
        "clear wishlist": "clear wishlist",
        "save this": "save this",
        "surprise me": "surprise me",
        "surprise me again": "surprise me",
    }
    if stripped in chip_map:
        return chip_map[stripped]

    text = text.lower().strip()

    # Normalise product selection: "View 1", "1.", "#1", "select 1" etc. → "1"
    num_match = re.match(
        r"^(?:view\s+|#\s*|select\s+|option\s+|number\s+|item\s+|pick\s+)?([1-9])[\.\s]*$",
        text
    )
    if num_match:
        return num_match.group(1)

    # 'more more more' → treat as 'show more'
    if re.match(r"^(?:more\s+)+$", text):
        return "show more"

    # Remove consecutive duplicate words: "black black" → "black"
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text)

    # Remove consecutive duplicate price keywords: "under under" → "under"
    text = re.sub(r"\b(under|below|above|over|between)\s+\1\b", r"\1", text)

    # ── Typo correction for common product words ──
    # Maps misspelled variants → correct word using whole-word regex replacement.
    # Only correct when the typo is an obvious 1-2 char transposition/omission.
    TYPO_MAP = {
        # shoes
        r"\bshos\b": "shoes",
        r"\bsheos\b": "shoes",
        r"\bshose\b": "shoes",
        r"\bshes\b": "shoes",
        r"\bshoees\b": "shoes",
        r"\bsnaekers\b": "sneakers",
        r"\bsnekaers\b": "sneakers",
        r"\bsneaker\b": "sneaker",
        # hoodie
        r"\bhoodi\b": "hoodie",
        r"\bhooide\b": "hoodie",
        r"\bhoodie\b": "hoodie",
        r"\bhoodei\b": "hoodie",
        # jacket
        r"\bjackt\b": "jacket",
        r"\bjakcet\b": "jacket",
        r"\bjacket\b": "jacket",
        # hiking
        r"\bhiking\b": "hiking",
        r"\bhikng\b": "hiking",
        r"\bhikig\b": "hiking",
        r"\bhikking\b": "hiking",
        # running
        r"\brunning\b": "running",
        r"\brunng\b": "running",
        r"\brnning\b": "running",
        # training
        r"\btraning\b": "training",
        r"\btrainig\b": "training",
        # backpack
        r"\bbackpak\b": "backpack",
        r"\bbackpack\b": "backpack",
        # clothing
        r"\bclothing\b": "clothing",
        r"\bclothng\b": "clothing",
        r"\bcloting\b": "clothing",
        # casual
        r"\bcasual\b": "casual",
        r"\bcasul\b": "casual",
        r"\bcasaul\b": "casual",
        # sandals
        r"\bsandels\b": "sandals",
        r"\bsandal\b": "sandal",
        # pants
        r"\bpants\b": "pants",
        r"\bpant\b": "pant",
        # shorts
        r"\bshorts\b": "shorts",
        r"\bshort\b": "short",
    }
    for pattern, replacement in TYPO_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


# ===== Gender detection =====
GENDER_KEYWORDS = {
    "women": ["women", "woman", "female", "ladies", "girl", "girls", "womens", "her"],
    "men":   ["men", "man", "male", "guys", "guy", "mens", "his", "boys"],
    "kids":  ["kids", "kid", "children", "child", "junior", "youth", "boys", "girls"],
}

def detect_gender_from_text(query_text):
    """
    Returns 'women', 'men', 'kids', or None based on keywords in the query.
    """
    if not query_text:
        return None
    text = query_text.lower()
    for gender, keywords in GENDER_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                return gender
    return None


def apply_gender_filter(results, gender):
    """
    Filters a DataFrame by gender using breadcrumbs first (most reliable),
    then name/description as fallback. Searches for Women/Men/Kids breadcrumb paths.
    Returns empty DataFrame if no gender-tagged products found — never falls back
    to unfiltered results (that caused wrong products like Slides appearing for "women bags").
    """
    if not gender or results.empty:
        return results

    gender_mask = pd.Series(False, index=results.index)

    # Priority 1: breadcrumbs (e.g. "Women/Accessories", "Men/Shoes")
    if "breadcrumbs" in results.columns:
        gender_mask |= results["breadcrumbs"].str.contains(gender, case=False, na=False)

    # Priority 2: name contains gender word (e.g. "Women's Running Shoe")
    if "name" in results.columns:
        gender_mask |= results["name"].str.contains(
            r"\b" + re.escape(gender) + r"\b", case=False, na=False, regex=True
        )

    # Priority 3: description only if nothing found yet
    if not gender_mask.any() and "description" in results.columns:
        gender_mask |= results["description"].str.contains(gender, case=False, na=False)

    return results[gender_mask]  # returns empty df if no match — caller handles it


def strict_entity_filter(results, term):
    """
    Filters results for a product term using NAME + CATEGORY first (strict).
    Falls back to description/breadcrumbs only if strict search finds nothing.
    This prevents "bag" matching Slides/Tees that mention bags in their description.
    """
    variants = {term}
    if term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])
    else:
        variants.add(term + "s")
    canonical = ENTITY_SYNONYM_MAP.get(term, None)
    if canonical:
        variants.add(canonical)
        if canonical.endswith("s") and len(canonical) > 3:
            variants.add(canonical[:-1])
        else:
            variants.add(canonical + "s")

    strict_mask = pd.Series(False, index=results.index)
    for t in variants:
        for col in ["name", "category"]:
            if col in results.columns:
                strict_mask |= results[col].str.contains(re.escape(t), case=False, na=False)
    if strict_mask.any():
        return results[strict_mask]

    # Fallback: description + breadcrumbs
    loose_mask = pd.Series(False, index=results.index)
    for t in variants:
        for col in ["description", "breadcrumbs"]:
            if col in results.columns:
                loose_mask |= results[col].str.contains(re.escape(t), case=False, na=False)
    return results[loose_mask] if loose_mask.any() else results


def format_product(row, index=None):
    name = row.get("name", "Unknown")
    price = row.get("selling_price", None)
    orig_price = row.get("original_price", None)
    rating = row.get("average_rating", None)
    reviews = row.get("reviews_count", None)
    category = row.get("category", "")
    color = row.get("color", "")

    # Infer gender for list display
    row_dict = row.to_dict() if hasattr(row, "to_dict") else row
    gender = infer_gender_from_row(row_dict)
    gender_text = f"👤 {gender}" if gender and gender != "Unisex" else ""

    price_text = f"${price:.0f}" if pd.notna(price) else "N/A"
    rating_text = f"⭐ {rating:.1f}" if pd.notna(rating) else ""
    reviews_text = f"({int(reviews)} reviews)" if pd.notna(reviews) and reviews else ""
    color_text = f"🎨 {color}" if color and color != "nan" else ""
    category_text = f"🏷️ {category}" if category and category != "nan" else ""

    # Discount badge
    discount_text = ""
    try:
        if pd.notna(price) and pd.notna(orig_price) and float(orig_price) > float(price) > 0:
            pct = int(round((float(orig_price) - float(price)) / float(orig_price) * 100))
            if pct >= 5:
                discount_text = f"  🏷️ -{pct}%"
    except Exception:
        pass

    prefix = f"{index}. " if index is not None else ""
    parts = [f"{prefix}👟 {name}"]
    if category_text:
        parts.append(f"   {category_text}")
    if color_text:
        parts.append(f"   {color_text}")
    if gender_text:
        parts.append(f"   {gender_text}")
    parts.append(f"   💰 {price_text}{discount_text}  {rating_text} {reviews_text}".rstrip())
    return "\n".join(parts)


def search_products(params, query_text=""):
    """
    Core search function. Filters df based on Dialogflow params,
    with additional price parsing, subcategory detection, and
    preference normalization from raw query text as fallback.
    """
    results = df.copy()

    brand = get_param(params, "brand")
    color = get_param(params, "color")
    products = get_param(params, "products")
    usage = get_param(params, "usage")
    preference = get_param(params, "preference")
    max_price = get_param(params, "max_price")
    price_range = get_param(params, "price_range")

    # ===== BRAND FILTER =====
    if brand and str(brand).lower() not in ("adidas", ""):
        return pd.DataFrame()

    # ===== COLOR FILTER =====
    # Build the set of valid colors once from the dataset
    valid_colors = set(c.lower() for c in df["color"].dropna().unique())

    if color and "color" in results.columns:
        color_lower = str(color).lower().strip()
        # Validate: reject if the param value isn't a real color (e.g. Dialogflow
        # sometimes extracts the word "color" itself as a @sys.color entity)
        if color_lower in valid_colors or any(color_lower in vc or vc in color_lower for vc in valid_colors):
            results = results[results["color"].str.contains(re.escape(color_lower), case=False, na=False)]
        # else: bad color param — skip color filter, let free-text detection handle it

    if "color" in results.columns and len(results) == len(df):
        # No color filter applied yet — try free-text detection
        if query_text:
            text_lower = query_text.lower()
            for color_word in sorted(valid_colors, key=len, reverse=True):  # longest first
                if re.search(r"\b" + re.escape(color_word) + r"\b", text_lower):
                    filtered = results[results["color"].str.contains(re.escape(color_word), case=False, na=False)]
                    if not filtered.empty:
                        results = filtered
                        break

    # ===== PRODUCT/CATEGORY FILTER =====
    if products:
        # Dialogflow param: resolve synonym to canonical form, then search
        products_str = str(products).lower().strip()
        products_resolved = ENTITY_SYNONYM_MAP.get(products_str, products_str)
        product_mask = pd.Series(False, index=results.index)
        for search_term in {products_resolved, products_str}:
            for col in ["name", "category", "description"]:
                if col in results.columns:
                    product_mask |= results[col].str.contains(re.escape(search_term), case=False, na=False)
        if product_mask.any():
            results = results[product_mask]
    elif query_text:
        # Free-text: use strict_entity_filter (name+category priority)
        entity_terms = resolve_entity_synonyms(query_text)
        if entity_terms:
            # Apply each entity term as a strict filter on the FULL current results
            # Use OR logic: a row matches if it matches ANY of the entity terms
            combined_mask = pd.Series(False, index=results.index)
            for term in entity_terms:
                candidate = strict_entity_filter(results, term)
                combined_mask |= pd.Series(results.index.isin(candidate.index), index=results.index)
            if combined_mask.any():
                results = results[combined_mask]
        else:
            detected_cat = detect_category_from_text(query_text)
            if detected_cat and "category" in results.columns:
                results = results[results["category"].str.contains(detected_cat, case=False, na=False)]

    # ===== SUBCATEGORY FILTER (from usage param or raw query) =====
    if usage:
        usage_str = str(usage).lower().strip()
        usage_resolved = ENTITY_SYNONYM_MAP.get(usage_str, usage_str)
        usage_mask = pd.Series(False, index=results.index)
        for search_term in {usage_resolved, usage_str}:
            for col in ["category", "description", "name", "breadcrumbs"]:
                if col in results.columns:
                    usage_mask |= results[col].str.contains(re.escape(search_term), case=False, na=False)
        if usage_mask.any():
            results = results[usage_mask]
    elif query_text:
        subcat_keywords = detect_subcategory_from_text(query_text)
        if subcat_keywords:
            subcat_mask = pd.Series(False, index=results.index)
            for kw in subcat_keywords:
                for col in ["name", "description", "breadcrumbs"]:
                    if col in results.columns:
                        subcat_mask |= results[col].str.contains(re.escape(kw), case=False, na=False)
            if subcat_mask.any():
                results = results[subcat_mask]

    # ===== PRICE FILTER =====
    min_price, max_price_range = parse_price_range(price_range)

    if max_price:
        try:
            max_price_range = float(max_price)
        except Exception:
            pass

    # Parse price from raw query text if params gave nothing
    if min_price is None and max_price_range is None and query_text:
        min_price, max_price_range = parse_price_from_text(query_text)

    if "selling_price" in results.columns:
        results = results[results["selling_price"].notna()]

        if min_price is not None:
            results = results[results["selling_price"] >= min_price]

        if max_price_range is not None:
            results = results[results["selling_price"] <= max_price_range]

    # ===== DISCOUNT CALCULATION =====
    if "original_price" in results.columns and "selling_price" in results.columns:
        results["discount"] = results["original_price"] - results["selling_price"]
    else:
        results["discount"] = 0

    # ===== SMART SORTING =====
    # Detect preference from params, then fall back to raw text detection
    effective_pref = preference
    if not effective_pref and query_text:
        effective_pref = detect_preference_from_text(query_text)

    if effective_pref:
        pref = str(effective_pref).lower()

        if "cheap" in pref:
            results = results.sort_values("selling_price")
        elif "expensive" in pref:
            results = results.sort_values("selling_price", ascending=False)
        elif "best" in pref or "rating" in pref:
            results = results.sort_values(["average_rating", "reviews_count"], ascending=False)
        elif "popular" in pref:
            results = results.sort_values("reviews_count", ascending=False)
        elif "discount" in pref or "deal" in pref:
            results = results.sort_values("discount", ascending=False)
        else:
            results = results.sort_values(["average_rating", "reviews_count"], ascending=False)
    else:
        results = results.sort_values(["average_rating", "reviews_count"], ascending=False)

    # ===== GENDER FILTER =====
    gender = detect_gender_from_text(query_text)
    if gender:
        gender_filtered = apply_gender_filter(results, gender)
        # Only apply gender filter if it found results; otherwise keep current
        # (some product types like "balls" may not have gender tags)
        if not gender_filtered.empty:
            results = gender_filtered

    return results.reset_index(drop=True)


# ===== NEW: Detect greetings =====
GREETING_PATTERNS = re.compile(
    r"^\s*(?:hi|hello|hey|howdy|what'?s\s+up|sup|good\s+(?:morning|afternoon|evening|day))[!.,?]*\s*$",
    re.IGNORECASE,
)

DETAIL_PATTERNS = re.compile(
    r"\b(?:tell\s+me\s+about|what\s+is|what'?s|describe|details?\s+(?:of|about|on)|info(?:rmation)?\s+(?:on|about))\b",
    re.IGNORECASE,
)

# ===== NEW: Random/nonsense request detection =====
NONSENSE_PATTERNS = re.compile(
    r"\b(?:invisible|transparent|rainbow|imaginary|impossible|magic|unicorn|fake|nonexistent)\b",
    re.IGNORECASE,
)


def rebuild_results_message(shown_products, header="🔥 Top picks for you:", has_more=False):
    """
    Re-renders the product list message from a list of product row dicts.
    Used by 'Back to Results' to replay the exact same page the user was on.
    """
    separator = "─" * 28
    lines = [format_product(row, index=i + 1) for i, row in enumerate(shown_products)]
    body = f"\n{separator}\n".join(lines)
    message = f"{header}\n\n{body}"
    if has_more:
        message += f"\n{separator}\n💬 Say 'show more' to see more."
    chips = [f"View {i+1}" for i in range(len(shown_products))]
    if has_more:
        chips.append("Show More")
    return message, chips


def add_to_wishlist(session_id, product_row):
    """Adds a product dict to the session wishlist. Avoids duplicates by name."""
    wishlist = WISHLIST_CACHE.setdefault(session_id, [])
    name = product_row.get("name", "")
    if not any(p.get("name") == name for p in wishlist):
        wishlist.append(product_row)
        return True  # added
    return False  # already in wishlist


def format_wishlist(session_id):
    """Returns a formatted string of the wishlist for display."""
    wishlist = WISHLIST_CACHE.get(session_id, [])
    if not wishlist:
        return "💔 Your wishlist is empty. Browse products and tap 'Save this' to add items!", []
    separator = "─" * 28
    lines = [format_product(row, index=i + 1) for i, row in enumerate(wishlist)]
    body = f"\n{separator}\n".join(lines)
    chips = [f"View {i+1}" for i in range(len(wishlist))]
    chips.append("Clear Wishlist")
    return f"💖 Your Wishlist ({len(wishlist)} items):\n\n{body}", chips


def build_response(text, quick_replies=None, cards=None):
    """
    Builds a Dialogflow Messenger-compatible fulfillmentMessages response.

    - text: newline-delimited string → each non-empty line becomes a text bubble
    - quick_replies: list of chip labels shown as tappable buttons
    - cards: list of dicts with keys: title, subtitle, imageUri, (optional) buttons
      Each card renders as a rich product card with image in Dialogflow Messenger
    """
    messages = []

    # Add image cards FIRST so they appear above the text description
    if cards:
        for card in cards:
            card_payload = {
                "title": card.get("title", ""),
                "subtitle": card.get("subtitle", ""),
            }
            if card.get("imageUri"):
                card_payload["imageUri"] = card["imageUri"]
            if card.get("buttons"):
                card_payload["buttons"] = card["buttons"]
            messages.append({"card": card_payload})

    # Add text bubbles
    lines = [line for line in text.split("\n") if line.strip()]
    messages += [{"text": {"text": [line]}} for line in lines]

    # Add quick-reply chips last
    if quick_replies:
        messages.append({
            "quickReplies": {
                "title": "Quick options:",
                "quickReplies": quick_replies
            }
        })

    return jsonify({
        "fulfillmentMessages": messages,
        "fulfillmentText": text,
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True) or {}

    query_result = req.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName", "")
    params = query_result.get("parameters", {})
    session_id = req.get("session", "default-session")
    query_text = query_result.get("queryText", "")

    # ── Sanitize malformed / repeated queries ──
    query_text = sanitize_query(query_text)

    # ===== GREETING =====
    if GREETING_PATTERNS.match(query_text) or intent_name == INTENT_WELCOME:
        return build_response(
            "👋 Hi! Welcome to the Adidas USA Store.\n"
            "\n"
            "🛍️ Here's what I can do for you:\n"
            "\n"
            "1️⃣  Browse by gender\n"
            "   → Type: women shoes  /  men hoodies  /  kids clothing\n"
            "\n"
            "2️⃣  Search by category & color\n"
            "   → Type: black running shoes under $100\n"
            "\n"
            "3️⃣  Sort by preference\n"
            "   → Type: cheapest shoes  /  best rated hoodies\n"
            "\n"
            "4️⃣  Compare products\n"
            "   → Type: compare Ultraboost and Runfalcon\n"
            "\n"
            "5️⃣  Get product details\n"
            "   → After results appear, reply with the number (1, 2 or 3)\n"
            "\n"
            "6️⃣  Save favourites\n"
            "   → View a product, then tap 'Save this' to wishlist it\n"
            "\n"
            "7️⃣  Feeling lucky?\n"
            "   → Type: surprise me\n"
            "\n"
            "💡 Type 'help' anytime to see this guide again.",
            quick_replies=["👩 Women", "👨 Men", "👟 Shoes", "👕 Clothing", "🎒 Accessories", "🎲 Surprise Me"]
        )

    # ===== HELP =====
    if intent_name == INTENT_HELP:
        return build_response(
                "🆘 Here's what I can do:\n"
                "\n"
                "🔍 Search by category & price\n"
                "   e.g. shoes under 100\n"
                "\n"
                "🎨 Filter by color\n"
                "   e.g. black running shoes\n"
                "\n"
                "⭐ Sort by preference\n"
                "   e.g. cheap clothing / best accessories\n"
                "\n"
                "💰 Price range\n"
                "   e.g. shoes between 50 and 150\n"
                "\n"
                "📊 Compare products\n"
                "   e.g. compare Ultraboost and Runfalcon\n"
                "\n"
                "📂 Browse categories\n"
                "   Type: show all categories\n"
                "\n"
                "💖 Save favourites\n"
                "   View a product → tap 'Save this'\n"
                "   Type: show my wishlist\n"
                "\n"
                "🎲 Random pick\n"
                "   Type: surprise me\n"
                "\n"
                "➡️ See more results\n"
                "   Type: show more\n"
                "\n"
                "⬅️ Go back\n"
                "   Type: back to results",
                quick_replies=["🎲 Surprise Me", "💖 View Wishlist", "👟 Shoes", "👕 Clothing"]
            )

    # ===== GOODBYE =====
    if intent_name == INTENT_GOODBYE:
        return build_response("👋 Bye! Come back anytime if you want to search for more products.")

    # ===== NEGATIVE =====
    if intent_name == INTENT_NEGATIVE:
        return build_response("Okay — tell me another color, brand, category, or budget and I'll search again.")

    # ===== LIST CATEGORIES =====
    if intent_name == INTENT_LIST_CATEGORIES:
        if "category" not in df.columns:
            return build_response("I cannot find categories in the dataset.")

        categories = (
            df["category"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .str.lower()
            .drop_duplicates()
            .head(15)
            .tolist()
        )

        if not categories:
            return build_response("No categories found in the dataset.")

        return build_response("📂 Here are some categories I found:\n- " + "\n- ".join(categories))

    # ===== COMPARE PRODUCTS =====
    if intent_name == INTENT_COMPARE or is_comparison_query(query_text):
        result = compare_products(query_text)
        if result:
            return build_response(result)
        return build_response("❌ I couldn't identify two products to compare. Try: 'compare Ultraboost and Runfalcon'.")

    # ===== PRODUCT DETAIL =====
    if intent_name == INTENT_PRODUCT_DETAIL or DETAIL_PATTERNS.search(query_text):
        detail_result = get_product_detail(query_text)
        if detail_result:
            detail_text, detail_img = detail_result
            rich_card = [{"title": query_text.title(), "imageUri": detail_img}] if detail_img else None
            return build_response(detail_text, cards=rich_card,
                                  quick_replies=["« Back to results", "Show More", "📂 Categories"])
        # Fall through to product search if no specific product found

    # ===== WISHLIST: VIEW =====
    if WISHLIST_VIEW_PATTERN.search(query_text):
        wl_text, wl_chips = format_wishlist(session_id)
        # Put wishlist items into shown_products so View 1/2/3 works on them
        wishlist = WISHLIST_CACHE.get(session_id, [])
        if wishlist:
            cache = SESSION_CACHE.setdefault(session_id, {})
            cache["shown_products"] = wishlist
        return build_response(wl_text, quick_replies=wl_chips if wl_chips else ["👟 Shoes", "👕 Clothing"])

    # ===== WISHLIST: CLEAR =====
    if WISHLIST_CLEAR_PATTERN.search(query_text):
        WISHLIST_CACHE[session_id] = []
        return build_response(
            "🗑️ Wishlist cleared! Start browsing to add new items.",
            quick_replies=["👩 Women", "👨 Men", "👟 Shoes", "👕 Clothing"]
        )

    # ===== WISHLIST: SAVE CURRENT PRODUCT =====
    if WISHLIST_ADD_PATTERN.search(query_text):
        cache = SESSION_CACHE.get(session_id, {})
        shown_products = cache.get("shown_products", [])
        # Save the most recently viewed single product (last detail view)
        last_viewed = cache.get("last_viewed_product")
        target = last_viewed or (shown_products[0] if shown_products else None)
        if target:
            added = add_to_wishlist(session_id, target)
            name = target.get("name", "Product")
            msg = (
                f"💖 Added '{name}' to your wishlist!\n"
                f"You have {len(WISHLIST_CACHE.get(session_id, []))} item(s) saved."
                if added else
                f"✅ '{name}' is already in your wishlist."
            )
            return build_response(msg, quick_replies=["💖 View Wishlist", "Show More", "« Back to results"])
        return build_response(
            "😕 No product selected to save. Browse and view a product first, then say 'save this'.",
            quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"]
        )

    # ===== SURPRISE ME =====
    if SURPRISE_PATTERN.search(query_text):
        import random
        surprise = df[df["selling_price"].notna()].copy()
        # Filter to well-rated products only
        surprise = surprise[surprise["average_rating"] >= 4.0] if "average_rating" in surprise.columns else surprise
        if surprise.empty:
            surprise = df[df["selling_price"].notna()].copy()
        pick = surprise.sample(1).iloc[0]
        card_text, image_url = build_product_detail_card(pick.to_dict())
        rich_card = [{
            "title": pick.get("name", "Product"),
            "subtitle": f"${float(pick.get('selling_price', 0) or 0):.0f}  ⭐ {pick.get('average_rating', '')}",
            "imageUri": image_url,
        }] if image_url else None
        # Store as shown so user can save/compare it
        cache = SESSION_CACHE.setdefault(session_id, {})
        cache["shown_products"] = [pick.to_dict()]
        cache["last_viewed_product"] = pick.to_dict()
        intro = "🎲 Here's a random top-rated pick for you!\n\n"
        return build_response(
            intro + card_text,
            quick_replies=["🎲 Surprise Me Again", "💖 Save this", "Show More", "« Back to results"],
            cards=rich_card
        )

    # ===== BACK TO RESULTS =====
    if BACK_TO_RESULTS_PATTERN.search(query_text):
        cache = SESSION_CACHE.get(session_id, {})
        last_msg = cache.get("last_result_message")
        last_chips = cache.get("last_result_chips")
        shown_products = cache.get("shown_products", [])
        if last_msg and shown_products:
            return build_response(last_msg, quick_replies=last_chips or [])
        return build_response(
            "😕 No previous results to go back to. Try a new search!",
            quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"]
        )

    # ===== SHOW MORE (check early for "show more" phrasing redirected by sanitize) =====
    # NOTE: bare '\bmore\b' is intentionally NOT included here — it is too greedy and
    # catches phrases like "with more message", "more details", "show more options" mid-sentence.
    # Standalone "show more", "more results", "next" are safe to match anywhere.
    # Bare "more" or "next" alone (full query) are handled by the _qt_lower exact-match check.
    _qt_lower = query_text.lower().strip()
    _is_show_more = (
        intent_name == INTENT_SHOW_MORE
        or re.search(r"\bshow\s+more\b|\bmore\s+results?\b", _qt_lower)
        or _qt_lower in ("more", "next", "show more")
    )
    if _is_show_more:
        cache = SESSION_CACHE.get(session_id)

        if not cache:
            return build_response("Please search for a product first before asking for more.")

        # If user just says "more", reuse previous search
        if not any(v for v in params.values() if v not in (None, "", [], {})):
            params = cache.get("last_params", {})
            query_text_for_more = cache.get("last_query", "")
        else:
            query_text_for_more = query_text

        # ── Multi-segment show more ──
        cached_segments = cache.get("segments")
        if cached_segments:
            all_lines = []
            shown_ids = cache.get("shown_ids", [])

            for seg in cached_segments:
                seg_results = search_segment(
                    color=seg.get("color"),
                    product=seg.get("product"),
                    max_price=seg.get("max_price"),
                )
                seg_results = seg_results[~seg_results.index.isin(shown_ids)]

                color_label = seg.get("color", "").title() if seg.get("color") else ""
                product_label = seg.get("product", "").title()
                header = f"🛍️ More {color_label} {product_label}".strip() + ":"

                if seg_results.empty:
                    all_lines.append(f"{header}\n✅ No more products.")
                else:
                    top = seg_results.head(PAGE_SIZE)
                    shown_ids.extend(top.index.tolist())
                    separator = "─" * 28
                    seg_lines = [format_product(row, index=i + 1) for i, (_, row) in enumerate(top.iterrows())]
                    body = f"\n{separator}\n".join(seg_lines)
                    all_lines.append(f"{header}\n\n{body}")

            cache["shown_ids"] = shown_ids

            if all(("No more" in l or "not found" in l.lower()) for l in all_lines):
                return build_response("✅ No more products found for your search.")

            multi_more_msg = "\n\n".join(all_lines)
            cache["last_result_message"] = multi_more_msg
            cache["last_result_chips"] = []  # no chips for multi-segment more
            return build_response(multi_more_msg)

        # ── Single-segment show more ──
        results = search_products(params, query_text_for_more)

        # Color availability check
        products_param = get_param(params, "products")
        color_param = get_param(params, "color")

        if products_param:
            product_only = df.copy()
            product_mask = pd.Series(False, index=product_only.index)
            for col in ["name", "category", "description"]:
                if col in product_only.columns:
                    product_mask |= product_only[col].str.contains(
                        str(products_param), case=False, na=False
                    )
            product_only = product_only[product_mask]

            if not product_only.empty and color_param:
                color_match = product_only[
                    product_only["color"].str.contains(str(color_param), case=False, na=False)
                ]
                if color_match.empty:
                    available_colors = (
                        product_only["color"].dropna().str.lower().unique().tolist()
                    )
                    return build_response(
                            f"😕 Sorry, we do not have {color_param} {products_param}.\n"
                            f"Available colors: {', '.join(available_colors[:6])}."
                        )

        item_terms_raw = extract_query_item_terms(query_text_for_more, df)
        entity_terms_direct = resolve_entity_synonyms(query_text_for_more)
        all_search_terms_more = set()
        for t in entity_terms_direct + item_terms_raw:
            canonical = ENTITY_SYNONYM_MAP.get(t, t)
            all_search_terms_more.update([t, canonical,
                t[:-1] if t.endswith("s") and len(t) > 3 else t + "s",
                canonical[:-1] if canonical.endswith("s") and len(canonical) > 3 else canonical + "s"])
        item_terms = list(all_search_terms_more)
        if item_terms:
            category_mask = pd.Series(False, index=results.index)
            for term in item_terms:
                for col in ["category", "name", "description", "breadcrumbs"]:
                    if col in results.columns:
                        category_mask |= results[col].str.contains(
                            re.escape(term), case=False, na=False
                        )
            if category_mask.any():
                results = results[category_mask]

        shown_ids = cache.get("shown_ids", [])
        results = results[~results.index.isin(shown_ids)]
        next_chunk = results.head(PAGE_SIZE)

        if next_chunk.empty:
            return build_response("✅ No more products found. Try a new search!")

        cache["shown_ids"].extend(next_chunk.index.tolist())

        shown_products_more = [row.to_dict() for _, row in next_chunk.iterrows()]
        cache["shown_products"] = shown_products_more
        lines = [format_product(row, index=i + 1) for i, (_, row) in enumerate(next_chunk.iterrows())]
        separator = "─" * 28
        body = f"\n{separator}\n".join(lines)
        message = f"📦 More results:\n\n{body}"
        chips_more = [f"View {i+1}" for i in range(len(next_chunk))]
        # Update back-to-results snapshot to this "more" page
        cache["last_result_message"] = message
        cache["last_result_chips"] = chips_more
        cache["last_result_header"] = "📦 More results:"
        return build_response(message, quick_replies=chips_more)

    # ===== SELECT PRODUCT BY NUMBER (click / type 1, 2, 3) =====
    number_match = re.match(r"^(?:view\s+)?([1-9])$", query_text.strip(), re.IGNORECASE)
    if number_match or intent_name == INTENT_SELECT_PRODUCT:
        cache = SESSION_CACHE.get(session_id, {})
        shown_products = cache.get("shown_products", [])
        if not shown_products:
            return build_response(
                "🔍 No recent results to select from.\n"
                "Please search for a product first, then reply with its number.",
                quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"]
            )
        # Resolve index
        if number_match:
            pick_index = int(number_match.group(1)) - 1
        else:
            try:
                raw_num = str(get_param(params, "number", "ordinal") or "1")
                pick_index = int(re.search(r"\d+", raw_num).group()) - 1
            except Exception:
                pick_index = 0
        # Clamp to valid range
        pick_index = max(0, min(pick_index, len(shown_products) - 1))
        row = shown_products[pick_index]
        card_text, image_url = build_product_detail_card(row)
        # Remember the last viewed product so "save this" knows what to save
        cache = SESSION_CACHE.get(session_id, {})
        cache["last_viewed_product"] = row
        SESSION_CACHE[session_id] = cache
        rich_card = [{
            "title": row.get("name", "Product"),
            "subtitle": f"${float(row.get('selling_price', 0) or 0):.0f}  ⭐ {row.get('average_rating', '')}",
            "imageUri": image_url,
        }] if image_url else None
        return build_response(card_text, quick_replies=["« Back to results", "💖 Save this", "Show More", "📂 Categories"], cards=rich_card)

    # ===== PRODUCT SEARCH =====
    if intent_name == INTENT_PRODUCT_SEARCH or intent_name not in (
        INTENT_SHOW_MORE, INTENT_LIST_CATEGORIES, INTENT_HELP,
        INTENT_GOODBYE, INTENT_NEGATIVE, INTENT_WELCOME,
        INTENT_COMPARE, INTENT_PRODUCT_DETAIL, INTENT_AVAILABILITY,
        INTENT_SELECT_PRODUCT, INTENT_GENDER_FILTER,
    ):
        # ── NONSENSE / impossible color/product check ──
        if NONSENSE_PATTERNS.search(query_text):
            return build_response(
                    "😅 Hmm, we don't carry that in our catalog!\n"
                    "Try a real color (black, white, red...) or product type.\n"
                    "Type 'show all categories' to browse what we have."
                )

        # ── MULTI-SEGMENT QUERY ──
        segments = parse_multi_segment_query(query_text)

        if segments:
            all_lines = []
            cache_ids = []
            all_shown_products = []  # flat list: all products shown across segments
            gender = detect_gender_from_text(query_text)  # shared gender for all segments

            for seg_idx, seg in enumerate(segments):
                seg_results = search_segment(
                    color=seg.get("color"),
                    product=seg.get("product"),
                    max_price=seg.get("max_price"),
                    gender=gender,
                )

                color_label = seg.get("color", "").title() if seg.get("color") else ""
                product_label = seg.get("product", "").title()
                gender_label = f" ({gender.title()})" if gender else ""
                header = f"🛍️ {color_label} {product_label}{gender_label}".strip() + ":"

                if seg_results.empty:
                    all_lines.append(f"{header}\n❌ No products found.")
                else:
                    top = seg_results.head(PAGE_SIZE)
                    cache_ids.extend(top.index.tolist())
                    seg_products = [row.to_dict() for _, row in top.iterrows()]
                    all_shown_products.extend(seg_products)
                    separator = "─" * 28
                    # Number each product globally across segments so View 1/2/3 map correctly
                    start_idx = len(all_shown_products) - len(seg_products)
                    seg_lines = [
                        format_product(row, index=start_idx + i + 1)
                        for i, (_, row) in enumerate(top.iterrows())
                    ]
                    body = f"\n{separator}\n".join(seg_lines)
                    entry = f"{header}\n\n{body}"
                    if len(seg_results) > PAGE_SIZE:
                        entry += f"\n{separator}\n💬 Say \'show more\' for more."
                    all_lines.append(entry)

            multi_message = "\n\n".join(all_lines)
            multi_chips = [f"View {i+1}" for i in range(min(len(all_shown_products), 9))]
            multi_chips.append("Show More")

            SESSION_CACHE[session_id] = {
                "shown_ids": cache_ids,
                "last_params": params,
                "last_query": query_text,
                "segments": segments,
                "shown_products": all_shown_products,
                # ── Back-to-results snapshot ──
                "last_result_message": multi_message,
                "last_result_chips": multi_chips,
                "last_result_header": "🛍️ Multi-search results:",
            }

            return build_response(multi_message, quick_replies=multi_chips)

        # ── SINGLE-SEGMENT SEARCH ──
        brand = get_param(params, "brand")
        if brand and str(brand).lower() not in ("adidas", ""):
            return build_response("❌ Item not found. We only carry Adidas products.")

        matched_products = detect_products_from_text(query_text, df)
        results = search_products(params, query_text)

        if isinstance(results, pd.DataFrame) and results.empty and brand and str(brand).lower() not in ("adidas", ""):
            return build_response("❌ Item not found. We only carry Adidas products.")

        # ── Strip gender words before item-term filtering ──
        # Gender filtering already happened inside search_products().
        all_gender_kws = [kw for kws in GENDER_KEYWORDS.values() for kw in kws]
        gender_strip_pattern = r"\b(?:" + "|".join(re.escape(g) for g in all_gender_kws) + r")\b"
        query_for_item_terms = re.sub(gender_strip_pattern, "", query_text, flags=re.IGNORECASE).strip()

        # ── Item-term refinement ──
        # Use entity synonym map to normalise terms (bags→bag, hoodies→hoodie etc.)
        # so plurals and synonyms match actual dataset values.
        item_terms_raw = extract_query_item_terms(query_for_item_terms, df)
        entity_terms_direct = resolve_entity_synonyms(query_for_item_terms)

        # Build a unified search set: canonical entity terms + raw item terms + their
        # singular/plural variants so nothing slips through
        def _variants(term):
            """Return term + its singular/plural so both forms are searched."""
            variants = {term}
            if term.endswith("s") and len(term) > 3:
                variants.add(term[:-1])           # bags → bag
            else:
                variants.add(term + "s")          # bag → bags
            # Also look up in entity map
            canonical = ENTITY_SYNONYM_MAP.get(term, None)
            if canonical:
                variants.add(canonical)
                if canonical.endswith("s"):
                    variants.add(canonical[:-1])
                else:
                    variants.add(canonical + "s")
            return variants

        all_search_terms = set()
        for t in entity_terms_direct + item_terms_raw:
            all_search_terms.update(_variants(t))

        item_terms = list(all_search_terms)  # keep variable name for fallback check below

        if item_terms and not results.empty:
            # Use strict_entity_filter: name+category first, prevents description false-matches
            refined = results.copy()
            for term in (entity_terms_direct if entity_terms_direct else item_terms_raw):
                candidate = strict_entity_filter(refined, term)
                if not candidate.empty:
                    refined = candidate
            if not refined.empty and len(refined) < len(results):
                results = refined
            # If filter didn't narrow anything useful, keep existing results

        # ── Broad fallback ONLY when results still empty and no terms found ──
        if results.empty and not item_terms and query_for_item_terms:
            text_mask = extract_terms_from_query_text(query_for_item_terms, df)
            if text_mask.any():
                results = df[text_mask].sort_values(
                    ["average_rating", "reviews_count"], ascending=False
                ).reset_index(drop=True)

        # ── Exact product name boosting ──
        exact_matches = pd.DataFrame()
        name_mask = pd.Series(False, index=results.index)

        if matched_products and "name" in results.columns:
            for product_name in matched_products:
                name_mask |= results["name"].str.contains(
                    re.escape(product_name), case=False, na=False
                )
            exact_matches = results[name_mask]

        if matched_products and not exact_matches.empty:
            results["priority"] = results.get(
                "priority", pd.Series(0, index=results.index)
            ).fillna(0)
            results.loc[name_mask, "priority"] = 1
            results = results.sort_values(
                ["priority", "average_rating", "reviews_count"], ascending=False
            )

        # ── Price sanity: if requested price range has no results, say so clearly ──
        if results.empty:
            min_p, max_p = parse_price_from_text(query_text)
            if max_p is not None and max_p < 10:
                return build_response(
                        f"😕 No products found under ${max_p:.0f}. "
                        f"Our lowest price is ${df['selling_price'].min():.0f}. "
                        "Try a higher budget?"
                    )
            if min_p is not None and min_p > df["selling_price"].max():
                return build_response(
                        f"😕 No products above ${min_p:.0f}. "
                        f"Our highest price is ${df['selling_price'].max():.0f}."
                    )
            # ── Did you mean? ──
            # Try running the query through sanitize_query (which applies typo fixes)
            # and see if a corrected version yields results.
            corrected = sanitize_query(query_text)
            if corrected.lower() != query_text.lower():
                corrected_results = search_products(params, corrected)
                if not corrected_results.empty:
                    top_corr = corrected_results.head(PAGE_SIZE)
                    shown_products_corr = [row.to_dict() for _, row in top_corr.iterrows()]
                    has_more_corr = len(corrected_results) > PAGE_SIZE
                    msg_corr, chips_corr = rebuild_results_message(
                        shown_products_corr,
                        header=f"🔍 Did you mean '{corrected}'? Here's what I found:",
                        has_more=has_more_corr,
                    )
                    SESSION_CACHE[session_id] = {
                        "shown_ids": corrected_results.index.tolist()[:PAGE_SIZE],
                        "last_params": params,
                        "last_query": corrected,
                        "shown_products": shown_products_corr,
                        "last_result_message": msg_corr,
                        "last_result_chips": chips_corr,
                        "last_result_header": f"🔍 Did you mean '{corrected}'?",
                    }
                    return build_response(msg_corr, quick_replies=chips_corr)
            return build_response(
                    "😕 No products found matching your request.\n"
                    "Try a different color, category, brand, or price range.\n"
                    "Type 'show all categories' to see what's available.",
                    quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories", "🎲 Surprise Me"]
                )

        top_rows = results.head(PAGE_SIZE)
        shown_products = [row.to_dict() for _, row in top_rows.iterrows()]

        if len(matched_products) > 1:
            header = "🛒 Multiple products found:"
        elif len(matched_products) == 1:
            header = "🎯 Here's what I found:"
        else:
            header = "🔥 Top picks for you:"

        has_more = len(results) > PAGE_SIZE
        message, chips = rebuild_results_message(shown_products, header=header, has_more=has_more)

        SESSION_CACHE[session_id] = {
            "shown_ids": results.index.tolist()[:PAGE_SIZE],
            "last_params": params,
            "last_query": query_text,
            "shown_products": shown_products,
            # ── Back-to-results snapshot ──
            "last_result_message": message,
            "last_result_chips": chips,
            "last_result_header": header,
        }

        return build_response(message, quick_replies=chips)

    # ===== GENDER FILTER INTENT (e.g. "show me women shoes") =====
    if intent_name == INTENT_GENDER_FILTER:
        gender = detect_gender_from_text(query_text)
        if not gender:
            return build_response(
                "Please specify: women, men, or kids.",
                quick_replies=["👩 Women", "👨 Men", "🧒 Kids"]
            )
        gender_results = apply_gender_filter(df.copy(), gender)
        gender_results = gender_results.sort_values(
            ["average_rating", "reviews_count"], ascending=False
        ).reset_index(drop=True)
        if gender_results.empty:
            return build_response(f"😕 No products found for {gender}.")
        top_rows = gender_results.head(PAGE_SIZE)
        shown_products = [row.to_dict() for _, row in top_rows.iterrows()]
        icon = "👩" if gender == "women" else ("👨" if gender == "men" else "🧒")
        gender_header = f"{icon} Top {gender.title()} picks:"
        has_more_gender = len(gender_results) > PAGE_SIZE
        gender_message, gender_chips = rebuild_results_message(
            shown_products, header=gender_header, has_more=has_more_gender
        )
        SESSION_CACHE[session_id] = {
            "shown_ids": gender_results.index.tolist()[:PAGE_SIZE],
            "last_params": params,
            "last_query": query_text,
            "shown_products": shown_products,
            # ── Back-to-results snapshot ──
            "last_result_message": gender_message,
            "last_result_chips": gender_chips,
            "last_result_header": gender_header,
        }
        return build_response(gender_message, quick_replies=gender_chips)

    # ===== FALLBACK =====
    # Last-chance check: if user typed a number (even via Default Fallback),
    # and there are recent shown_products in session, treat it as product selection.
    # This catches cases where Dialogflow routes "1", "#1", "2." to fallback
    # instead of the Select Product intent.
    fallback_num = re.match(r"^(?:view\s+|#?\s*)?([1-9])[.\s]*$", query_text.strip(), re.IGNORECASE)
    if fallback_num:
        cache = SESSION_CACHE.get(session_id, {})
        shown_products = cache.get("shown_products", [])
        if shown_products:
            pick_index = max(0, min(int(fallback_num.group(1)) - 1, len(shown_products) - 1))
            row = shown_products[pick_index]
            card_text, image_url = build_product_detail_card(row)
            cache["last_viewed_product"] = row
            SESSION_CACHE[session_id] = cache
            rich_card = [{
                "title": row.get("name", "Product"),
                "subtitle": f"${float(row.get('selling_price', 0) or 0):.0f}  ⭐ {row.get('average_rating', '')}",
                "imageUri": image_url,
            }] if image_url else None
            return build_response(card_text, quick_replies=["« Back to results", "💖 Save this", "Show More", "📂 Categories"], cards=rich_card)

    return build_response(
            "🤔 I didn't understand that.\n"
            "\n"
            "Try:\n"
            "• women shoes / men hoodies\n"
            "• black shoes under $100\n"
            "• compare Ultraboost vs Runfalcon\n"
            "• Type a number (1, 2, 3) to view a product\n"
            "• Type 'help' for the full guide",
            quick_replies=["👩 Women", "👨 Men", "👟 Shoes", "👕 Clothing", "help"]
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
