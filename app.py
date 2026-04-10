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
INTENT_PRODUCT_SEARCH  = "Product Search"
INTENT_LIST_CATEGORIES = "List Categories"
INTENT_SHOW_MORE       = "Show More"
INTENT_HELP            = "help"
INTENT_GOODBYE         = "goodbye"
INTENT_NEGATIVE        = "Negative Intent"
INTENT_WELCOME         = "Default Welcome Intent"
INTENT_COMPARE         = "Compare Products"
INTENT_PRODUCT_DETAIL  = "Product Detail"
INTENT_AVAILABILITY    = "Check Availability"
INTENT_SELECT_PRODUCT  = "Select Product"
INTENT_GENDER_FILTER   = "Gender Filter"

PAGE_SIZE = 3
SESSION_CACHE = {}

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
ENTITY_SYNONYM_MAP = {
    # ── Footwear ──
    "footwear": "shoes", "shoes": "shoes", "shoe": "shoes",
    "sneakers": "shoes", "sneaker": "shoes",
    "trainers": "shoes", "trainer": "shoes",
    "running shoes": "shoes", "sport shoes": "shoes",
    "kicks": "shoes", "joggers": "shoes",
    "slides": "slides",
    # ── Hoodie ──
    "hoodie": "hoodie", "hoodies": "hoodie",
    "sweatshirt": "hoodie", "sweatshort": "hoodie",
    "pullover": "hoodie", "zip up hoodie": "hoodie", "zip hoodie": "hoodie",
    # ── T-shirt ──
    "t shirt": "t-shirt", "t-shirt": "t-shirt", "tshirt": "t-shirt",
    "tee": "t-shirt", "tees": "t-shirt",
    "shirt": "shirt", "shirts": "shirt", "short sleeve shirt": "shirt",
    "clothes": "clothing",
    # ── Jacket ──
    "jacket": "jacket", "jackets": "jacket",
    "coat": "jacket", "coats": "jacket",
    "windbreaker": "windbreaker", "windbreakers": "windbreaker",
    "outerwear": "jacket", "outerwears": "jacket",
    # ── Pants ──
    "pants": "pants", "trousers": "pants", "joggers pants": "pants",
    "track pants": "pants", "sweatpants": "pants",
    "trouser": "pants", "jogger": "pants",
    "track pant": "pants", "sweatpant": "pants",
    # ── Shorts ──
    "shorts": "shorts", "sport shorts": "shorts", "running shorts": "shorts",
    # ── Socks ──
    "socks": "socks", "sock": "socks",
    "ankle socks": "socks", "sports socks": "socks", "crew socks": "socks",
    # ── Bag ──
    "bag": "bag", "bags": "bag",
    "backpack": "backpack", "backpacks": "backpack",
    "gym bag": "bag", "gym bags": "bag",
    "duffel bag": "duffel", "duffel bags": "duffel",
    "duffle bags": "duffel", "duffle bag": "duffel",
    "sack": "bag", "sacks": "bag",
    "gym sacks": "bag", "gym sack": "bag",
    # ── Cap / Hat ──
    "cap": "cap", "caps": "cap",
    "hat": "hat", "hats": "hat",
    "beanie": "beanie", "beanies": "beanie",
    "headwear": "cap", "headwears": "cap",
    # ── Gloves ──
    "gloves": "gloves", "glove": "gloves",
    # ── Ball ──
    "ball": "ball", "balls": "ball",
    "football": "ball", "soccer ball": "ball", "basketball ball": "ball",
    # ── Accessories ──
    "accessories": "accessories", "accessory": "accessories", "gear": "accessories",
    # ── Activity subcategories ──
    "running": "running", "run": "running", "jog": "running", "jogging": "running",
    "training": "training", "gym": "training", "workout": "training", "exercise": "training",
    "soccer": "soccer", "football boots": "soccer", "cleats": "soccer", "cleat": "soccer",
    "golf": "golf",
    "basketball": "basketball",
    "climbing": "climbing",
    "cycling": "cycling", "bike": "cycling",
    "hiking": "hiking", "hike": "hiking", "trail": "hiking",
    "casual": "casual", "lifestyle": "casual", "everyday": "casual",
}

# Reverse lookup: canonical → list of keywords for subcategory filtering
SUBCATEGORY_MAP = {
    "running":    ["running", "run", "jog", "jogging"],
    "casual":     ["casual", "lifestyle", "originals", "everyday"],
    "training":   ["training", "gym", "workout", "exercise"],
    "soccer":     ["soccer", "football", "cleat", "cleats"],
    "golf":       ["golf"],
    "basketball": ["basketball"],
    "climbing":   ["climbing", "hiangle", "kestrel"],
    "cycling":    ["cycling", "bike", "mountain bike"],
    "slides":     ["slide", "adilette", "sandal"],
    "sandals":    ["sandal", "slide", "adilette"],
    "hiking":     ["hiking", "hike", "trail"],
}

# ===== Preference synonyms =====
PREFERENCE_SYNONYMS = {
    "cheap": [
        "cheap", "cheapest", "affordable", "budget", "inexpensive",
        "low price", "low-price", "low cost", "not expensive",
        "not too expensive", "not that expensive", "budget friendly",
        "budget-friendly", "value",
    ],
    "best": [
        "best", "top rated", "top-rated", "highest rated", "highest-rated",
        "best rated", "best-rated", "good", "quality", "recommended",
        "most reviewed", "popular",
    ],
    "expensive": [
        "expensive", "premium", "luxury", "high end", "high-end",
        "most expensive", "priciest", "top price",
    ],
    "discount": [
        "discount", "deal", "sale", "offer", "most savings",
        "biggest discount", "on sale",
    ],
}

# ===== Gender keywords =====
GENDER_KEYWORDS = {
    "women": ["women", "woman", "female", "ladies", "girl", "girls", "womens", "her"],
    "men":   ["men", "man", "male", "guys", "guy", "mens", "his", "boys"],
    "kids":  ["kids", "kid", "children", "child", "junior", "youth"],
}


# ─────────────────────────────────────────────────────────────
#  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def normalize_plural(term):
    """
    Returns the singular form of a term so 'hoodies'→'hoodie',
    'bags'→'bag', 'shoes'→'shoe', while preserving words that
    naturally end in 's' (e.g. 'socks', 'accessories').
    We use the ENTITY_SYNONYM_MAP first, then a simple s-strip heuristic.
    """
    lower = term.lower().strip()
    # Direct map lookup
    if lower in ENTITY_SYNONYM_MAP:
        return ENTITY_SYNONYM_MAP[lower]
    # Strip trailing 's' if result is ≥3 chars and meaningful
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"           # hoodies → hoodie
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        return lower[:-1]                 # bags → bag, shoes → shoe
    return lower


def term_variants(term):
    """
    Returns a set of search variants for a term:
    original, singular, plural, and entity-map canonical.
    E.g. 'hoodies' → {'hoodies', 'hoodie', 'hoodies', 'hoodie'}
    """
    variants = set()
    lower = term.lower().strip()
    variants.add(lower)

    singular = normalize_plural(lower)
    variants.add(singular)

    # plural form
    if not lower.endswith("s"):
        variants.add(lower + "s")
    if not singular.endswith("s"):
        variants.add(singular + "s")

    # entity-map canonical
    canonical = ENTITY_SYNONYM_MAP.get(lower) or ENTITY_SYNONYM_MAP.get(singular)
    if canonical:
        variants.add(canonical)
        variants.add(normalize_plural(canonical))
        if not canonical.endswith("s"):
            variants.add(canonical + "s")

    return variants


def resolve_entity_synonyms(query_text):
    """
    Scans query_text for any synonym in ENTITY_SYNONYM_MAP and returns
    a list of canonical search terms.  Longest phrase matches take priority.
    """
    if not query_text:
        return []
    text = query_text.lower().strip()
    found = {}
    sorted_synonyms = sorted(ENTITY_SYNONYM_MAP.keys(), key=len, reverse=True)
    for synonym in sorted_synonyms:
        pattern = r"\b" + re.escape(synonym) + r"\b"
        if re.search(pattern, text):
            canonical = ENTITY_SYNONYM_MAP[synonym]
            found[canonical] = True
    return list(found.keys())


def _truncate(text, max_len=300):
    """Truncate text to max_len characters, appending '...' if cut."""
    if not text or text == "nan":
        return "N/A"
    return text[:max_len] + ("..." if len(text) > max_len else "")


def get_param(params, *names):
    """Return the first non-empty parameter value from a list of possible names."""
    for name in names:
        value = params.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


# ─────────────────────────────────────────────────────────────
#  BUILD CSV TERM INDEX
# ─────────────────────────────────────────────────────────────

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
            for i in range(len(words) - 1):
                bigram = words[i] + " " + words[i + 1]
                if words[i] not in noise and words[i + 1] not in noise and len(bigram) >= 6:
                    term_set.add(bigram)
    return term_set


CSV_TERM_INDEX = build_csv_term_index(df)


# ─────────────────────────────────────────────────────────────
#  TERM EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_query_item_terms(query_text, dataframe):
    """
    Scans the user query against the CSV term index and returns every
    item-type term that (a) appears in the query AND (b) matches ≥1 product.
    FIX: normalises plural forms before dedup so 'hoodies' → searches 'hoodie'.
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
            # FIX: also search normalised form so 'hoodies' matches 'hoodie' rows
            search_term = normalize_plural(term)
            if (dataframe[col].str.contains(re.escape(term), case=False, na=False).any() or
                    dataframe[col].str.contains(re.escape(search_term), case=False, na=False).any()):
                # Store the normalised (singular) form to avoid dedup issues
                found_terms.append(search_term)
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
    CSV-backed fallback search using every meaningful token in the query.
    Returns a boolean mask over dataframe rows.
    """
    if not query_text:
        return pd.Series(False, index=dataframe.index)
    text = query_text.lower()
    stop = {
        "and","or","the","a","an","for","me","please","show","find","get","want",
        "need","some","any","under","below","above","around","with","in","on","at",
        "of","to","my","i","like","give","both","also","can","you","do","have",
        "what","which","is","are","just",
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
    Detects SPECIFIC named products by matching meaningful keywords from
    each unique product name against the query. Requires ALL non-generic words to match.
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


# ─────────────────────────────────────────────────────────────
#  STRICT ENTITY FILTER
# ─────────────────────────────────────────────────────────────

def strict_entity_filter(results, term):
    """
    Filters results for a product term using NAME + CATEGORY first (strict),
    falling back to description/breadcrumbs only if strict search finds nothing.
    Prevents false matches like "bag" matching Tees that mention bags in description.
    Uses term_variants() so plural/singular both hit correctly.
    """
    variants = term_variants(term)

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


# ─────────────────────────────────────────────────────────────
#  CATEGORY / GENDER / PREFERENCE DETECTION
# ─────────────────────────────────────────────────────────────

def detect_preference_from_text(query_text):
    """
    Extracts a canonical preference label from messy natural language.
    Conflict resolution: best+cheap→best, cheap+expensive→cheap.
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
    if "best" in found and "cheap" in found:
        return "best"
    if "expensive" in found and "cheap" in found:
        return "cheap"
    if "best" in found and "expensive" in found:
        return "best"
    for pref in ["best", "cheap", "expensive", "discount"]:
        if pref in found:
            return pref
    return None


def detect_subcategory_from_text(query_text):
    """Returns subcategory filter keywords detected in query."""
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


def detect_category_from_text(query_text):
    """
    Returns most likely top-level category (Shoes/Clothing/Accessories) or None.
    Uses resolve_entity_synonyms for full synonym coverage.
    """
    if not query_text:
        return None
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
    for canonical in resolve_entity_synonyms(query_text):
        if canonical in shoe_canonicals:
            counts["Shoes"] += 1
        elif canonical in clothing_canonicals:
            counts["Clothing"] += 1
        elif canonical in accessory_canonicals:
            counts["Accessories"] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def detect_gender_from_text(query_text):
    """Returns 'women', 'men', 'kids', or None."""
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
    Filters DataFrame by gender using breadcrumbs first, then name, then description.
    Returns empty DataFrame (never falls back) if nothing matches — prevents wrong products.
    """
    if not gender or results.empty:
        return results
    gender_mask = pd.Series(False, index=results.index)
    if "breadcrumbs" in results.columns:
        gender_mask |= results["breadcrumbs"].str.contains(gender, case=False, na=False)
    if "name" in results.columns:
        gender_mask |= results["name"].str.contains(
            r"\b" + re.escape(gender) + r"\b", case=False, na=False, regex=True
        )
    if not gender_mask.any() and "description" in results.columns:
        gender_mask |= results["description"].str.contains(gender, case=False, na=False)
    return results[gender_mask]


def infer_gender_from_row(row):
    """Returns 'Women', 'Men', 'Kids', or 'Unisex' for a product row dict."""
    text = " ".join([
        str(row.get("breadcrumbs", "")),
        str(row.get("name", "")),
        str(row.get("category", "")),
        str(row.get("description", "")),
    ]).lower()
    if re.search(r"\bwomen|\bwomens|\bladies|\bfemale|\bgirl", text):
        return "Women"
    if re.search(r"\bmen\b|\bmens\b|\bmale|\bhis\b|\bguy|\bboy\b", text):
        return "Men"
    if re.search(r"\bkids|\bjunior|\byouth|\bchild", text):
        return "Kids"
    return "Unisex"


# ─────────────────────────────────────────────────────────────
#  PRICE PARSING
# ─────────────────────────────────────────────────────────────

def parse_price_range(value):
    """Parses a price range from a Dialogflow parameter value."""
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


def parse_price_from_text(text):
    """
    Extract (min_price, max_price) directly from raw query text.
    Handles: 'under 100', 'below 50', 'above 80', 'between 50 and 150',
             '50 to 150', 'around 80' (±20%), 'less than 120'.
    """
    if not text:
        return None, None
    t = text.lower().replace(",", "")

    m = re.search(r"between\s+\$?(\d+(?:\.\d+)?)\s+and\s+\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))

    m = re.search(r"\$?(\d+(?:\.\d+)?)\s+to\s+\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))

    m = re.search(
        r"(?:under|below|less\s+than|up\s+to|max(?:imum)?|no\s+more\s+than|<)\s*\$?(\d+(?:\.\d+)?)", t
    )
    if m:
        return None, float(m.group(1))

    m = re.search(r"\$?(\d+(?:\.\d+)?)\s+or\s+(?:less|below)", t)
    if m:
        return None, float(m.group(1))

    m = re.search(
        r"(?:above|over|more\s+than|at\s+least|min(?:imum)?|>)\s*\$?(\d+(?:\.\d+)?)", t
    )
    if m:
        return float(m.group(1)), None

    m = re.search(r"(?:around|approximately|about|~)\s*\$?(\d+(?:\.\d+)?)", t)
    if m:
        mid = float(m.group(1))
        return mid * 0.8, mid * 1.2

    return None, None


# ─────────────────────────────────────────────────────────────
#  QUERY SANITIZER
# ─────────────────────────────────────────────────────────────

def sanitize_query(query_text):
    """
    Cleans malformed queries:
    - 'black black shoes' → 'black shoes'
    - 'shoes under under 100' → 'shoes under 100'
    - 'more more more more' → 'show more'
    - '1', '#1', 'View 1' → normalised number string
    """
    if not query_text:
        return query_text

    text = query_text.strip()

    # Normalise product selection: "View 1", "#1", "select 1" etc.
    num_match = re.match(
        r"^(?:view\s+|#\s*|select\s+|option\s+|number\s+|item\s+|pick\s+)?([1-9])[\.\s]*$",
        text, re.IGNORECASE
    )
    if num_match:
        return num_match.group(1)

    text = text.lower().strip()

    # 'more more more' → 'show more'
    if re.match(r"^(?:more\s+)+$", text):
        return "show more"

    # Remove consecutive duplicate words: "black black" → "black"
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text)

    # Remove consecutive duplicate price keywords: "under under" → "under"
    text = re.sub(r"\b(under|below|above|over|between)\s+\1\b", r"\1", text)

    return text


# ─────────────────────────────────────────────────────────────
#  PRODUCT IMAGE
# ─────────────────────────────────────────────────────────────

def get_product_image(row):
    """Extracts the first product image URL from tilde-separated images column."""
    raw = row.get("images", "") if isinstance(row, dict) else str(row)
    if not raw or raw in ("nan", "None", ""):
        return None
    first = raw.split("~")[0].strip()
    return first if first.startswith("http") else None


# ─────────────────────────────────────────────────────────────
#  PRODUCT DETAIL CARD
# ─────────────────────────────────────────────────────────────

def build_product_detail_card(row):
    """Builds a rich product detail card. Returns (text, image_url)."""
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

    in_stock = "in stock" in avail.lower() or avail.lower() in ("true", "1", "yes", "available")
    stock_badge = "✅ In Stock" if in_stock else "❌ Out of Stock"

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


def get_product_detail(query_text):
    """Returns (card_text, image_url) for a named product, or None."""
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
    return build_product_detail_card(row.to_dict())


# ─────────────────────────────────────────────────────────────
#  SUGGEST SIMILAR
# ─────────────────────────────────────────────────────────────

def suggest_similar(product_name, exclude_names=None, top_n=3):
    """Suggests similar products from the same category."""
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


# ─────────────────────────────────────────────────────────────
#  COMPARISON
# ─────────────────────────────────────────────────────────────

def is_comparison_query(query_text):
    """Returns True if the query looks like a product comparison request."""
    text = query_text.lower()
    for t in [r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b",
              r"\bdifference between\b", r"\bcompare between\b"]:
        if re.search(t, text):
            return True
    return False


def fuzzy_product_find(term, min_word_match=1):
    """
    Finds the best matching product for a term using word-level fuzzy matching.
    Returns (row, matched_name) or (None, None).
    Tries: exact → partial → word-by-word overlap.
    """
    # 1. Exact substring match in name or description
    mask = pd.Series(False, index=df.index)
    for col in ["name", "description"]:
        if col in df.columns:
            mask |= df[col].str.contains(re.escape(term), case=False, na=False)
    matched = df[mask]
    if not matched.empty:
        best = matched.sort_values(["average_rating", "reviews_count"], ascending=False).iloc[0]
        return best, best.get("name", term)

    # 2. Word-level overlap: split term into words and find products matching most words
    words = [w for w in re.findall(r"[a-z0-9]+", term.lower()) if len(w) >= 3]
    if not words:
        return None, None

    best_row = None
    best_score = 0
    best_name = None
    for _, row in df.iterrows():
        name_lower = str(row.get("name", "")).lower()
        desc_lower = str(row.get("description", "")).lower()
        score = sum(1 for w in words if w in name_lower or w in desc_lower)
        if score > best_score and score >= min_word_match:
            best_score = score
            best_row = row
            best_name = row.get("name", term)

    return (best_row, best_name) if best_row is not None else (None, None)


def compare_products(query_text):
    """
    Side-by-side comparison of two products.
    Supports 'compare X and Y', 'X vs Y', 'difference between X and Y'.
    Uses fuzzy_product_find so partial names still work.
    """
    text = query_text.lower()
    clean = re.sub(
        r"^(?:compare|comparison(?:\s+between)?|difference\s+between|compare\s+between)\s+",
        "", text
    ).strip()

    vs_pattern  = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", clean)
    and_pattern = re.search(r"(.+?)\s+(?:and|with)\s+(.+)", clean)

    term_a, term_b = None, None
    if vs_pattern:
        term_a, term_b = vs_pattern.group(1).strip(), vs_pattern.group(2).strip()
    elif and_pattern:
        term_a, term_b = and_pattern.group(1).strip(), and_pattern.group(2).strip()

    if not term_a or not term_b:
        return None

    product_a, name_a = fuzzy_product_find(term_a)
    product_b, name_b = fuzzy_product_find(term_b)

    if product_a is None and product_b is None:
        return (f"❌ Could not find products matching '{term_a}' or '{term_b}' in our catalog.\n"
                "Try using a product name visible in search results.")
    if product_a is None:
        return (f"❌ Could not find '{term_a}' in our catalog.\n"
                "Try searching for it first to see the exact name.")
    if product_b is None:
        return (f"❌ Could not find '{term_b}' in our catalog.\n"
                "Try searching for it first to see the exact name.")

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
        except Exception:
            pass
        return ""

    def product_card(p, label):
        row_dict = p.to_dict() if hasattr(p, "to_dict") else p
        name    = row_dict.get("name", "Unknown")
        cat     = _s(row_dict.get("category"))
        color   = _s(row_dict.get("color"))
        price   = _s(row_dict.get("selling_price"), "$", "", 0)
        orig    = _s(row_dict.get("original_price"), "$", "", 0)
        rating  = _s(row_dict.get("average_rating"), "", "/5", 1)
        reviews = _s(row_dict.get("reviews_count"))
        avail   = _stock(row_dict.get("availability", ""))
        disc    = _discount(row_dict.get("original_price"), row_dict.get("selling_price"))
        stars   = _stars(row_dict.get("average_rating"))
        gender  = infer_gender_from_row(row_dict)
        desc    = _truncate(str(row_dict.get("description", "N/A")), 220)
        price_line = f"💰 {price}"
        if disc:
            price_line += f"  {disc}  (was {orig})"
        return (
            f"{label}\n"
            f"👟 {name}\n\n"
            f"{avail}\n"
            f"🏷️ {cat}   🎨 {color}   👤 {gender}\n"
            f"{price_line}\n"
            f"{stars} {rating}  💬 {reviews} reviews\n\n"
            f"📝 {desc}"
        )

    divider = "═" * 30
    winner_note = ""
    try:
        a_dict = product_a.to_dict() if hasattr(product_a, "to_dict") else product_a
        b_dict = product_b.to_dict() if hasattr(product_b, "to_dict") else product_b
        r_a = float(a_dict.get("average_rating") or 0)
        r_b = float(b_dict.get("average_rating") or 0)
        p_a = float(a_dict.get("selling_price") or 0)
        p_b = float(b_dict.get("selling_price") or 0)
        if r_a > r_b:
            winner_note += f"\n🏆 Better rated: {name_a}"
        elif r_b > r_a:
            winner_note += f"\n🏆 Better rated: {name_b}"
        if p_a and p_b:
            if p_a < p_b:
                winner_note += f"\n💸 Better value: {name_a}"
            elif p_b < p_a:
                winner_note += f"\n💸 Better value: {name_b}"
    except Exception:
        pass

    return (
        f"📊 Comparison\n"
        f"{divider}\n"
        f"{product_card(product_a, '🅰️ Product A')}\n"
        f"{divider}\n"
        f"{product_card(product_b, '🅱️ Product B')}\n"
        f"{divider}"
        f"{winner_note}"
    )


# ─────────────────────────────────────────────────────────────
#  FORMAT PRODUCT (list view)
# ─────────────────────────────────────────────────────────────

def format_product(row, index=None):
    row_dict = row.to_dict() if hasattr(row, "to_dict") else row
    name     = row_dict.get("name", "Unknown")
    price    = row_dict.get("selling_price", None)
    rating   = row_dict.get("average_rating", None)
    reviews  = row_dict.get("reviews_count", None)
    category = row_dict.get("category", "")
    color    = row_dict.get("color", "")
    gender   = infer_gender_from_row(row_dict)

    gender_text  = f"👤 {gender}" if gender and gender != "Unisex" else ""
    price_text   = f"${price:.0f}" if pd.notna(price) else "N/A"
    rating_text  = f"⭐ {rating:.1f}" if pd.notna(rating) else ""
    reviews_text = f"({int(reviews)} reviews)" if pd.notna(reviews) and reviews else ""
    color_text   = f"🎨 {color}" if color and color != "nan" else ""
    category_text= f"🏷️ {category}" if category and category != "nan" else ""

    prefix = f"{index}. " if index is not None else ""
    parts = [f"{prefix}👟 {name}"]
    if category_text:
        parts.append(f"   {category_text}")
    if color_text:
        parts.append(f"   {color_text}")
    if gender_text:
        parts.append(f"   {gender_text}")
    parts.append(f"   💰 {price_text}  {rating_text} {reviews_text}".rstrip())
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
#  MULTI-SEGMENT QUERY PARSER
# ─────────────────────────────────────────────────────────────

def _extract_global_price(text):
    """Extract a single global max-price constraint from full query text."""
    m = re.search(
        r"(?:under|below|less\s+than|up\s+to|max(?:imum)?|<)\s*\$?(\d+(?:\.\d+)?)"
        r"|\$?(\d+(?:\.\d+)?)\s*(?:or\s+)?(?:less|below|max)",
        text
    )
    if m:
        nums = [float(x) for x in m.groups() if x is not None]
        return max(nums) if nums else None
    return None


def parse_multi_segment_query(query_text):
    """
    Detects queries for MULTIPLE product types, e.g.:
      "pink hoodies and burgundy duffel bags under 500"
    Returns a list of segment dicts or [] if single-product query.
    FIX: normalises product terms so 'hoodies' → 'hoodie' before passing to search_segment.
    """
    if not query_text:
        return []
    text = query_text.lower()

    if is_comparison_query(text):
        return []

    raw_segments = re.split(r"\band\b", text)
    if len(raw_segments) < 2:
        return []

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

        # Color
        for color_word in colors_in_dataset:
            if re.search(r"\b" + re.escape(color_word) + r"\b", seg):
                color = color_word
                break

        # Product: entity synonym map first (most reliable), then CSV index
        entity_terms = resolve_entity_synonyms(seg)
        if entity_terms:
            # FIX: use canonical form directly — already normalised by ENTITY_SYNONYM_MAP
            product = entity_terms[0]
        else:
            item_terms = extract_query_item_terms(seg, df)
            if item_terms:
                product = item_terms[0]  # already normalised to singular by extract_query_item_terms

        if product:
            segments.append({
                "color": color,
                "product": product,
                "max_price": max_price,
                "raw": seg,
            })

    return segments if len(segments) >= 2 else []


# ─────────────────────────────────────────────────────────────
#  SEARCH SEGMENT (single color+product+price slice)
# ─────────────────────────────────────────────────────────────

def search_segment(color=None, product=None, max_price=None, preference=None, gender=None):
    """
    Focused search for a single segment.
    FIX: uses term_variants() so 'hoodie' also matches 'Hoodie', 'Hoodies' etc.
    """
    results = df.copy()

    if color and "color" in results.columns:
        results = results[results["color"].str.contains(re.escape(color), case=False, na=False)]

    if product:
        product_mask = pd.Series(False, index=results.index)
        for variant in term_variants(product):
            for col in ["name", "category", "description"]:
                if col in results.columns:
                    product_mask |= results[col].str.contains(
                        re.escape(variant), case=False, na=False
                    )
        results = results[product_mask]

    if max_price is not None and "selling_price" in results.columns:
        results = results[results["selling_price"].notna()]
        results = results[results["selling_price"] <= max_price]

    if gender:
        gender_filtered = apply_gender_filter(results, gender)
        if not gender_filtered.empty:
            results = gender_filtered

    if "original_price" in results.columns and "selling_price" in results.columns:
        results["discount"] = results["original_price"] - results["selling_price"]
    else:
        results["discount"] = 0

    if preference:
        pref = str(preference).lower()
        if "cheap" in pref:
            results = results.sort_values("selling_price")
        elif "expensive" in pref:
            results = results.sort_values("selling_price", ascending=False)
        elif "best" in pref or "rating" in pref:
            results = results.sort_values(["average_rating", "reviews_count"], ascending=False)
        elif "discount" in pref:
            results = results.sort_values("discount", ascending=False)
        else:
            results = results.sort_values(["average_rating", "reviews_count"], ascending=False)
    else:
        results = results.sort_values(["average_rating", "reviews_count"], ascending=False)

    return results.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  CORE SEARCH
# ─────────────────────────────────────────────────────────────

def search_products(params, query_text=""):
    """
    Core search function with full filter chain:
    brand → color → product/entity → subcategory → price → sort → gender.
    """
    results = df.copy()

    brand      = get_param(params, "brand")
    color      = get_param(params, "color")
    products   = get_param(params, "products")
    usage      = get_param(params, "usage")
    preference = get_param(params, "preference")
    max_price  = get_param(params, "max_price")
    price_range= get_param(params, "price_range")

    # ── Brand ──
    if brand and str(brand).lower() not in ("adidas", ""):
        return pd.DataFrame()

    # ── Color ──
    if color and "color" in results.columns:
        results = results[results["color"].str.contains(str(color), case=False, na=False)]
    elif not color and query_text:
        colors_in_dataset = set()
        for c in df["color"].dropna().unique():
            for w in re.findall(r"[a-z]+", c.lower()):
                if len(w) >= 3:
                    colors_in_dataset.add(w)
        for color_word in sorted(colors_in_dataset, key=len, reverse=True):
            if re.search(r"\b" + re.escape(color_word) + r"\b", query_text.lower()):
                results = results[
                    results["color"].str.contains(re.escape(color_word), case=False, na=False)
                ]
                break

    # ── Product / Category ──
    if products:
        products_resolved = ENTITY_SYNONYM_MAP.get(str(products).lower(), str(products))
        product_mask = pd.Series(False, index=results.index)
        for variant in term_variants(products_resolved):
            for col in ["name", "category", "description"]:
                if col in results.columns:
                    product_mask |= results[col].str.contains(
                        re.escape(variant), case=False, na=False
                    )
        results = results[product_mask]
    elif not products and query_text:
        entity_terms = resolve_entity_synonyms(query_text)
        if entity_terms:
            entity_filtered = results.copy()
            for term in entity_terms:
                candidate = strict_entity_filter(entity_filtered, term)
                if not candidate.empty:
                    entity_filtered = candidate
            if not entity_filtered.empty:
                results = entity_filtered
            else:
                detected_cat = detect_category_from_text(query_text)
                if detected_cat:
                    results = results[results["category"].str.contains(
                        detected_cat, case=False, na=False
                    )]
        else:
            detected_cat = detect_category_from_text(query_text)
            if detected_cat:
                results = results[results["category"].str.contains(
                    detected_cat, case=False, na=False
                )]

    # ── Subcategory / Usage ──
    if usage:
        usage_resolved = ENTITY_SYNONYM_MAP.get(str(usage).lower(), str(usage))
        usage_mask = pd.Series(False, index=results.index)
        for col in ["category", "description", "name", "breadcrumbs"]:
            if col in results.columns:
                usage_mask |= results[col].str.contains(
                    re.escape(usage_resolved), case=False, na=False
                )
        if usage_mask.any():
            results = results[usage_mask]
    elif not usage and query_text:
        subcat_keywords = detect_subcategory_from_text(query_text)
        if subcat_keywords:
            subcat_mask = pd.Series(False, index=results.index)
            for kw in subcat_keywords:
                for col in ["name", "description", "breadcrumbs"]:
                    if col in results.columns:
                        subcat_mask |= results[col].str.contains(
                            re.escape(kw), case=False, na=False
                        )
            if subcat_mask.any():
                results = results[subcat_mask]

    # ── Price ──
    min_price, max_price_range = parse_price_range(price_range)
    if max_price:
        try:
            max_price_range = float(max_price)
        except Exception:
            pass
    if min_price is None and max_price_range is None and query_text:
        min_price, max_price_range = parse_price_from_text(query_text)

    if "selling_price" in results.columns:
        results = results[results["selling_price"].notna()]
        if min_price is not None:
            results = results[results["selling_price"] >= min_price]
        if max_price_range is not None:
            results = results[results["selling_price"] <= max_price_range]

    # ── Discount ──
    if "original_price" in results.columns and "selling_price" in results.columns:
        results["discount"] = results["original_price"] - results["selling_price"]
    else:
        results["discount"] = 0

    # ── Sort ──
    effective_pref = preference or detect_preference_from_text(query_text)
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

    # ── Gender ──
    gender = detect_gender_from_text(query_text)
    if gender:
        gender_filtered = apply_gender_filter(results, gender)
        if not gender_filtered.empty:
            results = gender_filtered

    return results.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  RESPONSE BUILDER
# ─────────────────────────────────────────────────────────────

def build_response(text, quick_replies=None, cards=None):
    """
    Builds Dialogflow Messenger-compatible fulfillmentMessages response.
    cards: list of {title, subtitle, imageUri, buttons}
    quick_replies: list of chip label strings
    """
    messages = []

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

    lines = [line for line in text.split("\n") if line.strip()]
    messages += [{"text": {"text": [line]}} for line in lines]

    if quick_replies:
        messages.append({
            "quickReplies": {
                "title": "Quick options:",
                "quickReplies": quick_replies,
            }
        })

    return jsonify({
        "fulfillmentMessages": messages,
        "fulfillmentText": text,
    })


# ─────────────────────────────────────────────────────────────
#  PATTERNS
# ─────────────────────────────────────────────────────────────

GREETING_PATTERNS = re.compile(
    r"^\s*(?:hi|hello|hey|howdy|what'?s\s+up|sup|good\s+(?:morning|afternoon|evening|day))[!.,?]*\s*$",
    re.IGNORECASE,
)
DETAIL_PATTERNS = re.compile(
    r"\b(?:tell\s+me\s+about|what\s+is|what'?s|describe|details?\s+(?:of|about|on)|info(?:rmation)?\s+(?:on|about))\b",
    re.IGNORECASE,
)
NONSENSE_PATTERNS = re.compile(
    r"\b(?:invisible|transparent|rainbow|imaginary|impossible|magic|unicorn|fake|nonexistent)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
#  WEBHOOK
# ─────────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True) or {}
    query_result = req.get("queryResult", {})
    intent_name  = query_result.get("intent", {}).get("displayName", "")
    params       = query_result.get("parameters", {})
    session_id   = req.get("session", "default-session")
    query_text   = query_result.get("queryText", "")

    query_text = sanitize_query(query_text)

    # ── GREETING ──
    if GREETING_PATTERNS.match(query_text) or intent_name == INTENT_WELCOME:
        return build_response(
            "👋 Hi! Welcome to the Adidas USA Store.\n"
            "\n"
            "🛍️ What I can do:\n"
            "\n"
            "1️⃣  Browse by gender — women shoes / men hoodies / kids clothing\n"
            "2️⃣  Search by category & color — black running shoes under $100\n"
            "3️⃣  Sort by preference — cheapest shoes / best rated hoodies\n"
            "4️⃣  Compare products — compare Ultraboost and EQ21 Run\n"
            "5️⃣  View product detail — reply with a number (1, 2, 3) after results\n"
            "\n"
            "💡 Type 'help' anytime.",
            quick_replies=["👩 Women", "👨 Men", "👟 Shoes", "👕 Clothing", "🎒 Accessories", "📂 Categories"]
        )

    # ── HELP ──
    if intent_name == INTENT_HELP:
        return build_response(
            "🆘 Here's what I can do:\n"
            "\n"
            "🔍 Search — shoes under 100\n"
            "🎨 Color filter — black running shoes\n"
            "⭐ Sort — cheap clothing / best accessories\n"
            "💰 Price range — shoes between 50 and 150\n"
            "📊 Compare — compare Ultraboost and Supernova\n"
            "📂 Categories — show all categories\n"
            "➡️ More — show more"
        )

    # ── GOODBYE ──
    if intent_name == INTENT_GOODBYE:
        return build_response("👋 Bye! Come back anytime to find more products.")

    # ── NEGATIVE ──
    if intent_name == INTENT_NEGATIVE:
        return build_response(
            "Okay — try another color, category, or budget.",
            quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories", "help"]
        )

    # ── LIST CATEGORIES ──
    if intent_name == INTENT_LIST_CATEGORIES:
        if "category" not in df.columns:
            return build_response("I cannot find categories in the dataset.")
        categories = (
            df["category"].dropna().astype(str).str.strip()
            .replace("", pd.NA).dropna().str.lower()
            .drop_duplicates().head(15).tolist()
        )
        if not categories:
            return build_response("No categories found in the dataset.")
        return build_response(
            "📂 Here are the categories:\n- " + "\n- ".join(categories),
            quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"]
        )

    # ── COMPARE ──
    if intent_name == INTENT_COMPARE or is_comparison_query(query_text):
        result = compare_products(query_text)
        if result:
            return build_response(result, quick_replies=["Show More", "📂 Categories", "help"])
        return build_response(
            "❌ I couldn't identify two products to compare.\n"
            "Try: 'compare Ultraboost and Supernova' or 'EQ21 Run vs ZX 1K Boost'."
        )

    # ── PRODUCT DETAIL ──
    if intent_name == INTENT_PRODUCT_DETAIL or DETAIL_PATTERNS.search(query_text):
        detail_result = get_product_detail(query_text)
        if detail_result:
            detail_text, detail_img = detail_result
            rich_card = [{"title": query_text.title(), "imageUri": detail_img}] if detail_img else None
            return build_response(detail_text, cards=rich_card,
                                  quick_replies=["Show More", "📂 Categories"])
        # Fall through to product search

    # ── SHOW MORE ──
    if (intent_name == INTENT_SHOW_MORE
            or re.search(r"\bshow\s+more\b|\bmore\s+results?\b|\bnext\b|\bmore\b", query_text.lower())):
        cache = SESSION_CACHE.get(session_id)
        if not cache:
            return build_response(
                "Please search for a product first before asking for more.",
                quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"]
            )

        if not any(v for v in params.values() if v not in (None, "", [], {})):
            params           = cache.get("last_params", {})
            query_text_more  = cache.get("last_query", "")
        else:
            query_text_more = query_text

        # ── Multi-segment show more ──
        cached_segments = cache.get("segments")
        if cached_segments:
            all_lines = []
            shown_ids = cache.get("shown_ids", [])
            all_shown = []

            for seg in cached_segments:
                seg_results = search_segment(
                    color=seg.get("color"),
                    product=seg.get("product"),
                    max_price=seg.get("max_price"),
                )
                seg_results = seg_results[~seg_results.index.isin(shown_ids)]
                color_label   = seg.get("color", "").title() if seg.get("color") else ""
                product_label = seg.get("product", "").title()
                header = f"🛍️ More {color_label} {product_label}".strip() + ":"

                if seg_results.empty:
                    all_lines.append(f"{header}\n✅ No more products.")
                else:
                    top = seg_results.head(PAGE_SIZE)
                    shown_ids.extend(top.index.tolist())
                    all_shown.extend([row.to_dict() for _, row in top.iterrows()])
                    separator = "─" * 28
                    seg_lines = [format_product(row, index=i + 1)
                                 for i, (_, row) in enumerate(top.iterrows())]
                    all_lines.append(f"{header}\n\n" + f"\n{separator}\n".join(seg_lines))

            cache["shown_ids"] = shown_ids
            if all_shown:
                cache["shown_products"] = cache.get("shown_products", []) + all_shown

            if all(("No more" in l or "not found" in l.lower()) for l in all_lines):
                return build_response("✅ No more products found.")

            chips = [f"View {i+1}" for i in range(len(all_shown))]
            chips.append("Show More")
            return build_response("\n\n".join(all_lines), quick_replies=chips)

        # ── Single-segment show more ──
        results = search_products(params, query_text_more)

        # Apply same item-term filter as original search
        entity_terms = resolve_entity_synonyms(query_text_more)
        if entity_terms:
            for term in entity_terms:
                candidate = strict_entity_filter(results, term)
                if not candidate.empty:
                    results = candidate
                    break

        shown_ids  = cache.get("shown_ids", [])
        results    = results[~results.index.isin(shown_ids)]
        next_chunk = results.head(PAGE_SIZE)

        if next_chunk.empty:
            return build_response(
                "✅ No more products found. Try a new search!",
                quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"]
            )

        cache["shown_ids"].extend(next_chunk.index.tolist())
        new_products = [row.to_dict() for _, row in next_chunk.iterrows()]
        cache["shown_products"] = cache.get("shown_products", []) + new_products

        separator = "─" * 28
        lines = [format_product(row, index=i + 1)
                 for i, (_, row) in enumerate(next_chunk.iterrows())]
        body    = f"\n{separator}\n".join(lines)
        message = f"📦 More results:\n\n{body}"
        chips   = [f"View {i+1}" for i in range(len(next_chunk))]
        chips.append("Show More")
        return build_response(message, quick_replies=chips)

    # ── SELECT PRODUCT BY NUMBER ──
    number_match = re.match(r"^(?:view\s+)?([1-9])$", query_text.strip(), re.IGNORECASE)
    if number_match or intent_name == INTENT_SELECT_PRODUCT:
        cache = SESSION_CACHE.get(session_id, {})
        shown_products = cache.get("shown_products", [])
        if not shown_products:
            return build_response(
                "🔍 No recent results to select from.\nPlease search for a product first.",
                quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"]
            )
        if number_match:
            pick_index = int(number_match.group(1)) - 1
        else:
            try:
                raw_num = str(get_param(params, "number", "ordinal") or "1")
                pick_index = int(re.search(r"\d+", raw_num).group()) - 1
            except Exception:
                pick_index = 0
        pick_index = max(0, min(pick_index, len(shown_products) - 1))
        row = shown_products[pick_index]
        card_text, image_url = build_product_detail_card(row)
        rich_card = [{
            "title": row.get("name", "Product"),
            "subtitle": f"${float(row.get('selling_price', 0) or 0):.0f}  ⭐ {row.get('average_rating', '')}",
            "imageUri": image_url,
        }] if image_url else None
        return build_response(card_text, cards=rich_card,
                              quick_replies=["Show More", "📂 Categories"])

    # ── PRODUCT SEARCH ──
    if intent_name == INTENT_PRODUCT_SEARCH or intent_name not in (
        INTENT_SHOW_MORE, INTENT_LIST_CATEGORIES, INTENT_HELP,
        INTENT_GOODBYE, INTENT_NEGATIVE, INTENT_WELCOME,
        INTENT_COMPARE, INTENT_PRODUCT_DETAIL, INTENT_AVAILABILITY,
        INTENT_SELECT_PRODUCT, INTENT_GENDER_FILTER,
    ):
        # Nonsense guard
        if NONSENSE_PATTERNS.search(query_text):
            return build_response(
                "😅 We don't carry that!\n"
                "Try a real color (black, white, red...) or product type.",
                quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories", "📂 Categories"]
            )

        # ── Multi-segment ──
        segments = parse_multi_segment_query(query_text)
        if segments:
            all_lines      = []
            cache_ids      = []
            all_shown_prods= []
            gender = detect_gender_from_text(query_text)

            for seg in segments:
                seg_results = search_segment(
                    color=seg.get("color"),
                    product=seg.get("product"),
                    max_price=seg.get("max_price"),
                    gender=gender,
                )
                color_label   = seg.get("color", "").title() if seg.get("color") else ""
                product_label = seg.get("product", "").title()
                gender_label  = f" ({gender.title()})" if gender else ""
                header = f"🛍️ {color_label} {product_label}{gender_label}".strip() + ":"

                if seg_results.empty:
                    all_lines.append(f"{header}\n❌ No products found.")
                else:
                    top  = seg_results.head(PAGE_SIZE)
                    cache_ids.extend(top.index.tolist())
                    prods = [row.to_dict() for _, row in top.iterrows()]
                    all_shown_prods.extend(prods)
                    separator = "─" * 28
                    start     = len(all_shown_prods) - len(prods)
                    seg_lines = [format_product(row, index=start + i + 1)
                                 for i, (_, row) in enumerate(top.iterrows())]
                    entry = f"{header}\n\n" + f"\n{separator}\n".join(seg_lines)
                    if len(seg_results) > PAGE_SIZE:
                        entry += f"\n{separator}\n💬 Say 'show more' for more."
                    all_lines.append(entry)

            SESSION_CACHE[session_id] = {
                "shown_ids": cache_ids,
                "last_params": params,
                "last_query": query_text,
                "segments": segments,
                "shown_products": all_shown_prods,
            }
            chips = [f"View {i+1}" for i in range(min(len(all_shown_prods), 9))]
            chips.append("Show More")
            return build_response("\n\n".join(all_lines), quick_replies=chips)

        # ── Single-segment ──
        brand = get_param(params, "brand")
        if brand and str(brand).lower() not in ("adidas", ""):
            return build_response("❌ We only carry Adidas products.")

        matched_products = detect_products_from_text(query_text, df)
        results = search_products(params, query_text)

        # Strip gender words then run item-term filter
        all_gender_kws = [kw for kws in GENDER_KEYWORDS.values() for kw in kws]
        gender_strip   = r"\b(?:" + "|".join(re.escape(g) for g in all_gender_kws) + r")\b"
        query_for_terms= re.sub(gender_strip, "", query_text, flags=re.IGNORECASE).strip()

        entity_terms  = resolve_entity_synonyms(query_for_terms)
        item_terms_raw= extract_query_item_terms(query_for_terms, df)

        if entity_terms or item_terms_raw:
            refined = results.copy()
            for term in (entity_terms if entity_terms else item_terms_raw):
                candidate = strict_entity_filter(refined, term)
                if not candidate.empty:
                    refined = candidate
            if not refined.empty:
                results = refined

        # Broad fallback — only when nothing found AND no item terms
        if results.empty and not entity_terms and not item_terms_raw and query_for_terms:
            text_mask = extract_terms_from_query_text(query_for_terms, df)
            if text_mask.any():
                results = df[text_mask].sort_values(
                    ["average_rating", "reviews_count"], ascending=False
                ).reset_index(drop=True)

        # Exact product name boosting
        name_mask = pd.Series(False, index=results.index)
        if matched_products and "name" in results.columns:
            for product_name in matched_products:
                name_mask |= results["name"].str.contains(
                    re.escape(product_name), case=False, na=False
                )
            if name_mask.any():
                results["priority"] = results.get(
                    "priority", pd.Series(0, index=results.index)
                ).fillna(0)
                results.loc[name_mask, "priority"] = 1
                results = results.sort_values(
                    ["priority", "average_rating", "reviews_count"], ascending=False
                )

        # Price sanity messages
        if results.empty:
            min_p, max_p = parse_price_from_text(query_text)
            if max_p is not None and max_p < df["selling_price"].min():
                return build_response(
                    f"😕 No products under ${max_p:.0f}.\n"
                    f"Our lowest price is ${df['selling_price'].min():.0f}. Try a higher budget?"
                )
            if min_p is not None and min_p > df["selling_price"].max():
                return build_response(
                    f"😕 No products above ${min_p:.0f}.\n"
                    f"Our highest price is ${df['selling_price'].max():.0f}."
                )
            return build_response(
                "😕 No products found matching your request.\n"
                "Try a different color, category, or price range.\n"
                "Type 'show all categories' to see what's available.",
                quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories", "📂 Categories"]
            )

        top_rows      = results.head(PAGE_SIZE)
        shown_products= [row.to_dict() for _, row in top_rows.iterrows()]

        SESSION_CACHE[session_id] = {
            "shown_ids":      results.index.tolist()[:PAGE_SIZE],
            "last_params":    params,
            "last_query":     query_text,
            "shown_products": shown_products,
        }

        if len(matched_products) > 1:
            header = "🛒 Multiple products found:"
        elif len(matched_products) == 1:
            header = "🎯 Here's what I found:"
        else:
            header = "🔥 Top picks for you:"

        separator = "─" * 28
        lines     = [format_product(row, index=i + 1)
                     for i, (_, row) in enumerate(top_rows.iterrows())]
        body      = f"\n{separator}\n".join(lines)
        message   = f"{header}\n\n{body}"
        if len(results) > PAGE_SIZE:
            message += f"\n{separator}\n💬 Say 'show more' to see more."

        chips = [f"View {i+1}" for i in range(len(top_rows))]
        if len(results) > PAGE_SIZE:
            chips.append("Show More")
        return build_response(message, quick_replies=chips)

    # ── GENDER FILTER INTENT ──
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
        top_rows       = gender_results.head(PAGE_SIZE)
        shown_products = [row.to_dict() for _, row in top_rows.iterrows()]
        SESSION_CACHE[session_id] = {
            "shown_ids":      gender_results.index.tolist()[:PAGE_SIZE],
            "last_params":    params,
            "last_query":     query_text,
            "shown_products": shown_products,
        }
        separator = "─" * 28
        lines     = [format_product(row, index=i + 1)
                     for i, (_, row) in enumerate(top_rows.iterrows())]
        body      = f"\n{separator}\n".join(lines)
        icon      = "👩" if gender == "women" else ("👨" if gender == "men" else "🧒")
        message   = f"{icon} Top {gender.title()} picks:\n\n{body}"
        if len(gender_results) > PAGE_SIZE:
            message += f"\n{separator}\n💬 Say 'show more' to see more."
        chips = [f"View {i+1}" for i in range(len(top_rows))]
        if len(gender_results) > PAGE_SIZE:
            chips.append("Show More")
        return build_response(message, quick_replies=chips)

    # ── FALLBACK (also handles stray number taps) ──
    fallback_num = re.match(r"^(?:view\s+|#?\s*)?([1-9])[.\s]*$",
                            query_text.strip(), re.IGNORECASE)
    if fallback_num:
        cache = SESSION_CACHE.get(session_id, {})
        shown = cache.get("shown_products", [])
        if shown:
            idx = max(0, min(int(fallback_num.group(1)) - 1, len(shown) - 1))
            row = shown[idx]
            card_text, image_url = build_product_detail_card(row)
            rich_card = [{
                "title":    row.get("name", "Product"),
                "subtitle": f"${float(row.get('selling_price', 0) or 0):.0f}  ⭐ {row.get('average_rating', '')}",
                "imageUri": image_url,
            }] if image_url else None
            return build_response(card_text, cards=rich_card,
                                  quick_replies=["Show More", "📂 Categories"])

    return build_response(
        "🤔 I didn't understand that.\n"
        "\n"
        "Try:\n"
        "• women shoes / men hoodies\n"
        "• black shoes under $100\n"
        "• compare Ultraboost and Supernova\n"
        "• a number (1, 2, 3) to view a product\n"
        "• 'help' for the full guide",
        quick_replies=["👩 Women", "👨 Men", "👟 Shoes", "👕 Clothing", "help"]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
