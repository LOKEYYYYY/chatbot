from flask import Flask, request, jsonify
import pandas as pd
import os
import re

app = Flask(__name__)

# ===== Load dataset =====
df = pd.read_csv("adidas_usa.csv")
df.columns = df.columns.str.strip().str.lower()

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
INTENT_SELECT_PRODUCT = "Select Product"
INTENT_GENDER_FILTER = "Gender Filter"

PAGE_SIZE = 3
SESSION_CACHE = {}
WISHLIST_CACHE = {}

# ===== Regex patterns =====
BACK_TO_RESULTS_PATTERN = re.compile(
    r"(?:back\s+to\s+results?|go\s+back|previous\s+results?|back)",
    re.IGNORECASE,
)
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
SURPRISE_PATTERN = re.compile(
    r"\b(?:surprise\s+me|random\s+pick|pick\s+(?:something|one)\s+for\s+me|recommend\s+(?:something|one)|just\s+pick\s+one)\b",
    re.IGNORECASE,
)
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

# ===== Generic words to ignore in product name matching =====
GENERIC_WORDS = {
    "shoes", "shoe", "sneakers", "sneaker", "boots", "boot", "sandals", "sandal",
    "slides", "slide", "footwear", "trainers", "trainer", "kicks",
    "hoodie", "hoodies", "jacket", "jackets", "shirt", "shirts", "shorts", "pants",
    "clothing", "clothes", "wear", "apparel", "tee", "tees", "pullover",
    "sweatshirt", "windbreaker", "windbreakers", "outerwear", "leggings", "tights",
    "dress", "jersey", "sweater", "top", "outfit",
    "socks", "sock", "gloves", "glove", "cap", "caps", "hat", "hats",
    "beanie", "beanies", "bag", "bags", "backpack", "backpacks",
    "accessories", "accessory", "gear",
    "and", "or", "the", "a", "an", "for", "me", "please", "show", "find",
    "get", "want", "need", "some", "any", "under", "below", "above", "around",
    "with", "in", "on", "at", "of", "to", "my", "i", "like", "give",
}

# ===== Entity synonym map =====
ENTITY_SYNONYM_MAP = {
    "footwear": "shoes", "shoes": "shoes", "shoe": "shoes",
    "sneakers": "shoes", "sneaker": "shoes", "trainers": "shoes", "trainer": "shoes",
    "running shoes": "shoes", "sport shoes": "shoes", "kicks": "shoes", "joggers": "shoes",
    "slides": "slides",
    "hoodie": "hoodie", "hoodies": "hoodie", "sweatshirt": "hoodie",
    "sweatshort": "hoodie", "pullover": "hoodie",
    "zip up hoodie": "hoodie", "zip hoodie": "hoodie",
    "t shirt": "t-shirt", "t-shirt": "t-shirt", "tshirt": "t-shirt",
    "tee": "t-shirt", "tees": "t-shirt",
    "shirt": "shirt", "shirts": "shirt", "short sleeve shirt": "shirt",
    "clothes": "clothing",
    "jacket": "jacket", "jackets": "jacket", "coat": "jacket", "coats": "jacket",
    "windbreaker": "windbreaker", "windbreakers": "windbreaker",
    "outerwear": "jacket", "outerwears": "jacket",
    "pants": "pants", "trousers": "pants", "joggers pants": "pants",
    "track pants": "pants", "sweatpants": "pants", "trouser": "pants",
    "jogger": "pants", "track pant": "pants", "sweatpant": "pants",
    "shorts": "shorts", "sport shorts": "shorts", "running shorts": "shorts",
    "socks": "socks", "sock": "socks", "ankle socks": "socks",
    "sports socks": "socks", "crew socks": "socks",
    "bag": "bag", "bags": "bag", "backpack": "backpack", "backpacks": "backpack",
    "gym bag": "bag", "gym bags": "bag",
    "duffel bag": "duffel", "duffel bags": "duffel", "duffle bags": "duffel",
    "sack": "bag", "sacks": "bag", "gym sacks": "bag", "gym sack": "bag",
    "cap": "cap", "caps": "cap", "hat": "hat", "hats": "hat",
    "beanie": "beanie", "beanies": "beanie", "headwear": "cap", "headwears": "cap",
    "gloves": "gloves", "glove": "gloves",
    "ball": "ball", "balls": "ball", "football": "ball",
    "soccer ball": "ball", "basketball ball": "ball",
    "accessories": "accessories", "accessory": "accessories", "gear": "accessories",
    "running": "running", "run": "running", "jog": "running", "jogging": "running",
    "training": "training", "gym": "training", "workout": "training", "exercise": "training",
    "soccer": "soccer", "football boots": "soccer", "cleats": "soccer", "cleat": "soccer",
    "golf": "golf", "basketball": "basketball", "climbing": "climbing",
    "cycling": "cycling", "bike": "cycling",
    "hiking": "hiking", "hike": "hiking", "trail": "hiking",
    "casual": "casual", "lifestyle": "casual", "everyday": "casual",
}

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

# Product-type terms vs subcategory/activity terms
PRODUCT_TYPE_CANONICALS = {
    "shoes", "hoodie", "shirt", "t-shirt", "jacket", "windbreaker", "pants",
    "shorts", "clothing", "socks", "bag", "backpack", "duffel", "cap", "hat",
    "beanie", "gloves", "ball", "accessories", "slides", "boots", "sandals",
}

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

GENDER_KEYWORDS = {
    "women": ["women", "woman", "female", "ladies", "girl", "girls", "womens", "her"],
    "men":   ["men", "man", "male", "guys", "guy", "mens", "his"],
    "kids":  ["kids", "kid", "children", "child", "junior", "youth", "boys", "girls"],
}

# ===== Typo correction map =====
TYPO_MAP = {
    r"\bshos\b": "shoes",   r"\bsheos\b": "shoes",  r"\bshose\b": "shoes",
    r"\bshes\b": "shoes",   r"\bshoees\b": "shoes",  r"\bsho\b": "shoes",
    r"\bsnaekers\b": "sneakers", r"\bsnekaers\b": "sneakers", r"\bsnkrs\b": "sneakers",
    r"\bhoodi\b": "hoodie", r"\bhooide\b": "hoodie", r"\bhoodei\b": "hoodie",
    r"\bjackt\b": "jacket", r"\bjakcet\b": "jacket",
    r"\bhikng\b": "hiking", r"\bhikig\b": "hiking",  r"\bhikking\b": "hiking",
    r"\brunng\b": "running", r"\brnning\b": "running",
    r"\btraning\b": "training", r"\btrainig\b": "training",
    r"\bbackpak\b": "backpack",
    r"\bclothng\b": "clothing", r"\bcloting\b": "clothing",
    r"\bcasul\b": "casual",  r"\bcasaul\b": "casual",
    r"\bsandels\b": "sandals", r"\bsandles\b": "sandals",
    r"\bpnats\b": "pants",
}


# ===== Helper functions =====

def resolve_entity_synonyms(query_text):
    if not query_text:
        return []
    text = query_text.lower().strip()
    found = {}
    for synonym in sorted(ENTITY_SYNONYM_MAP.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(synonym) + r"\b", text):
            found[ENTITY_SYNONYM_MAP[synonym]] = True
    return list(found.keys())


def build_csv_term_index(dataframe):
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
    for col in ["name", "description", "category"]:
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


def extract_query_item_terms(query_text, dataframe):
    if not query_text:
        return []
    text = query_text.lower()
    found_terms = []
    for term in CSV_TERM_INDEX:
        if not re.search(r"\b" + re.escape(term) + r"\b", text):
            continue
        for col in ["name", "description", "category"]:
            if col not in dataframe.columns:
                continue
            if dataframe[col].str.contains(re.escape(term), case=False, na=False).any():
                found_terms.append(term)
                break
    found_terms.sort(key=len, reverse=True)
    deduped = []
    for term in found_terms:
        if not any(term in longer for longer in deduped):
            deduped.append(term)
    return deduped


def extract_terms_from_query_text(query_text, dataframe):
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
        if sum(1 for w in words if w in text) == len(words):
            matched.append(name)
    return list(set(matched))


def detect_preference_from_text(query_text):
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
    if not query_text:
        return None
    text = query_text.lower()
    for kw in GENDER_KEYWORDS["kids"]:
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            return "kids"
    for kw in GENDER_KEYWORDS["women"]:
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            return "women"
    for kw in GENDER_KEYWORDS["men"]:
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            return "men"
    return None


def apply_gender_filter(results, gender):
    if not gender or results.empty:
        return results
    if gender == "women":
        search_terms = [r"women", r"womens", r"ladies", r"female"]
    elif gender == "men":
        search_terms = [r"\bmen\b", r"\bmens\b", r"\bmale\b"]
    elif gender == "kids":
        search_terms = [r"kids", r"junior", r"youth", r"child"]
    else:
        search_terms = [re.escape(gender)]
    gender_mask = pd.Series(False, index=results.index)
    for col in ["breadcrumbs", "name", "category", "description"]:
        if col not in results.columns:
            continue
        for term in search_terms:
            gender_mask |= results[col].str.contains(term, case=False, na=False, regex=True)
    filtered = results[gender_mask]
    return filtered if not filtered.empty else results


def strict_entity_filter(results, term):
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
    loose_mask = pd.Series(False, index=results.index)
    for t in variants:
        for col in ["description", "breadcrumbs"]:
            if col in results.columns:
                loose_mask |= results[col].str.contains(re.escape(t), case=False, na=False)
    return results[loose_mask] if loose_mask.any() else results


def is_comparison_query(query_text):
    triggers = [
        r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b",
        r"\bdifference between\b", r"\bcompare between\b",
    ]
    text = query_text.lower()
    return any(re.search(t, text) for t in triggers)


def infer_gender_from_row(row):
    text = " ".join([
        str(row.get("breadcrumbs", "")), str(row.get("name", "")),
        str(row.get("category", "")), str(row.get("description", "")),
    ]).lower()
    if re.search(r"\bwomen|\bwomens|\bladies|\bfemale|\bher\b|\bgirl", text):
        return "Women"
    if re.search(r"\bmen\b|\bmens\b|\bmale|\bhis\b|\bguy|\bboy\b", text):
        return "Men"
    if re.search(r"\bkids|\bjunior|\byouth|\bchild", text):
        return "Kids"
    return "Unisex"


def _truncate(text, max_len=300):
    if not text or text == "nan":
        return "N/A"
    return text[:max_len] + ("..." if len(text) > max_len else "")


def get_product_image(row):
    raw = row.get("images", "") if isinstance(row, dict) else str(row)
    if not raw or raw in ("nan", "None", ""):
        return None
    first = raw.split("~")[0].strip()
    return first if first.startswith("http") else None


def get_param(params, *names):
    for name in names:
        value = params.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def parse_price_from_text(text):
    if not text:
        return None, None
    t = text.lower().replace(",", "")
    m = re.search(r"between\s+\$?(\d+(?:\.\d+)?)\s+and\s+\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"\$?(\d+(?:\.\d+)?)\s+to\s+\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(?:under|below|less\s+than|up\s+to|max(?:imum)?|no\s+more\s+than|<)\s*\$?(\d+(?:\.\d+)?)", t)
    if m:
        return None, float(m.group(1))
    m = re.search(r"\$?(\d+(?:\.\d+)?)\s+or\s+(?:less|below)", t)
    if m:
        return None, float(m.group(1))
    m = re.search(r"(?:above|over|more\s+than|at\s+least|min(?:imum)?|>)\s*\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), None
    m = re.search(r"(?:around|approximately|about|~)\s*\$?(\d+(?:\.\d+)?)", t)
    if m:
        mid = float(m.group(1))
        return mid * 0.8, mid * 1.2
    return None, None


def _extract_global_price(text):
    m = re.search(
        r"(?:under|below|less\s+than|up\s+to|max(?:imum)?|<)\s*\$?(\d+(?:\.\d+)?)"
        r"|\$?(\d+(?:\.\d+)?)\s*(?:or\s+)?(?:less|below|max)",
        text
    )
    if m:
        nums = [float(x) for x in m.groups() if x is not None]
        return max(nums) if nums else None
    return None


def sanitize_query(query_text):
    if not query_text:
        return query_text
    text = query_text.strip()
    stripped = re.sub(r"[^\w\s]", "", text).strip().lower()
    chip_map = {
        "women": "women", "men": "men", "kids": "kids",
        "shoes": "shoes", "clothing": "clothing", "accessories": "accessories",
        "categories": "show all categories", "search again": "help",
        "show more": "show more", "view wishlist": "show my wishlist",
        "clear wishlist": "clear wishlist", "save this": "save this",
        "surprise me": "surprise me", "surprise me again": "surprise me",
    }
    if stripped in chip_map:
        return chip_map[stripped]
    text = text.lower().strip()
    num_match = re.match(
        r"^(?:view\s+|#\s*|select\s+|option\s+|number\s+|item\s+|pick\s+)?([1-9])[\.\s]*$", text
    )
    if num_match:
        return num_match.group(1)
    if re.match(r"^(?:more\s+)+$", text):
        return "show more"
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text)
    text = re.sub(r"\b(under|below|above|over|between)\s+\1\b", r"\1", text)
    for pattern, replacement in TYPO_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ===== Product formatting =====

def format_product(row, index=None):
    name = row.get("name", "Unknown")
    price = row.get("selling_price", None)
    orig_price = row.get("original_price", None)
    rating = row.get("average_rating", None)
    reviews = row.get("reviews_count", None)
    category = row.get("category", "")
    color = row.get("color", "")
    row_dict = row.to_dict() if hasattr(row, "to_dict") else row
    gender = infer_gender_from_row(row_dict)
    gender_text = f"👤 {gender}" if gender and gender != "Unisex" else ""
    price_text = f"${price:.0f}" if pd.notna(price) else "N/A"
    rating_text = f"⭐ {rating:.1f}" if pd.notna(rating) else ""
    reviews_text = f"({int(reviews)} reviews)" if pd.notna(reviews) and reviews else ""
    color_text = f"🎨 {color}" if color and color != "nan" else ""
    category_text = f"🏷️ {category}" if category and category != "nan" else ""
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


def build_product_detail_card(row):
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
        f"🛍️ {name}\n\n{stock_badge}\n\n"
        f"💰 Price      : {price}{discount_line}\n"
        f"🏷️ Category   : {category}   🎨 Color: {color}\n"
        f"👤 Gender     : {gender}\n"
        f"{star_str} Rating: {rating}\n"
        f"💬 Reviews    : {reviews}\n\n"
        f"📝 Description:\n{desc}"
    )
    return card, image_url


def get_product_detail(query_text):
    text = query_text.lower()
    matched = detect_products_from_text(text, df)
    if not matched and "name" in df.columns:
        for name in df["name"].dropna().unique():
            if str(name).lower() in text:
                matched = [name]
                break
    if not matched:
        return None
    mask = df["name"].str.contains(re.escape(matched[0]), case=False, na=False)
    row = df[mask].sort_values(["average_rating", "reviews_count"], ascending=False).iloc[0]
    return build_product_detail_card(row.to_dict())


def rebuild_results_message(shown_products, header="🔥 Top picks for you:", has_more=False):
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
    wishlist = WISHLIST_CACHE.setdefault(session_id, [])
    name = product_row.get("name", "")
    if not any(p.get("name") == name for p in wishlist):
        wishlist.append(product_row)
        return True
    return False


def format_wishlist(session_id):
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
    messages = []
    if cards:
        for card in cards:
            card_payload = {"title": card.get("title", ""), "subtitle": card.get("subtitle", "")}
            if card.get("imageUri"):
                card_payload["imageUri"] = card["imageUri"]
            if card.get("buttons"):
                card_payload["buttons"] = card["buttons"]
            messages.append({"card": card_payload})
    lines = [line for line in text.split("\n") if line.strip()]
    messages += [{"text": {"text": [line]}} for line in lines]
    if quick_replies:
        messages.append({"quickReplies": {"title": "Quick options:", "quickReplies": quick_replies}})
    return jsonify({"fulfillmentMessages": messages, "fulfillmentText": text})


# ===== Core search functions =====

def parse_price_range(value):
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


def parse_multi_segment_query(query_text):
    if not query_text:
        return []
    text = query_text.lower()
    if is_comparison_query(text):
        return []
    # Strip price-range "and" before splitting
    price_and_stripped = re.sub(r"between\s+\$?\d+(?:\.\d+)?\s+and\s+\$?\d+(?:\.\d+)?", "", text)
    price_and_stripped = re.sub(r"\$?\d+(?:\.\d+)?\s+and\s+\$?\d+(?:\.\d+)?", "", price_and_stripped)
    raw_segments = re.split(r"\band\b", price_and_stripped)
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
        if not seg:
            continue
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
        for color_word in colors_in_dataset:
            if re.search(r"\b" + re.escape(color_word) + r"\b", seg):
                color = color_word
                break
        entity_terms = resolve_entity_synonyms(seg)
        if entity_terms:
            product = entity_terms[0]
        else:
            item_terms = extract_query_item_terms(seg, df)
            if item_terms:
                product = item_terms[0]
        if product:
            segments.append({"color": color, "product": product, "max_price": max_price, "raw": seg})
    return segments if len(segments) >= 2 else []


def search_segment(color=None, product=None, max_price=None, gender=None):
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
    if gender:
        results = apply_gender_filter(results, gender)
    if "original_price" in results.columns and "selling_price" in results.columns:
        results["discount"] = results["original_price"] - results["selling_price"]
    else:
        results["discount"] = 0
    return results.sort_values(["average_rating", "reviews_count"], ascending=False).reset_index(drop=True)


def compare_products(query_text):
    text = query_text.lower()
    clean = re.sub(
        r"^(?:compare|comparison(?:\s+between)?|difference\s+between|compare\s+between)\s+", "", text
    ).strip()
    vs_pattern = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", clean)
    and_pattern = re.search(r"(.+?)\s+(?:and|with)\s+(.+)", clean)
    term_a, term_b = None, None
    if vs_pattern:
        term_a, term_b = vs_pattern.group(1).strip(), vs_pattern.group(2).strip()
    elif and_pattern:
        term_a, term_b = and_pattern.group(1).strip(), and_pattern.group(2).strip()
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
        return matched.sort_values(["average_rating", "reviews_count"], ascending=False).iloc[0]

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
        return "✅ In Stock" if ("in stock" in a or a in ("true", "1", "yes", "available")) else "❌ Out of Stock"

    def _discount(orig, sell):
        try:
            o, s = float(orig), float(sell)
            if o > s > 0:
                return f"🏷️ {int(round((o - s) / o * 100))}% off"
            return ""
        except Exception:
            return ""

    def product_card(p, label):
        price  = _s(p.get("selling_price"), "$", "", 0)
        orig   = _s(p.get("original_price"), "$", "", 0)
        disc   = _discount(p.get("original_price"), p.get("selling_price"))
        price_line = f"💰 {price}" + (f"  {disc}  (was {orig})" if disc else "")
        return (
            f"{label}\n👟 {p.get('name', 'Unknown')}\n\n"
            f"{_stock(p.get('availability', ''))}\n"
            f"🏷️ {_s(p.get('category'))}   🎨 {_s(p.get('color'))}   "
            f"👤 {infer_gender_from_row(p if isinstance(p, dict) else p.to_dict())}\n"
            f"{price_line}\n"
            f"{_stars(p.get('average_rating'))} {_s(p.get('average_rating'), '', '/5', 1)}"
            f"  💬 {_s(p.get('reviews_count'))} reviews\n\n"
            f"📝 {_truncate(str(p.get('description', 'N/A')), 220)}"
        )

    divider = "═" * 30
    winner_note = ""
    try:
        r_a, r_b = float(product_a.get("average_rating") or 0), float(product_b.get("average_rating") or 0)
        p_a, p_b = float(product_a.get("selling_price") or 0), float(product_b.get("selling_price") or 0)
        name_a, name_b = product_a.get("name", term_a), product_b.get("name", term_b)
        if r_a > r_b:
            winner_note = f"\n🏆 Better rated: {name_a}"
        elif r_b > r_a:
            winner_note = f"\n🏆 Better rated: {name_b}"
        if p_a and p_b:
            winner_note += f"\n💸 Better value: {name_a if p_a < p_b else name_b}"
    except Exception:
        pass

    return f"📊 Comparison\n{divider}\n{product_card(product_a, '🅰️ Product A')}\n{divider}\n{product_card(product_b, '🅱️ Product B')}\n{divider}{winner_note}"


def search_products(params, query_text=""):
    results = df.copy()
    brand    = get_param(params, "brand")
    color    = get_param(params, "color")
    products = get_param(params, "products")
    usage    = get_param(params, "usage")
    preference   = get_param(params, "preference")
    max_price    = get_param(params, "max_price")
    price_range  = get_param(params, "price_range")

    detected_gender = detect_gender_from_text(query_text)

    if brand and str(brand).lower() not in ("adidas", ""):
        return pd.DataFrame()

    # Color filter
    valid_colors = set(c.lower() for c in df["color"].dropna().unique())
    if color and "color" in results.columns:
        color_lower = str(color).lower().strip()
        if color_lower in valid_colors or any(color_lower in vc or vc in color_lower for vc in valid_colors):
            results = results[results["color"].str.contains(re.escape(color_lower), case=False, na=False)]
    if "color" in results.columns and len(results) == len(df) and query_text:
        for color_word in sorted(valid_colors, key=len, reverse=True):
            if re.search(r"\b" + re.escape(color_word) + r"\b", query_text.lower()):
                filtered = results[results["color"].str.contains(re.escape(color_word), case=False, na=False)]
                if not filtered.empty:
                    results = filtered
                    break

    # Product / category filter
    detected_product_type = None
    if products:
        products_str = str(products).lower().strip()
        products_resolved = ENTITY_SYNONYM_MAP.get(products_str, products_str)
        detected_product_type = products_resolved
        product_mask = pd.Series(False, index=results.index)
        for search_term in {products_resolved, products_str}:
            for col in ["name", "category", "description"]:
                if col in results.columns:
                    product_mask |= results[col].str.contains(re.escape(search_term), case=False, na=False)
        if product_mask.any():
            results = results[product_mask]
    elif query_text:
        entity_terms = resolve_entity_synonyms(query_text)
        product_type_terms = [t for t in entity_terms if t in PRODUCT_TYPE_CANONICALS]
        activity_terms     = [t for t in entity_terms if t not in PRODUCT_TYPE_CANONICALS]
        if product_type_terms:
            detected_product_type = product_type_terms[0]
            combined_mask = pd.Series(False, index=results.index)
            for term in product_type_terms:
                candidate = strict_entity_filter(results, term)
                combined_mask |= pd.Series(results.index.isin(candidate.index), index=results.index)
            if combined_mask.any():
                results = results[combined_mask]
            if activity_terms:
                activity_mask = pd.Series(False, index=results.index)
                for term in activity_terms:
                    for col in ["name", "category", "description", "breadcrumbs"]:
                        if col in results.columns:
                            activity_mask |= results[col].str.contains(re.escape(term), case=False, na=False)
                if activity_mask.any():
                    refined = results[activity_mask]
                    if not refined.empty:
                        results = refined
        elif activity_terms:
            combined_mask = pd.Series(False, index=results.index)
            for term in activity_terms:
                candidate = strict_entity_filter(results, term)
                combined_mask |= pd.Series(results.index.isin(candidate.index), index=results.index)
            if combined_mask.any():
                results = results[combined_mask]
        else:
            detected_cat = detect_category_from_text(query_text)
            if detected_cat and "category" in results.columns:
                results = results[results["category"].str.contains(detected_cat, case=False, na=False)]

    # Subcategory filter (only when no hard product type pinned)
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
    elif query_text and detected_product_type is None:
        subcat_keywords = detect_subcategory_from_text(query_text)
        if subcat_keywords:
            subcat_mask = pd.Series(False, index=results.index)
            for kw in subcat_keywords:
                for col in ["name", "description", "breadcrumbs"]:
                    if col in results.columns:
                        subcat_mask |= results[col].str.contains(re.escape(kw), case=False, na=False)
            if subcat_mask.any():
                results = results[subcat_mask]

    # Price filter
    min_price, max_price_range = parse_price_range(price_range)
    if max_price:
        try:
            max_price_range = float(max_price)
        except Exception:
            pass
    if min_price is None and max_price_range is None:
        min_price, max_price_range = parse_price_from_text(query_text)
    else:
        text_min, text_max = parse_price_from_text(query_text)
        if text_min is not None and min_price is None:
            min_price = text_min
        if text_max is not None and max_price_range is None:
            max_price_range = text_max

    if "selling_price" in results.columns:
        results = results[results["selling_price"].notna()]
        if min_price is not None:
            results = results[results["selling_price"] >= min_price]
        if max_price_range is not None:
            results = results[results["selling_price"] <= max_price_range]

    # Discount column
    if "original_price" in results.columns and "selling_price" in results.columns:
        results["discount"] = results["original_price"] - results["selling_price"]
    else:
        results["discount"] = 0

    # Sorting
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

    # Gender filter
    if detected_gender:
        gender_filtered = apply_gender_filter(results, detected_gender)
        if not gender_filtered.empty:
            results = gender_filtered

    return results.reset_index(drop=True)


# ===== Webhook =====

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True) or {}
    query_result = req.get("queryResult", {})
    intent_name  = query_result.get("intent", {}).get("displayName", "")
    params       = query_result.get("parameters", {})
    session_id   = req.get("session", "default-session")
    query_text   = sanitize_query(query_result.get("queryText", ""))

    # ===== GREETING =====
    if GREETING_PATTERNS.match(query_text) or intent_name == INTENT_WELCOME:
        return build_response(
            "👋 Hi! Welcome to the Adidas USA Store.\n\n"
            "🛍️ Here's what I can do for you:\n\n"
            "1️⃣  Browse by gender\n   → Type: women shoes  /  men hoodies  /  kids clothing\n\n"
            "2️⃣  Search by category & color\n   → Type: black running shoes under $100\n\n"
            "3️⃣  Sort by preference\n   → Type: cheapest shoes  /  best rated hoodies\n\n"
            "4️⃣  Compare products\n   → Type: compare Ultraboost and Runfalcon\n\n"
            "5️⃣  Get product details\n   → After results appear, reply with the number (1, 2 or 3)\n\n"
            "6️⃣  Save favourites\n   → View a product, then tap 'Save this' to wishlist it\n\n"
            "7️⃣  Feeling lucky?\n   → Type: surprise me\n\n"
            "💡 Type 'help' anytime to see this guide again.",
            quick_replies=["👩 Women", "👨 Men", "👟 Shoes", "👕 Clothing", "🎒 Accessories", "🎲 Surprise Me"]
        )

    # ===== HELP =====
    if intent_name == INTENT_HELP:
        return build_response(
            "🆘 Here's what I can do:\n\n"
            "🔍 Search by category & price — e.g. shoes under 100\n"
            "🎨 Filter by color — e.g. black running shoes\n"
            "⭐ Sort by preference — e.g. cheap clothing / best accessories\n"
            "💰 Price range — e.g. shoes between 50 and 150\n"
            "📊 Compare products — e.g. compare Ultraboost and Runfalcon\n"
            "📂 Browse categories — Type: show all categories\n"
            "💖 Save favourites — View a product → tap 'Save this'\n"
            "🎲 Random pick — Type: surprise me\n"
            "➡️ See more results — Type: show more\n"
            "⬅️ Go back — Type: back to results",
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
            df["category"].dropna().astype(str).str.strip()
            .replace("", pd.NA).dropna().str.lower().drop_duplicates().head(15).tolist()
        )
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

    # ===== WISHLIST: VIEW =====
    if WISHLIST_VIEW_PATTERN.search(query_text):
        wl_text, wl_chips = format_wishlist(session_id)
        wishlist = WISHLIST_CACHE.get(session_id, [])
        if wishlist:
            SESSION_CACHE.setdefault(session_id, {})["shown_products"] = wishlist
        return build_response(wl_text, quick_replies=wl_chips or ["👟 Shoes", "👕 Clothing"])

    # ===== WISHLIST: CLEAR =====
    if WISHLIST_CLEAR_PATTERN.search(query_text):
        WISHLIST_CACHE[session_id] = []
        return build_response("🗑️ Wishlist cleared! Start browsing to add new items.",
                              quick_replies=["👩 Women", "👨 Men", "👟 Shoes", "👕 Clothing"])

    # ===== WISHLIST: SAVE =====
    if WISHLIST_ADD_PATTERN.search(query_text):
        cache = SESSION_CACHE.get(session_id, {})
        target = cache.get("last_viewed_product") or (cache.get("shown_products") or [None])[0]
        if target:
            added = add_to_wishlist(session_id, target)
            name = target.get("name", "Product")
            msg = (
                f"💖 Added '{name}' to your wishlist!\nYou have {len(WISHLIST_CACHE.get(session_id, []))} item(s) saved."
                if added else f"✅ '{name}' is already in your wishlist."
            )
            return build_response(msg, quick_replies=["💖 View Wishlist", "Show More", "« Back to results"])
        return build_response("😕 No product selected to save. Browse and view a product first, then say 'save this'.",
                              quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"])

    # ===== SURPRISE ME =====
    if SURPRISE_PATTERN.search(query_text):
        import random
        surprise = df[df["selling_price"].notna()].copy()
        if "average_rating" in surprise.columns:
            surprise = surprise[surprise["average_rating"] >= 4.0]
        if surprise.empty:
            surprise = df[df["selling_price"].notna()].copy()
        pick = surprise.sample(1).iloc[0]
        card_text, image_url = build_product_detail_card(pick.to_dict())
        rich_card = [{
            "title": pick.get("name", "Product"),
            "subtitle": f"${float(pick.get('selling_price', 0) or 0):.0f}  ⭐ {pick.get('average_rating', '')}",
            "imageUri": image_url,
        }] if image_url else None
        cache = SESSION_CACHE.setdefault(session_id, {})
        cache["shown_products"] = [pick.to_dict()]
        cache["last_viewed_product"] = pick.to_dict()
        return build_response(
            "🎲 Here's a random top-rated pick for you!\n\n" + card_text,
            quick_replies=["🎲 Surprise Me Again", "💖 Save this", "Show More", "« Back to results"],
            cards=rich_card
        )

    # ===== BACK TO RESULTS =====
    if BACK_TO_RESULTS_PATTERN.search(query_text):
        cache = SESSION_CACHE.get(session_id, {})
        last_msg = cache.get("last_result_message")
        if last_msg and cache.get("shown_products"):
            return build_response(last_msg, quick_replies=cache.get("last_result_chips") or [])
        return build_response("😕 No previous results to go back to. Try a new search!",
                              quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"])

    # ===== SHOW MORE =====
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
        if not any(v for v in params.values() if v not in (None, "", [], {})):
            params = cache.get("last_params", {})
            query_text_for_more = cache.get("last_query", "")
        else:
            query_text_for_more = query_text

        # Multi-segment show more
        cached_segments = cache.get("segments")
        if cached_segments:
            all_lines = []
            shown_ids = cache.get("shown_ids", [])
            for seg in cached_segments:
                seg_results = search_segment(
                    color=seg.get("color"), product=seg.get("product"), max_price=seg.get("max_price")
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
                    separator = "─" * 28
                    seg_lines = [format_product(row, index=i + 1) for i, (_, row) in enumerate(top.iterrows())]
                    all_lines.append(f"{header}\n\n" + f"\n{separator}\n".join(seg_lines))
            cache["shown_ids"] = shown_ids
            if all("No more" in l or "not found" in l.lower() for l in all_lines):
                return build_response("✅ No more products found for your search.")
            multi_more_msg = "\n\n".join(all_lines)
            cache["last_result_message"] = multi_more_msg
            cache["last_result_chips"] = []
            return build_response(multi_more_msg)

        # Single-segment show more
        results = search_products(params, query_text_for_more)
        item_terms_raw     = extract_query_item_terms(query_text_for_more, df)
        entity_terms_direct = resolve_entity_synonyms(query_text_for_more)
        all_search_terms_more = set()
        for t in entity_terms_direct + item_terms_raw:
            canonical = ENTITY_SYNONYM_MAP.get(t, t)
            all_search_terms_more.update([
                t, canonical,
                t[:-1] if t.endswith("s") and len(t) > 3 else t + "s",
                canonical[:-1] if canonical.endswith("s") and len(canonical) > 3 else canonical + "s"
            ])
        if all_search_terms_more:
            category_mask = pd.Series(False, index=results.index)
            for term in all_search_terms_more:
                for col in ["category", "name", "description", "breadcrumbs"]:
                    if col in results.columns:
                        category_mask |= results[col].str.contains(re.escape(term), case=False, na=False)
            if category_mask.any():
                results = results[category_mask]

        shown_ids = cache.get("shown_ids", [])
        remaining = results[~results.index.isin(shown_ids)]
        next_chunk = remaining.head(PAGE_SIZE)
        if next_chunk.empty:
            return build_response("✅ No more products found. Try a new search!")

        cache["shown_ids"] = shown_ids + next_chunk.index.tolist()
        has_more_after = len(remaining) > PAGE_SIZE
        shown_products_more = [row.to_dict() for _, row in next_chunk.iterrows()]
        cache["shown_products"] = shown_products_more
        separator = "─" * 28
        lines = [format_product(row, index=i + 1) for i, (_, row) in enumerate(next_chunk.iterrows())]
        message = f"📦 More results:\n\n" + f"\n{separator}\n".join(lines)
        if has_more_after:
            message += f"\n{separator}\n💬 Say 'show more' to see more."
        chips_more = [f"View {i+1}" for i in range(len(next_chunk))]
        if has_more_after:
            chips_more.append("Show More")
        cache["last_result_message"] = message
        cache["last_result_chips"] = chips_more
        return build_response(message, quick_replies=chips_more)

    # ===== SELECT PRODUCT BY NUMBER =====
    number_match = re.match(r"^(?:view\s+)?([1-9])$", query_text.strip(), re.IGNORECASE)
    if number_match or intent_name == INTENT_SELECT_PRODUCT:
        cache = SESSION_CACHE.get(session_id, {})
        shown_products = cache.get("shown_products", [])
        if not shown_products:
            return build_response("🔍 No recent results to select from. Please search for a product first.",
                                  quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories"])
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
        cache["last_viewed_product"] = row
        SESSION_CACHE[session_id] = cache
        rich_card = [{
            "title": row.get("name", "Product"),
            "subtitle": f"${float(row.get('selling_price', 0) or 0):.0f}  ⭐ {row.get('average_rating', '')}",
            "imageUri": image_url,
        }] if image_url else None
        return build_response(card_text,
                              quick_replies=["« Back to results", "💖 Save this", "Show More", "📂 Categories"],
                              cards=rich_card)

    # ===== PRODUCT SEARCH =====
    if intent_name == INTENT_PRODUCT_SEARCH or intent_name not in (
        INTENT_SHOW_MORE, INTENT_LIST_CATEGORIES, INTENT_HELP, INTENT_GOODBYE,
        INTENT_NEGATIVE, INTENT_WELCOME, INTENT_COMPARE, INTENT_PRODUCT_DETAIL,
        INTENT_SELECT_PRODUCT, INTENT_GENDER_FILTER,
    ):
        if NONSENSE_PATTERNS.search(query_text):
            return build_response(
                "😅 Hmm, we don't carry that in our catalog!\n"
                "Try a real color (black, white, red...) or product type.\n"
                "Type 'show all categories' to browse what we have."
            )

        # Multi-segment
        segments = parse_multi_segment_query(query_text)
        if segments:
            all_lines = []
            cache_ids = []
            all_shown_products = []
            gender = detect_gender_from_text(query_text)
            for seg in segments:
                seg_results = search_segment(
                    color=seg.get("color"), product=seg.get("product"),
                    max_price=seg.get("max_price"), gender=gender,
                )
                color_label   = seg.get("color", "").title() if seg.get("color") else ""
                product_label = seg.get("product", "").title()
                gender_label  = f" ({gender.title()})" if gender else ""
                header = f"🛍️ {color_label} {product_label}{gender_label}".strip() + ":"
                if seg_results.empty:
                    all_lines.append(f"{header}\n❌ No products found.")
                else:
                    top = seg_results.head(PAGE_SIZE)
                    cache_ids.extend(top.index.tolist())
                    seg_products = [row.to_dict() for _, row in top.iterrows()]
                    all_shown_products.extend(seg_products)
                    separator = "─" * 28
                    start_idx = len(all_shown_products) - len(seg_products)
                    seg_lines = [format_product(row, index=start_idx + i + 1) for i, (_, row) in enumerate(top.iterrows())]
                    entry = f"{header}\n\n" + f"\n{separator}\n".join(seg_lines)
                    if len(seg_results) > PAGE_SIZE:
                        entry += f"\n{separator}\n💬 Say 'show more' for more."
                    all_lines.append(entry)
            multi_message = "\n\n".join(all_lines)
            multi_chips = [f"View {i+1}" for i in range(min(len(all_shown_products), 9))]
            multi_chips.append("Show More")
            SESSION_CACHE[session_id] = {
                "shown_ids": cache_ids, "last_params": params, "last_query": query_text,
                "segments": segments, "shown_products": all_shown_products,
                "last_result_message": multi_message, "last_result_chips": multi_chips,
            }
            return build_response(multi_message, quick_replies=multi_chips)

        # Single-segment
        brand = get_param(params, "brand")
        if brand and str(brand).lower() not in ("adidas", ""):
            return build_response("❌ Item not found. We only carry Adidas products.")

        matched_products = detect_products_from_text(query_text, df)
        results = search_products(params, query_text)

        if results.empty:
            min_p, max_p = parse_price_from_text(query_text)
            if max_p is not None and max_p < 10:
                return build_response(
                    f"😕 No products found under ${max_p:.0f}. "
                    f"Our lowest price is ${df['selling_price'].min():.0f}. Try a higher budget?"
                )
            if min_p is not None and min_p > df["selling_price"].max():
                return build_response(
                    f"😕 No products above ${min_p:.0f}. "
                    f"Our highest price is ${df['selling_price'].max():.0f}."
                )
            # Did you mean?
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
                        "last_params": params, "last_query": corrected,
                        "shown_products": shown_products_corr,
                        "last_result_message": msg_corr, "last_result_chips": chips_corr,
                    }
                    return build_response(msg_corr, quick_replies=chips_corr)
            return build_response(
                "😕 No products found matching your request.\n"
                "Try a different color, category, brand, or price range.\n"
                "Type 'show all categories' to see what's available.",
                quick_replies=["👟 Shoes", "👕 Clothing", "🎒 Accessories", "🎲 Surprise Me"]
            )

        # Boost exact name matches to top
        if matched_products and "name" in results.columns:
            name_mask = pd.Series(False, index=results.index)
            for product_name in matched_products:
                name_mask |= results["name"].str.contains(re.escape(product_name), case=False, na=False)
            if name_mask.any():
                results["priority"] = pd.Series(0, index=results.index)
                results.loc[name_mask, "priority"] = 1
                results = results.sort_values(["priority", "average_rating", "reviews_count"], ascending=False)

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
            "shown_ids": results.index.tolist()[:PAGE_SIZE], "last_params": params,
            "last_query": query_text, "shown_products": shown_products,
            "last_result_message": message, "last_result_chips": chips,
            "total_results": len(results), "all_result_ids": results.index.tolist(),
        }
        return build_response(message, quick_replies=chips)

    # ===== GENDER FILTER INTENT =====
    if intent_name == INTENT_GENDER_FILTER:
        gender = detect_gender_from_text(query_text)
        if not gender:
            return build_response("Please specify: women, men, or kids.",
                                  quick_replies=["👩 Women", "👨 Men", "🧒 Kids"])
        gender_results = apply_gender_filter(df.copy(), gender)
        gender_results = gender_results.sort_values(["average_rating", "reviews_count"], ascending=False).reset_index(drop=True)
        if gender_results.empty:
            return build_response(f"😕 No products found for {gender}.")
        top_rows = gender_results.head(PAGE_SIZE)
        shown_products = [row.to_dict() for _, row in top_rows.iterrows()]
        icon = "👩" if gender == "women" else ("👨" if gender == "men" else "🧒")
        gender_header = f"{icon} Top {gender.title()} picks:"
        has_more_gender = len(gender_results) > PAGE_SIZE
        gender_message, gender_chips = rebuild_results_message(shown_products, header=gender_header, has_more=has_more_gender)
        SESSION_CACHE[session_id] = {
            "shown_ids": gender_results.index.tolist()[:PAGE_SIZE], "last_params": params,
            "last_query": query_text, "shown_products": shown_products,
            "last_result_message": gender_message, "last_result_chips": gender_chips,
            "total_results": len(gender_results), "all_result_ids": gender_results.index.tolist(),
        }
        return build_response(gender_message, quick_replies=gender_chips)

    # ===== FALLBACK =====
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
            return build_response(card_text,
                                  quick_replies=["« Back to results", "💖 Save this", "Show More", "📂 Categories"],
                                  cards=rich_card)

    return build_response(
        "🤔 I didn't understand that.\n\n"
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
