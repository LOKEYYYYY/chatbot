from flask import Flask, request, jsonify
import pandas as pd
import os
import re

app = Flask(__name__)

# ===== Load dataset =====
df = pd.read_csv("adidas_usa.csv")
df.columns = df.columns.str.strip().str.lower()

# Make sure these columns are usable
for col in ["name", "brand", "color", "category", "description", "breadcrumbs", "availability"]:
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
INTENT_COMPARE = "Compare Products"         # NEW: for product comparison
INTENT_PRODUCT_DETAIL = "Product Detail"    # NEW: for single product detail view
INTENT_AVAILABILITY = "Check Availability"  # NEW: check if product/color is available

PAGE_SIZE = 3
SESSION_CACHE = {}

# ===== Stop words: never count these as meaningful product-name tokens =====
GENERIC_WORDS = {
    "shoes", "shoe", "hoodie", "hoodies", "jacket", "jackets", "shirt", "shirts",
    "shorts", "pants", "clothing", "clothes", "wear", "apparel", "sneakers",
    "sneaker", "boots", "boot", "sandals", "sandal", "socks", "sock",
    "and", "or", "the", "a", "an", "for", "me", "please", "show", "find",
    "get", "want", "need", "some", "any", "under", "below", "above", "around",
    "with", "in", "on", "at", "of", "to", "my", "i", "like", "give",
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

        # Extract product terms
        item_terms = extract_query_item_terms(seg, df)
        if item_terms:
            product = item_terms[0]  # Take longest/most specific match

        if product:
            segments.append({
                "color": color,
                "product": product,
                "max_price": max_price,
                "raw": seg
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


def search_segment(color=None, product=None, max_price=None, preference=None):
    """
    Run a focused search for a single color+product+price segment.
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

    if "original_price" in results.columns:
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

    # Patterns for comparison queries
    vs_pattern = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", text)
    compare_pattern = re.search(
        r"(?:compare|comparison between|difference between|compare between)\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+)",
        text
    )
    and_pattern = re.search(r"(.+?)\s+and\s+(.+)", text)

    term_a, term_b = None, None

    if vs_pattern:
        term_a = vs_pattern.group(1).strip()
        term_b = vs_pattern.group(2).strip()
    elif compare_pattern:
        term_a = compare_pattern.group(1).strip()
        term_b = compare_pattern.group(2).strip()
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
        # Return the best-rated match
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

    def safe(val, prefix="", suffix="", decimals=None):
        if pd.isna(val) or val is None:
            return "N/A"
        if decimals is not None:
            return f"{prefix}{float(val):.{decimals}f}{suffix}"
        return f"{prefix}{val}{suffix}"

    lines = [
        f"📊 Comparison: {product_a.get('name', term_a)} vs {product_b.get('name', term_b)}",
        "",
        f"{'Attribute':<22} {'Product A':<35} {'Product B':<35}",
        "-" * 92,
        f"{'Name':<22} {str(product_a.get('name',''))[:34]:<35} {str(product_b.get('name',''))[:34]:<35}",
        f"{'Category':<22} {str(product_a.get('category',''))[:34]:<35} {str(product_b.get('category',''))[:34]:<35}",
        f"{'Color':<22} {str(product_a.get('color',''))[:34]:<35} {str(product_b.get('color',''))[:34]:<35}",
        f"{'Selling Price':<22} {safe(product_a.get('selling_price'), '$', '', 0):<35} {safe(product_b.get('selling_price'), '$', '', 0):<35}",
        f"{'Original Price':<22} {safe(product_a.get('original_price'), '$', '', 0):<35} {safe(product_b.get('original_price'), '$', '', 0):<35}",
        f"{'Rating':<22} {safe(product_a.get('average_rating'), '⭐', '', 1):<35} {safe(product_b.get('average_rating'), '⭐', '', 1):<35}",
        f"{'Reviews':<22} {safe(product_a.get('reviews_count')):<35} {safe(product_b.get('reviews_count')):<35}",
        f"{'Availability':<22} {str(product_a.get('availability',''))[:34]:<35} {str(product_b.get('availability',''))[:34]:<35}",
        "",
        f"📝 {product_a.get('name', term_a)} Description:",
        _truncate(str(product_a.get("description", "N/A")), 300),
        "",
        f"📝 {product_b.get('name', term_b)} Description:",
        _truncate(str(product_b.get("description", "N/A")), 300),
    ]

    return "\n".join(lines)


def _truncate(text, max_len=300):
    """Truncate text to max_len characters, appending '...' if cut."""
    if not text or text == "nan":
        return "N/A"
    return text[:max_len] + ("..." if len(text) > max_len else "")


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


# ===== NEW: Product detail / info =====
def get_product_detail(query_text):
    """
    Returns full detail for a single named product found in the query.
    Used when user asks 'tell me about X' or 'what is X'.
    """
    text = query_text.lower()
    matched = detect_products_from_text(text, df)
    if not matched:
        # Broader fallback: any row whose name is contained in query
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

    def safe(val, prefix="", suffix="", decimals=None):
        if pd.isna(val):
            return "N/A"
        if decimals is not None:
            return f"{prefix}{float(val):.{decimals}f}{suffix}"
        return f"{prefix}{val}{suffix}"

    lines = [
        f"🔍 {row.get('name', target)}",
        f"Category   : {row.get('category', 'N/A')}",
        f"Color      : {row.get('color', 'N/A')}",
        f"Price      : {safe(row.get('selling_price'), '$', '', 0)}",
        f"Rating     : {safe(row.get('average_rating'), '⭐', '', 1)} ({safe(row.get('reviews_count'))} reviews)",
        f"Availability: {row.get('availability', 'N/A')}",
        f"",
        f"📝 Description:",
        _truncate(str(row.get("description", "N/A")), 400),
    ]
    return "\n".join(lines)


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

    similar = similar[~mask]  # exclude the product itself
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
             'shoes above 80', 'over 200', '50 to 150', 'less than 120'
    Returns (min_price, max_price) — either may be None.
    """
    if not text:
        return None, None

    t = text.lower().replace(",", "")

    # between X and Y / X to Y
    m = re.search(r"between\s+\$?(\d+(?:\.\d+)?)\s+and\s+\$?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), float(m.group(2))

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

    return None, None


def format_product(row):
    name = row.get("name", "Unknown")
    price = row.get("selling_price", None)
    rating = row.get("average_rating", None)
    category = row.get("category", "")
    color = row.get("color", "")

    price_text = f"${price:.0f}" if pd.notna(price) else "N/A"
    rating_text = f"⭐{rating:.1f}" if pd.notna(rating) else ""

    return f"{name} | {category} | {color} | {price_text} {rating_text}"


def search_products(params, query_text=""):
    """
    Core search function. Filters df based on Dialogflow params,
    with additional price parsing from raw query text as fallback.
    """
    results = df.copy()

    brand = get_param(params, "brand")
    color = get_param(params, "color")
    products = get_param(params, "products")
    usage = get_param(params, "usage")
    preference = get_param(params, "preference")
    max_price = get_param(params, "max_price")
    price_range = get_param(params, "price_range")

    # ===== FILTERS =====
    if brand and str(brand).lower() != "adidas":
        return pd.DataFrame()  # Return empty df; caller handles the brand message

    if color and "color" in results.columns:
        results = results[results["color"].str.contains(str(color), case=False, na=False)]

    if products:
        product_mask = pd.Series(False, index=results.index)
        for col in ["name", "category", "description"]:
            if col in results.columns:
                product_mask |= results[col].str.contains(str(products), case=False, na=False)
        results = results[product_mask]

    # ===== STRONG USAGE FILTER =====
    if usage:
        usage_mask = pd.Series(False, index=results.index)
        for col in ["category", "description", "name"]:
            if col in results.columns:
                usage_mask |= results[col].str.contains(str(usage), case=False, na=False)
        if usage_mask.any():
            results = results[usage_mask]

    # ===== PRICE FILTER =====
    min_price, max_price_range = parse_price_range(price_range)

    if max_price:
        try:
            max_price_range = float(max_price)
        except Exception:
            pass

    # NEW: Also parse price from raw query text if params gave nothing
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
    if preference:
        pref = str(preference).lower()

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


@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True) or {}

    query_result = req.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName", "")
    params = query_result.get("parameters", {})
    session_id = req.get("session", "default-session")
    query_text = query_result.get("queryText", "")

    # ===== GREETING (catch-all for hi/hello even if Dialogflow maps it wrong) =====
    if GREETING_PATTERNS.match(query_text) or intent_name == INTENT_WELCOME:
        return jsonify({
            "fulfillmentText": (
                "👋 Hi there! Welcome to the Adidas USA store.\n"
                "I can help you find shoes, clothing, accessories and more.\n"
                "Try: 'running shoes under $100', 'best rated hoodies', "
                "'compare Ultraboost and Runfalcon', or 'show all categories'."
            )
        })

    # ===== HELP =====
    if intent_name == INTENT_HELP:
        return jsonify({
            "fulfillmentText": (
                "🆘 Here's what you can ask me:\n"
                "• 'shoes under 100' — filter by category and price\n"
                "• 'black running shoes' — filter by color and usage\n"
                "• 'cheap clothing' / 'best accessories' — sort by preference\n"
                "• 'shoes between 50 and 150' — price range\n"
                "• 'compare Ultraboost and Runfalcon' — side-by-side comparison\n"
                "• 'show all categories' — browse available categories\n"
                "• 'show more' — see next batch of results\n"
                "• 'pink hoodies and burgundy duffel bags under 500' — multi-product search"
            )
        })

    # ===== GOODBYE =====
    if intent_name == INTENT_GOODBYE:
        return jsonify({
            "fulfillmentText": "👋 Bye! Come back anytime if you want to search for more products."
        })

    # ===== NEGATIVE =====
    if intent_name == INTENT_NEGATIVE:
        return jsonify({
            "fulfillmentText": "Okay — tell me another color, brand, category, or budget and I'll search again."
        })

    # ===== LIST CATEGORIES =====
    if intent_name == INTENT_LIST_CATEGORIES:
        if "category" not in df.columns:
            return jsonify({"fulfillmentText": "I cannot find categories in the dataset."})

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
            return jsonify({"fulfillmentText": "No categories found in the dataset."})

        return jsonify({
            "fulfillmentText": "📂 Here are some categories I found:\n- " + "\n- ".join(categories)
        })

    # ===== COMPARE PRODUCTS (intent-driven or free-text detection) =====
    if intent_name == INTENT_COMPARE or is_comparison_query(query_text):
        result = compare_products(query_text)
        if result:
            return jsonify({"fulfillmentText": result})
        return jsonify({
            "fulfillmentText": "❌ I couldn't identify two products to compare. Try: 'compare Ultraboost and Runfalcon'."
        })

    # ===== PRODUCT DETAIL =====
    if intent_name == INTENT_PRODUCT_DETAIL or DETAIL_PATTERNS.search(query_text):
        detail = get_product_detail(query_text)
        if detail:
            return jsonify({"fulfillmentText": detail})
        # Fall through to product search if no specific product found

    # ===== PRODUCT SEARCH =====
    if intent_name == INTENT_PRODUCT_SEARCH or intent_name not in (
        INTENT_SHOW_MORE, INTENT_LIST_CATEGORIES, INTENT_HELP,
        INTENT_GOODBYE, INTENT_NEGATIVE, INTENT_WELCOME,
        INTENT_COMPARE, INTENT_PRODUCT_DETAIL, INTENT_AVAILABILITY,
    ):
        # ── MULTI-SEGMENT QUERY (e.g. "pink hoodies and burgundy duffel bags") ──
        segments = parse_multi_segment_query(query_text)

        if segments:
            all_lines = []
            cache_ids = []

            for seg in segments:
                seg_results = search_segment(
                    color=seg.get("color"),
                    product=seg.get("product"),
                    max_price=seg.get("max_price"),
                )

                color_label = seg.get("color", "").title() if seg.get("color") else ""
                product_label = seg.get("product", "").title()
                header = f"🛍️ {color_label} {product_label}".strip() + ":"

                if seg_results.empty:
                    all_lines.append(f"{header}\n  ❌ No products found.")
                else:
                    top = seg_results.head(PAGE_SIZE)
                    cache_ids.extend(top.index.tolist())
                    seg_lines = [f"  - {format_product(row)}" for _, row in top.iterrows()]
                    all_lines.append(header + "\n" + "\n".join(seg_lines))
                    if len(seg_results) > PAGE_SIZE:
                        all_lines[-1] += f"\n  (Say 'show more {product_label.lower()}' for more)"

            SESSION_CACHE[session_id] = {
                "shown_ids": cache_ids,
                "last_params": params,
                "last_query": query_text,
                "segments": segments,
            }

            message = "\n\n".join(all_lines)
            return jsonify({"fulfillmentText": message})

        # ── SINGLE-SEGMENT SEARCH ──
        brand = get_param(params, "brand")
        if brand and str(brand).lower() not in ("adidas", ""):
            return jsonify({
                "fulfillmentText": "❌ Item not found. We only carry Adidas products."
            })

        matched_products = detect_products_from_text(query_text, df)
        results = search_products(params, query_text)

        # If brand was invalid, search_products returns empty DataFrame
        if isinstance(results, tuple):
            return results  # should not happen now but safe guard

        category_mask = pd.Series(False, index=results.index)
        item_terms = extract_query_item_terms(query_text, df)

        if item_terms:
            for term in item_terms:
                for col in ["category", "name", "description"]:
                    if col in results.columns:
                        category_mask |= results[col].str.contains(
                            re.escape(term), case=False, na=False
                        )

        if item_terms:
            if category_mask.any():
                results = results[category_mask]
            else:
                results = results.iloc[0:0]  # No fallback: keep empty

        # NEW: If still empty after item_term filtering AND we have a raw query,
        # try a broader token-level text search as a final fallback ONLY when
        # no item terms were extracted (avoids wrong-category results).
        if results.empty and not item_terms and query_text:
            text_mask = extract_terms_from_query_text(query_text, df)
            if text_mask.any():
                results = df[text_mask].sort_values(
                    ["average_rating", "reviews_count"], ascending=False
                ).reset_index(drop=True)

        exact_matches = pd.DataFrame()
        name_mask = pd.Series(False, index=results.index)

        if matched_products and "name" in results.columns:
            for product_name in matched_products:
                name_mask |= results["name"].str.contains(
                    re.escape(product_name), case=False, na=False
                )
            exact_matches = results[name_mask]

        if matched_products and not exact_matches.empty:
            results.loc[name_mask, "priority"] = 1
            results["priority"] = results.get("priority", pd.Series(0, index=results.index)).fillna(0)
            results = results.sort_values(
                ["priority", "average_rating", "reviews_count"], ascending=False
            )

        if results.empty:
            return jsonify({
                "fulfillmentText": (
                    "😕 No products found matching your request.\n"
                    "Try a different color, category, brand, or price range.\n"
                    "Type 'show all categories' to see what's available."
                )
            })

        SESSION_CACHE[session_id] = {
            "shown_ids": results.index.tolist()[:PAGE_SIZE],
            "last_params": params,
            "last_query": query_text,
        }

        top_rows = results.head(PAGE_SIZE)
        lines = [format_product(row) for _, row in top_rows.iterrows()]

        if len(matched_products) > 1:
            message = "🛒 You selected multiple products:\n"
        elif len(matched_products) == 1:
            message = "🎯 Product found:\n"
        else:
            message = "🔥 Top picks for you:\n"

        message += "\n".join(f"- {line}" for line in lines)

        if len(results) > PAGE_SIZE:
            message += "\n\nSay 'show more' to see more."

        return jsonify({"fulfillmentText": message})

    # ===== SHOW MORE =====
    if intent_name == INTENT_SHOW_MORE:
        cache = SESSION_CACHE.get(session_id)

        if not cache:
            return jsonify({
                "fulfillmentText": "Please search for a product first."
            })

        # If user just says "more", reuse previous search
        if not any(v for v in params.values() if v not in (None, "", [], {})):
            params = cache.get("last_params", {})
            query_text = cache.get("last_query", "")

        # ── Multi-segment show more ──
        cached_segments = cache.get("segments")
        if cached_segments:
            # Determine which segment user wants more of, or re-show all
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
                    all_lines.append(f"{header}\n  ✅ No more products.")
                else:
                    top = seg_results.head(PAGE_SIZE)
                    shown_ids.extend(top.index.tolist())
                    seg_lines = [f"  - {format_product(row)}" for _, row in top.iterrows()]
                    all_lines.append(header + "\n" + "\n".join(seg_lines))

            cache["shown_ids"] = shown_ids

            if all(("No more" in l or "not found" in l.lower()) for l in all_lines):
                return jsonify({"fulfillmentText": "✅ No more products found for your search."})

            return jsonify({"fulfillmentText": "\n\n".join(all_lines)})

        # ── Single-segment show more ──
        results = search_products(params, query_text)

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
                    return jsonify({
                        "fulfillmentText": (
                            f"😕 Sorry, we do not have {color_param} {products_param}.\n"
                            f"Available colors: {', '.join(available_colors[:6])}."
                        )
                    })

        item_terms = extract_query_item_terms(query_text, df)
        if item_terms:
            category_mask = pd.Series(False, index=results.index)
            for term in item_terms:
                for col in ["category", "name", "description"]:
                    if col in results.columns:
                        category_mask |= results[col].str.contains(
                            re.escape(term), case=False, na=False
                        )
            results = results[category_mask]

        shown_ids = cache.get("shown_ids", [])
        results = results[~results.index.isin(shown_ids)]

        next_chunk = results.head(PAGE_SIZE)

        if next_chunk.empty:
            return jsonify({"fulfillmentText": "✅ No more products found."})

        cache["shown_ids"].extend(next_chunk.index.tolist())

        lines = []
        for _, item in next_chunk.iterrows():
            price = item.get("selling_price")
            price_text = f"${float(price):.0f}" if pd.notna(price) else "price not listed"
            name = item.get("name", "Unknown")
            brand_val = item.get("brand", "")
            category = item.get("category", "")
            color_val = item.get("color", "")

            parts = [str(name)]
            if brand_val and brand_val != "nan":
                parts.append(str(brand_val))
            if category and category != "nan":
                parts.append(str(category))
            if color_val and color_val != "nan":
                parts.append(str(color_val))
            parts.append(price_text)

            lines.append(" | ".join(parts))

        message = "More products:\n" + "\n".join(f"- {line}" for line in lines)
        return jsonify({"fulfillmentText": message})

    # ===== FALLBACK =====
    return jsonify({
        "fulfillmentText": (
            "🤔 I didn't understand that.\n"
            "Try: 'black shoes under 100', 'show me jackets', "
            "'compare Ultraboost vs Runfalcon', or type 'help' for options."
        )
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
