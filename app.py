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

PAGE_SIZE = 3
SESSION_CACHE = {}

# ===== Stop words: never count these as meaningful product-name tokens =====
# This prevents "shoes" or "hoodie" in the query from matching unrelated product names
# word-by-word, and stops pure category words from triggering detect_products_from_text.
GENERIC_WORDS = {
    "shoes", "shoe", "hoodie", "hoodies", "jacket", "jackets", "shirt", "shirts",
    "shorts", "pants", "clothing", "clothes", "wear", "apparel", "sneakers",
    "sneaker", "boots", "boot", "sandals", "sandal", "socks", "sock",
    "and", "or", "the", "a", "an", "for", "me", "please", "show", "find",
    "get", "want", "need", "some", "any", "under", "below", "above", "around",
    "with", "in", "on", "at", "of", "to", "my", "i", "like", "give",
}

# ===== Build a dynamic term index from the CSV at startup =====
# Scans every product name and description to collect all searchable
# item-type keywords that actually exist in the dataset.  At query time we
# use this index instead of a hardcoded list, so ANY item type
# (tee, pants, slides, bag, tights, socks, backpack …) can be paired.

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

    This replaces the old hardcoded ["shoes","hoodie","shorts","shirt","jacket"]
    list so that ANY pairing works — tee + pants, socks + bag, slides + tights,
    tights + backpack, etc.
    """
    if not query_text:
        return []

    text = query_text.lower()
    found_terms = []

    for term in CSV_TERM_INDEX:
        # Term must appear as a whole word/phrase boundary in the query
        pattern = r"\b" + re.escape(term) + r"\b"
        if not re.search(pattern, text):
            continue

        # Confirm the term actually matches something in the dataset
        for col in ["name", "description", "category"]:
            if col not in dataframe.columns:
                continue
            if dataframe[col].str.contains(re.escape(term), case=False, na=False).any():
                found_terms.append(term)
                break

    # Remove shorter terms that are substrings of longer matched phrases
    # e.g. if both "tee" and "graphic tee" matched, keep only "graphic tee"
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
    every meaningful token in the user query, so the backend can surface
    results even when Dialogflow extracts nothing.

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

    # A row matches if ANY meaningful query token appears in its name or description
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

        # Strip generic/stop words so "Advantage Shoes" is not matched just
        # because the user said "shoes" — only brand/model tokens count.
        words = [w for w in name_lower.split() if w not in GENERIC_WORDS]

        # If all words were generic, skip this product name entirely
        if not words:
            continue

        # Require ALL meaningful keywords to appear in the query
        match_count = sum(1 for w in words if w in text)

        if match_count == len(words):
            matched.append(name)

    return list(set(matched))


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
        return numbers[0], numbers[0]

    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def format_product(row):
    name = row.get("name", "Unknown")
    price = row.get("selling_price", None)
    rating = row.get("average_rating", None)
    category = row.get("category", "")
    color = row.get("color", "")

    price_text = f"${price:.0f}" if pd.notna(price) else "N/A"
    rating_text = f"⭐{rating:.1f}" if pd.notna(rating) else ""

    return f"{name} | {category} | {color} | {price_text} {rating_text}"


def search_products(params):
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
        return jsonify({
            "fulfillmentText": "Item not found. We only carry Adidas products."
        })

    if color and "color" in results.columns:
        results = results[results["color"].str.contains(str(color), case=False, na=False)]

    if products:
        product_mask = pd.Series(False, index=results.index)

        for col in ["name", "category", "description"]:
            if col in results.columns:
                product_mask |= results[col].str.contains(str(products), case=False, na=False)

        results = results[product_mask]
    # ===== STRONG USAGE FILTER (NEW) =====
    if usage:
        usage_mask = pd.Series(False, index=results.index)

        for col in ["category", "description", "name"]:
            if col in results.columns:
                usage_mask |= results[col].str.contains(str(usage), case=False, na=False)

        # Only apply if it actually finds something
        if usage_mask.any():
            results = results[usage_mask]

    # ===== PRICE FILTER =====
    min_price, max_price_range = parse_price_range(price_range)

    if max_price:
        try:
            max_price_range = float(max_price)
        except:
            pass

    if "selling_price" in results.columns:
        results = results[results["selling_price"].notna()]

        if min_price is not None:
            results = results[results["selling_price"] >= min_price]

        if max_price_range is not None:
            results = results[results["selling_price"] <= max_price_range]

    # ===== DISCOUNT CALCULATION =====
    if "original_price" in results.columns:
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
        # default smart ranking
        results = results.sort_values(
            ["average_rating", "reviews_count"],
            ascending=False
        )

    return results.reset_index(drop=True)


@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True) or {}

    query_result = req.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName", "")
    params = query_result.get("parameters", {})
    session_id = req.get("session", "default-session")

    # Help
    if intent_name == INTENT_HELP:
        return jsonify({
            "fulfillmentText": (
                "Try a product search using color, brand, category, or price. "
                "Examples: 'red sneakers under 120', 'show me jackets', 'nike shoes under 150'."
            )
        })

    # Goodbye
    if intent_name == INTENT_GOODBYE:
        return jsonify({
            "fulfillmentText": "Bye! Come back anytime if you want to search for more products."
        })

    # Negative
    if intent_name == INTENT_NEGATIVE:
        return jsonify({
            "fulfillmentText": "Okay — tell me another color, brand, category, or budget and I'll search again."
        })

    # List categories
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
            "fulfillmentText": "Here are some categories I found:\n- " + "\n- ".join(categories)
        })

    # Product search
    if intent_name == INTENT_PRODUCT_SEARCH:

        query_text = query_result.get("queryText", "")

        matched_products = detect_products_from_text(query_text, df)

        results = search_products(params)

        # FIX 2: Always initialise category_mask to all-False BEFORE use.
        category_mask = pd.Series(False, index=results.index)

        # FIX 3: Use the dynamic CSV term index instead of a hardcoded list.
        # extract_query_item_terms scans the entire dataset vocabulary so any
        # item type (tee, pants, tights, bag, socks, slides …) can be paired,
        # not just the 5 terms that were previously hard-coded.
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
            # ---- CSV-backed fallback ----
            # Dialogflow extracted nothing AND no item terms were found via the
            # index — run a broad keyword search over name + description so the
            # backend can still find relevant products on its own.
            fallback_mask = extract_terms_from_query_text(query_text, results)
            if fallback_mask.any():
                results = results[fallback_mask]

        # FIX 4: Always initialise exact_matches to an empty DataFrame so the
        # reference below is safe even when matched_products is empty.
        exact_matches = pd.DataFrame()

        if matched_products and "name" in results.columns:
            name_mask = pd.Series(False, index=results.index)

            for product_name in matched_products:
                name_mask |= results["name"].str.contains(product_name, case=False, na=False)

            exact_matches = results[name_mask]

        if matched_products and not exact_matches.empty:
            # Prioritize exact matches but DO NOT discard others
            results["priority"] = 0
            results.loc[name_mask, "priority"] = 1

            results = results.sort_values(
                ["priority", "average_rating", "reviews_count"],
                ascending=False
            )

        if results.empty:
            return jsonify({
                "fulfillmentText": "No products found. Try a different color, category, brand, or price."
            })

        # Save results for Show More
        SESSION_CACHE[session_id] = {
            "results": results.to_dict("records"),
            "next_index": PAGE_SIZE
        }

        top_rows = results.head(PAGE_SIZE)
        lines = [format_product(row) for _, row in top_rows.iterrows()]

        # FIX 5: Consolidate the header into a single message so there is never a
        # duplicated emoji prefix (the old code prepended message_prefix and then
        # built a second "🎯 I found these exact products:" header inside `message`).
        if len(matched_products) > 1:
            message = "🛒 You selected multiple products:\n"
        elif len(matched_products) == 1:
            message = "🎯 Product found:\n"
        else:
            message = "🔥 Top picks for you:\n"

        message += "\n".join(f"- {line}" for line in lines)

        if len(results) > PAGE_SIZE:
            message += "\nSay 'show more' to see more."

        return jsonify({"fulfillmentText": message})

    # Show more
    if intent_name == INTENT_SHOW_MORE:
        cache = SESSION_CACHE.get(session_id)

        if not cache or not cache.get("results"):
            return jsonify({
                "fulfillmentText": "I do not have earlier results yet. Try a product search first."
            })

        all_results = cache["results"]
        start = cache.get("next_index", PAGE_SIZE)
        chunk = all_results[start:start + PAGE_SIZE]

        if not chunk:
            return jsonify({
                "fulfillmentText": "That is all I found from the previous search."
            })

        cache["next_index"] = start + PAGE_SIZE

        lines = []
        for item in chunk:
            price = item.get("selling_price")
            if price is not None and price == price:
                price_text = f"${float(price):.0f}"
            else:
                price_text = "price not listed"

            name = item.get("name", "Unknown")
            brand = item.get("brand", "")
            category = item.get("category", "")
            color = item.get("color", "")

            parts = [str(name)]
            if brand:
                parts.append(str(brand))
            if category:
                parts.append(str(category))
            if color:
                parts.append(str(color))
            parts.append(price_text)

            lines.append(" | ".join(parts))

        message = "More products:\n" + "\n".join(f"- {line}" for line in lines)

        if cache["next_index"] < len(all_results):
            message += "\nSay 'show more' again for more."

        return jsonify({"fulfillmentText": message})

    # Fallback
    return jsonify({
        "fulfillmentText": "I did not understand that. Try 'black shoes under 100' or 'show me jackets'."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
