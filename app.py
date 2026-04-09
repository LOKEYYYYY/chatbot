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

def detect_products_from_text(query_text, df):
    if not query_text:
        return []

    text = query_text.lower()
    matched = []

    if "name" not in df.columns:
        return matched

    for name in df["name"].dropna().unique():
        name_lower = str(name).lower()

        # Split into keywords
        words = name_lower.split()

        # If MOST words match → consider it a hit
        match_count = sum(1 for w in words if w in text)

        if match_count >= max(1, len(words) // 2):
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
    if brand and "brand" in results.columns:
        results = results[results["brand"].str.contains(str(brand), case=False, na=False)]

    if color and "color" in results.columns:
        results = results[results["color"].str.contains(str(color), case=False, na=False)]

    for term in [products, usage]:
        if term:
            mask = pd.Series(False, index=results.index)
            for col in ["name", "category", "description"]:
                if col in results.columns:
                    mask |= results[col].str.contains(str(term), case=False, na=False)
            results = results[mask]
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
            "fulfillmentText": "Okay — tell me another color, brand, category, or budget and I’ll search again."
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

        # STEP 3 (you already added this earlier)
        matched_products = detect_products_from_text(query_text, df)

        results = search_products(params)

        # Ensure category relevance from user query
        if "category" in df.columns:
            category_terms = ["shoes", "hoodie", "shorts", "shirt", "jacket"]

            category_mask = pd.Series(False, index=results.index)

            for term in category_terms:
                if term in query_text.lower():
                    category_mask |= results["category"].str.contains(term, case=False, na=False)

        if category_mask.any():
            # Only apply if NO usage specified
            if not get_param(params, "usage"):
                results = results[category_mask]

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

        # =========================
        # ✅ STEP 5 (prefix logic)
        # =========================
        if len(matched_products) > 1:
            message_prefix = "🛒 You selected multiple products:\n"
        elif len(matched_products) == 1:
            message_prefix = "🎯 Product found:\n"
        else:
            message_prefix = ""

        # =========================
        # ✅ STEP 4 (main message)
        # =========================
        if matched_products:
            message = "🎯 I found these exact products:\n"
        else:
            message = "🔥 Top picks for you:\n"

        message += "\n".join(f"- {line}" for line in lines)

        # combine prefix
        message = message_prefix + message

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
