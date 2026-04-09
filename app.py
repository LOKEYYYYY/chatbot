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
    brand = row.get("brand", "")
    color = row.get("color", "")
    category = row.get("category", "")
    price = row.get("selling_price", None)

    if pd.notna(price):
        price_text = f"${price:.0f}"
    else:
        price_text = "price not listed"

    parts = [str(name)]
    if brand and brand != "nan":
        parts.append(str(brand))
    if category and category != "nan":
        parts.append(str(category))
    if color and color != "nan":
        parts.append(str(color))
    parts.append(price_text)

    return " | ".join(parts)


def search_products(params):
    results = df.copy()

    brand = get_param(params, "brand")
    color = get_param(params, "color")
    products = get_param(params, "products")
    usage = get_param(params, "usage")
    preference = get_param(params, "preference")
    max_price = get_param(params, "max_price")
    price_range = get_param(params, "price_range")

    # Brand filter
    if brand and "brand" in results.columns:
        results = results[results["brand"].str.contains(str(brand), case=False, na=False)]

    # Color filter
    if color and "color" in results.columns:
        results = results[results["color"].str.contains(str(color), case=False, na=False)]

    # Products/category/usage filter
    for term in [products, usage]:
        if term:
            mask = pd.Series(False, index=results.index)
            for col in ["name", "category", "description", "breadcrumbs"]:
                if col in results.columns:
                    mask |= results[col].str.contains(str(term), case=False, na=False)
            results = results[mask]

    # Preference filter / ranking
    if preference:
        pref = str(preference).lower()

        if any(k in pref for k in ["popular", "top", "best seller", "bestseller"]):
            if "reviews_count" in results.columns:
                results = results.sort_values(["reviews_count", "average_rating"], ascending=[False, False])
        elif any(k in pref for k in ["rated", "rating", "best"]):
            if "average_rating" in results.columns:
                results = results.sort_values(["average_rating", "reviews_count"], ascending=[False, False])
        else:
            mask = pd.Series(False, index=results.index)
            for col in ["name", "category", "description", "breadcrumbs"]:
                if col in results.columns:
                    mask |= results[col].str.contains(str(preference), case=False, na=False)
            filtered = results[mask]
            if not filtered.empty:
                results = filtered

    # Price filtering
    min_price, max_price_range = parse_price_range(price_range)

    if max_price not in (None, ""):
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

    return results.reset_index(drop=True)


@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True) or {}

    query_result = req.get("queryResult", {})
    intent_name = query_result.get("intent", {}).get("displayName", "")
    params = query_result.get("parameters", {})
    session_id = req.get("session", "default-session")

    # Welcome
    if intent_name == INTENT_WELCOME:
        return jsonify({
            "fulfillmentText": (
                "Hi! I can help you find Adidas products by brand, color, category, usage, and price. "
                "Try saying: 'black shoes under 100' or 'show me jackets'."
            )
        })

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
        results = search_products(params)

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

        message = "Here are some products:\n" + "\n".join(f"- {line}" for line in lines)

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
