from flask import Flask, request, jsonify
import pandas as pd
import re
import os

app = Flask(__name__)

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("adidas_usa.csv")
df = df[df['selling_price'].notna()].copy()

df['selling_price']  = df['selling_price'].astype(float)
df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['reviews_count']  = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0).astype(int)
df['availability']   = df['availability'].fillna('Unknown')
df['color']          = df['color'].fillna('')
df['category']       = df['category'].fillna('')
df['breadcrumbs']    = df['breadcrumbs'].fillna('')
df['description']    = df['description'].fillna('')

session_memory = {}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def normalize_preference(pref):
    if isinstance(pref, list):
        pref = pref[0] if pref else None
    if not pref:
        return "popular"

    pref = pref.lower().strip()

    if pref in ["cheap", "budget", "lowest"]:
        return "cheap"
    if pref in ["expensive", "premium"]:
        return "expensive"
    if pref in ["discount", "sale"]:
        return "discount"

    return "popular"


def detect_product(query):
    query = query.lower()

    if any(w in query for w in ["shoe", "sneaker", "boot", "trainer"]):
        return "Shoes"
    if any(w in query for w in ["shirt", "hoodie", "jacket", "pants"]):
        return "Clothing"
    if any(w in query for w in ["bag", "cap", "sock", "hat"]):
        return "Accessories"

    return ""


def detect_color(query):
    colors = ["black","white","blue","red","green","grey","gray","pink","yellow"]
    for c in colors:
        if c in query:
            return "Grey" if c == "gray" else c.capitalize()
    return ""


def detect_usage(query):
    usage_map = {
        "Men": ["men", "mens"],
        "Women": ["women", "womens"],
        "Kids": ["kids", "children"],
        "Running": ["running"],
        "Training": ["gym", "training"],
        "Soccer": ["football", "soccer"]
    }

    for u, words in usage_map.items():
        if any(w in query for w in words):
            return u
    return ""


def extract_max_price(query):
    match = re.search(r"(under|below|less than)\s*(\d+)", query)
    return float(match.group(2)) if match else None


# 🔥 CLEAN FILTER FUNCTION
def apply_filters(df, product, price_range, max_price, query, color, usage):
    filtered = df.copy()

    # In stock only
    filtered = filtered[filtered['availability'] == "InStock"]

    # Category
    if product:
        temp = filtered[filtered['category'].str.contains(product, case=False, na=False)]
        if not temp.empty:
            filtered = temp

    # Color
    if color:
        temp = filtered[filtered['color'].str.contains(color, case=False, na=False)]
        if not temp.empty:
            filtered = temp

    # Usage (breadcrumbs + description)
    if usage:
        temp = filtered[filtered['breadcrumbs'].str.contains(usage, case=False, na=False)]
        if temp.empty:
            temp = filtered[filtered['description'].str.contains(usage, case=False, na=False)]
        if not temp.empty:
            filtered = temp

    # Price
    if price_range == "cheap":
        filtered = filtered[filtered['selling_price'] <= 35]
    elif price_range == "expensive":
        filtered = filtered[filtered['selling_price'] > 80]

    if max_price:
        filtered = filtered[filtered['selling_price'] <= max_price]

    # 🔥 SAFE keyword search (name + description)
    stopwords = {"i","need","want","show","find","for","with","under","adidas"}

    for word in query.split():
        if len(word) > 3 and word not in stopwords:
            temp = filtered[
                filtered['name'].str.contains(word, case=False, na=False) |
                filtered['description'].str.contains(word, case=False, na=False)
            ]
            if not temp.empty:
                filtered = temp

    return filtered


def apply_sorting(df, preference):
    if preference == "cheap":
        return df.sort_values(by='selling_price')
    if preference == "expensive":
        return df.sort_values(by='selling_price', ascending=False)
    return df.sort_values(by='reviews_count', ascending=False)


def format_reply(results):
    if results.empty:
        return "😢 No products found. Try another search!"

    lines = []
    for i, (_, row) in enumerate(results.iterrows(), 1):
        lines.append(
            f"{i}) {row['name']}\n"
            f"   💰 ${row['selling_price']:.2f}\n"
            f"   ⭐ {row['average_rating']} ({row['reviews_count']})\n"
            f"   🏷️ {row['category']}"
        )

    return "👟 Products:\n\n" + "\n\n".join(lines)


# -------------------------------
# WEBHOOK
# -------------------------------

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()

    params = req['queryResult']['parameters']
    query  = req['queryResult']['queryText'].lower()
    session_id = req['session']

    product     = params.get('product') or ""
    price_range = params.get('price_range') or ""
    preference  = normalize_preference(params.get('preference'))
    max_price   = params.get('max_price')
    color       = params.get('color') or ""
    usage       = params.get('usage') or ""

    # Smart detection
    product = detect_product(query) or product
    color   = detect_color(query) or color
    usage   = detect_usage(query) or usage
    max_price = extract_max_price(query) or max_price

    filtered = apply_filters(df, product, price_range, max_price, query, color, usage)
    filtered = apply_sorting(filtered, preference)

    results = filtered.head(3)

    # Save memory
    session_memory[session_id] = {
        "product": product,
        "query": query,
        "page": 1
    }

    return jsonify({"fulfillmentText": format_reply(results)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
