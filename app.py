from flask import Flask, request, jsonify
import pandas as pd
import re
import os

app = Flask(__name__)

# Load dataset once
df = pd.read_csv("products.csv")


# -------------------------------
# Helper Functions
# -------------------------------

def normalize_preference(pref):
    if isinstance(pref, list):
        pref = pref[0] if pref else None

    if not pref:
        return "popular"

    pref = pref.lower().strip()

    mapping = {
        "popular": ["top selling", "best selling", "trending", "top popularity"],
        "discount": ["sale", "deals", "discounted"],
        "best": ["top", "highest"],
        "high rating": ["rating", "high rating"]
    }

    for key, values in mapping.items():
        if pref in values:
            return key

    return pref


def detect_product(query_text):
    keyword_map = {
        "laptop": ["laptop", "notebook", "macbook"],
        "phone": ["phone", "smartphone", "iphone"],
        "camera": ["camera"],
        "headphones": ["headphones", "earphones"],
        "shoes": ["sneakers", "boots", "heels", "sandals"],
        "books": ["book", "novel", "comics"],
        "appliances": ["microwave", "blender", "washing machine"],
        "clothing": ["shirt", "jeans", "dress", "jacket"]
    }

    for key, words in keyword_map.items():
        if any(word in query_text for word in words):
            return key

    return ""


def extract_max_price(query_text):
    match = re.search(r"(under|below|less than)\s*(\d+)", query_text)
    if match:
        return float(match.group(2))
    return None


def apply_filters(df, product, price_range, max_price):
    filtered = df.copy()

    if product:
        filtered = filtered[
            (filtered['Category'].str.contains(product, case=False, na=False)) |
            (filtered['Product Name'].str.contains(product, case=False, na=False))
        ]

    if price_range == "cheap":
        filtered = filtered[filtered['Price'] <= 700]
    elif price_range == "mid range":
        filtered = filtered[(filtered['Price'] > 700) & (filtered['Price'] <= 1400)]
    elif price_range == "expensive":
        filtered = filtered[filtered['Price'] > 1400]

    if max_price:
        filtered = filtered[filtered['Price'] <= max_price]

    return filtered


def apply_sorting(filtered, preference):
    if preference == "popular":
        return filtered.sort_values(by='Popularity Index', ascending=False)

    elif preference == "discount":
        return filtered.sort_values(by='Discount', ascending=False)

    elif preference == "best":
        return filtered.sort_values(by=['Popularity Index', 'Discount'], ascending=False)

    elif preference == "high rating":
        return filtered.sort_values(by='Popularity Index', ascending=False)

    return filtered


def format_reply(results):
    if results.empty:
        suggestions = df['Category'].dropna().unique()[:3]
        reply = "😢 No exact match found.\n\nTry searching for:\n"
        reply += "\n".join([f"• {s}" for s in suggestions])
        return reply

    lines = [
        f"{i}) {row['Product Name']}\n"
        f"   💰 RM{row['Price']:.2f}\n"
        f"   ⭐ {row['Popularity Index']} | 🔻 {row['Discount']}%\n"
        f"   🏷 {row['Category']}"
        for i, (_, row) in enumerate(results.iterrows(), start=1)
    ]

    return "✨ Recommended Products ✨\n\n" + "\n\n".join(lines)


# -------------------------------
# Main Webhook
# -------------------------------

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    params = req['queryResult']['parameters']
    query_text = req['queryResult']['queryText'].lower()

    product = params.get('product') or ""
    price_range = params.get('price_range') or ""
    preference = normalize_preference(params.get('preference'))
    max_price = params.get('max_price')

    # Smart extraction from text
    detected_product = detect_product(query_text)
    extracted_price = extract_max_price(query_text)

    if detected_product:
        product = detected_product

    if extracted_price:
        max_price = extracted_price

    # Apply filtering + sorting
    filtered = apply_filters(df, product, price_range, max_price)
    filtered = apply_sorting(filtered, preference)

    results = filtered.head(3)

    reply = format_reply(results)

    return jsonify({"fulfillmentText": reply})


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
