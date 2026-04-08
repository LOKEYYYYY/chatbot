from flask import Flask, request, jsonify
import pandas as pd
import re
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("products.csv")

# Store user session memory
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

    mapping = {
        "popular": ["top", "best selling", "trending"],
        "discount": ["discount", "sale", "deal"],
        "cheap": ["cheap", "cheapest", "budget", "lowest price"],
        "expensive": ["expensive", "premium", "high price"]
    }

    for key, values in mapping.items():
        if pref in values:
            return key

    return pref


def detect_product(query_text):
    query_text = query_text.lower()

    category_map = {
        "electronics": ["laptop", "phone", "tablet", "camera", "headphones"],
        "footwear": ["shoes", "sneakers", "boots"],
        "books": ["book", "novel"],
        "appliances": ["microwave", "fridge", "air conditioner"],
        "clothing": ["shirt", "jeans", "jacket"]
    }

    for category, keywords in category_map.items():
        for word in keywords:
            if word in query_text:
                return category  # RETURN CATEGORY

    return ""


def extract_max_price(query_text):
    match = re.search(r"(under|below|less than)\s*(\d+)", query_text)
    if match:
        return float(match.group(2))
    return None


def apply_filters(df, product, price_range, max_price):
    filtered = df.copy()

    # Filter by category
    if product:
        filtered = filtered[
            filtered['Category'].str.contains(product, case=False, na=False)
        ]

    # Price range
    if price_range == "cheap":
        filtered = filtered[filtered['Price'] <= 700]
    elif price_range == "mid":
        filtered = filtered[(filtered['Price'] > 700) & (filtered['Price'] <= 1400)]
    elif price_range == "expensive":
        filtered = filtered[filtered['Price'] > 1400]

    # Max price
    if max_price:
        filtered = filtered[filtered['Price'] <= max_price]

    return filtered


def apply_sorting(filtered, preference):
    if preference == "popular":
        return filtered.sort_values(by='Popularity Index', ascending=False)

    elif preference == "discount":
        return filtered.sort_values(by='Discount', ascending=False)

    elif preference == "cheap":
        return filtered.sort_values(by='Price', ascending=True)

    elif preference == "expensive":
        return filtered.sort_values(by='Price', ascending=False)

    return filtered


def format_reply(results):
    if results.empty:
        return "😢 No products found. Try another search."

    reply = "✨ Recommended Products ✨\n\n"

    for i, (_, row) in enumerate(results.iterrows(), start=1):
        reply += (
            f"{i}) {row['Product Name']}\n"
            f"   💰 RM{row['Price']:.2f}\n"
            f"   ⭐ {row['Popularity Index']} | 🔻 {row['Discount']}%\n"
            f"   🏷️ {row['Category']}\n\n"
        )

    return reply.strip()


# -------------------------------
# MAIN WEBHOOK
# -------------------------------

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()

    query_text = req['queryResult']['queryText'].lower()
    params = req['queryResult']['parameters']
    intent_name = req['queryResult']['intent']['displayName']
    session_id = req['session']

    # -------------------------------
    # 1. LIST CATEGORIES
    # -------------------------------
    if intent_name == "List Categories":
        categories = sorted(df['Category'].dropna().unique())
        reply = "🛍️ Available categories:\n\n"
        reply += "\n".join([f"• {c}" for c in categories])
        return jsonify({"fulfillmentText": reply})


    # -------------------------------
    # 2. SHOW MORE
    # -------------------------------
    if intent_name == "Show More":
        if session_id not in session_memory:
            return jsonify({"fulfillmentText": "Please search first 😊"})

        memory = session_memory[session_id]

        filtered = apply_filters(df, memory["product"], memory["price_range"], memory["max_price"])
        filtered = apply_sorting(filtered, memory["preference"])

        page = memory.get("page", 1) + 1
        start = (page - 1) * 3
        end = start + 3

        results = filtered.iloc[start:end]

        if results.empty:
            return jsonify({"fulfillmentText": "No more results 😢"})

        session_memory[session_id]["page"] = page

        return jsonify({"fulfillmentText": format_reply(results)})


    # -------------------------------
    # 3. NORMAL SEARCH
    # -------------------------------
    product = params.get('product') or ""
    price_range = params.get('price_range') or ""
    preference = normalize_preference(params.get('preference'))
    max_price = params.get('max_price')

    # Detect preference manually
    if any(word in query_text for word in ["cheap", "cheapest", "budget"]):
        preference = "cheap"
    elif any(word in query_text for word in ["expensive", "premium"]):
        preference = "expensive"

    # Detect product
    detected_product = detect_product(query_text)
    if detected_product:
        product = detected_product
    elif session_id in session_memory:
        product = session_memory[session_id]["product"]

    # Extract price
    extracted_price = extract_max_price(query_text)
    if extracted_price:
        max_price = extracted_price

    # Apply logic
    filtered = apply_filters(df, product, price_range, max_price)
    filtered = apply_sorting(filtered, preference)

    # -------------------------------
    # 4. SHOW ALL
    # -------------------------------
    if "all" in query_text or intent_name == "Show All":
        results = filtered
    else:
        results = filtered.head(3)

    # Save memory
    session_memory[session_id] = {
        "product": product,
        "preference": preference,
        "price_range": price_range,
        "max_price": max_price,
        "page": 1
    }

    return jsonify({"fulfillmentText": format_reply(results)})


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
