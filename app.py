from flask import Flask, request, jsonify
import pandas as pd
import re
import os

app = Flask(__name__)

# Load dataset once
df = pd.read_csv("products.csv")

session_memory = {}


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
        "high rating": ["rating", "high rating"],
        "cheap": ["cheap", "cheapest", "lowest price", "budget"],
        "expensive": ["expensive", "most expensive", "premium", "high price"]
    }

    for key, values in mapping.items():
        if pref in values:
            return key

    return pref


def detect_product(query_text):
    keyword_map = {
        "electronics": [
            "electronics", "gadgets", "camera", "smartwatch", "monitor",
            "smartphone", "speaker", "tablet", "laptop", "tech", "gaming",
            "gaming console", "headphones", "phone", "air cond"
        ],
        "footwear": [
            "sneakers", "running shoes", "heels", "hiking shoes", "boots",
            "sandals", "flats", "formal shoes", "slippers"
        ],
        "books": [
            "book", "novel", "cookbooks", "non-fiction", "fiction",
            "comics", "textbooks", "magazines", "biographies"
        ],
        "appliances": [
            "home appliances", "kitchen appliances", "blender",
            "washing machine", "dishwasher", "microwave",
            "vacuum cleaner", "refrigerator", "air conditioner",
            "toaster"
        ],
        "clothing": [
            "apparel", "skirt", "socks", "sweater", "jeans",
            "shirt", "t-shirt", "dress", "fashion",
            "clothing", "jacket"
        ]
    }

    query_text = query_text.lower()

    for category, keywords in keyword_map.items():
        for word in keywords:
            if word in query_text:
                return category   # 🔥 FIX: return CATEGORY instead of word

    return ""


def extract_max_price(query_text):
    match = re.search(r"(under|below|less than)\s*(\d+)", query_text)
    if match:
        return float(match.group(2))
    return None


def apply_filters(df, product, price_range, max_price, query_text=""):
    filtered = df.copy()

    if product:
        # 🔥 Map internal category → dataset category
        category_map = {
            "electronics": "Electronics",
            "footwear": "Footwear",
            "books": "Books",
            "appliances": "Appliances",
            "clothing": "Apparel"
        }

        mapped_category = category_map.get(product, product)

        # 🔥 First filter by category
        filtered = filtered[
            filtered['Category'].str.contains(mapped_category, case=False, na=False)
        ]

        # 🔥 Then refine using actual keyword from user query
        for word in query_text.split():
            if len(word) > 3:  # avoid small words like "a", "me"
                filtered = filtered[
                    filtered['Product Name'].str.contains(word, case=False, na=False)
                ] if not filtered.empty else filtered

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
    
    elif preference == "cheap":
        return filtered.sort_values(by='Price', ascending=True)   # 🔥 lowest first

    elif preference == "expensive":
        return filtered.sort_values(by='Price', ascending=False)  # 🔥 highest first

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
        f"   🏷️ {row['Category']}"
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

    intent_name = req['queryResult']['intent']['displayName']
    session_id = req['session']

    # 🔥 LIST ALL CATEGORIES
    if intent_name == "List Categories":
        categories = df['Category'].dropna().unique()

        reply = "🛍️ We have these product categories:\n\n"
        reply += "\n".join([f"• {c}" for c in categories])

        return jsonify({"fulfillmentText": reply})

    # 🔥 HANDLE SHOW MORE FIRST
    if intent_name == "Show More Intent":
        if session_id in session_memory:
            memory = session_memory[session_id]
    
            product = memory.get("product", "")
            preference = memory["preference"]
            price_range = memory["price_range"]
            max_price = memory["max_price"]
            page = memory.get("page", 1) + 1
    
            apply_filters(df, product, price_range, max_price, query_text)

            if filtered.empty:
                return jsonify({"fulfillmentText": "No more results 😢 (filter returned empty)"})
            
            filtered = apply_sorting(filtered, preference)
    
            start = (page - 1) * 3
            end = start + 3
            results = filtered.iloc[start:end]
    
            if results.empty:
                return jsonify({"fulfillmentText": "No more results 😢"})
    
            session_memory[session_id]["page"] = page
    
            reply = format_reply(results)
            return jsonify({"fulfillmentText": reply})
    
        else:
            return jsonify({"fulfillmentText": "Please search for something first 😊"})

    product = params.get('product') or ""
    price_range = params.get('price_range') or ""
    preference = normalize_preference(params.get('preference'))
    max_price = params.get('max_price')

    # 🔥 FORCE override using user text (VERY IMPORTANT)
    if any(word in query_text for word in ["cheap", "cheapest", "lowest price", "budget"]):
        preference = "cheap"

    elif any(word in query_text for word in ["expensive", "most expensive", "highest price", "premium"]):
        preference = "expensive"

    # Smart extraction from text
    detected_product = detect_product(query_text)
    extracted_price = extract_max_price(query_text)

    if detected_product:
        product = detected_product
    elif not product and session_id in session_memory:
        product = session_memory[session_id].get("product", "")

    if extracted_price:
        max_price = extracted_price

    # Apply filtering + sorting
    filtered = apply_filters(df, product, price_range, max_price)
    filtered = apply_sorting(filtered, preference)

    # 🔥 SHOW ALL support
    if "all" in query_text or intent_name == "Show All":
        results = filtered
    else:
        results = filtered.head(3)

    reply = format_reply(results)

    # 🔥 SAVE CONTEXT
    session_memory[session_id] = {
        "product": product,
        "preference": preference,
        "price_range": price_range,
        "max_price": max_price,
        "page": 1
    }

    return jsonify({"fulfillmentText": reply})


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
