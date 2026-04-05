from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("products.csv")

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    params = req['queryResult']['parameters']

    product = params.get('product') or ""
    price_range = params.get('price_range') or ""
    preference = params.get('preference') or ""
    max_price = params.get('max_price')

    # Fix list issue (dialogflow use list py use str)
    if isinstance(preference, list):
        preference = preference[0] if preference else None

    # Normalize preference 
    if preference:
        preference = preference.lower().strip()

        if preference in ["top selling", "best selling", "trending", "top popularity"]:
            preference = "popular"

        elif preference in ["sale", "deals", "discounted"]:
            preference = "discount"

        elif preference in ["top", "highest"]:
            preference = "best"

    # fallback if no category
    if not preference:
        preference = "popular"

    filtered = df.copy()

    #Get keyword before filtering
    query_text = req['queryResult']['queryText'].lower()

    keywords = [
        # Electronics (your original + expanded)
        "electronics", "gadgets", "camera", "smartwatch", "monitor",
        "smartphone", "speaker", "tablet", "laptop", "tech", "gaming",
        "watches", "watch", "gaming console", "headphones", "phones",
        "phone", "laptops", "air cond",

        # Footwear
        "sneakers", "running shoes", "heels", "hiking shoes", "boots",
        "sandals", "flats", "formal shoes", "slippers", "footwears"

        # Books
        "book", "novel", "cookbooks", "non-fiction", "fiction", "comics",
        "textbooks", "magazines", "graphic novels", "biographies",

        # Home appliances
        "home appliances", "kitchen appliances", "blender", "washing machine",
        "dishwasher", "microwave", "vacuum cleaner", "refrigerator",
        "air conditioner", "toaster", "home products", "books"

        # Apparel
        "apparel", "skirt", "socks", "sweater", "jeans", "shirt", "t-shirt",
        "dress", "fashion", "clothing", "jackets", "jacket"
    ]

    detected_product = None
    for word in keywords:
        if word in query_text:
            detected_product = word
            break

    # override product if keyword found
    if detected_product:
        product = detected_product

    # Filter by category
    filtered = filtered[
        (filtered['Category'].str.contains(product, case=False, na=False)) |
        (filtered['Product Name'].str.contains(product, case=False, na=False))
    ]

    # Price filter
    if price_range == "cheap":
        filtered = filtered[filtered['Price'] <= 700]
    elif price_range == "mid range":
        filtered = filtered[(filtered['Price'] > 700) & (filtered['Price'] <= 1400)]
    elif price_range == "expensive":
        filtered = filtered[filtered['Price'] > 1400]

    results = filtered.head(3)
    
    # Max price 
    if max_price:
        filtered = filtered[filtered['Price'] <= max_price]

    # Preference sorting
    if preference == "popular":
        filtered = filtered.sort_values(by='Popularity Index', ascending=False)

    elif preference == "discount":
        filtered = filtered.sort_values(by='Discount', ascending=False)

    elif preference == "best":
        filtered = filtered.sort_values(by=['Popularity Index', 'Discount'], ascending=False)

    elif preference == "high rating":
        filtered = filtered.sort_values(by='Popularity Index', ascending=False)

    results = filtered.head(3)

    if results.empty:
        reply = "Sorry, no products found."
    else:
        lines = [
            f"{i}) {row['Product Name']} | RM{row['Price']:.2f} | ⭐ {row['Popularity Index']}"
            for i, (_, row) in enumerate(results.iterrows(), start=" ")
        ]

        reply = "✨ Recommended Products ✨\n\n" + "\n".join(lines)

    return jsonify({"fulfillmentText": reply})

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
