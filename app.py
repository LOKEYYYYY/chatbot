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

    # Normalize preference (FIXED VERSION)
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

    keywords = ["laptop", "phone", "camera", "headphones", "watch", "tablet"]

    detected_product = None
    for word in keywords:
        if word in query_text:
            detected_product = word
            break

    # override product if keyword found
    if detected_product:
        product = detected_product

    # Filter by category
    if product:
        filtered = filtered[
            filtered['Product Name'].str.contains(product, case=False, na=False)
        ]

    # Price filter
    if price_range == "cheap":
        filtered = filtered[filtered['Price'] <= 1000]
    elif price_range == "mid range":
        filtered = filtered[(filtered['Price'] > 1000) & (filtered['Price'] <= 3000)]
    elif price_range == "expensive":
        filtered = filtered[filtered['Price'] > 3000]

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
        reply = "Here are some recommendations:\n"
        for _, row in results.iterrows():
            reply += f"{row['Product Name']} - RM{row['Price']} ⭐{row['Popularity Index']}\n"

    return jsonify({"fulfillmentText": reply})

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
