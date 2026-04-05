from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("products.csv")

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    params = req['queryResult']['parameters']

    product = params.get('product')
    price_range = params.get('price_range')
    preference = params.get('preference')
    max_price = params.get('max_price')

    # Normalize preference
    if preference in ["top selling", "best selling", "trending"]:
        preference = "popular"

    filtered = df.copy()

    # Filter by category
    if product:
        filtered = filtered[filtered['Category'].str.contains(product, case=False)]

    # Price filter
    if price_range == "cheap":
        filtered = filtered[filtered['Price'] < 1000]
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
