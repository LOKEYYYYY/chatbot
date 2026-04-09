from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("adidas_usa.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert price to number
df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(force=True)

    # Get parameters safely
    params = req.get("queryResult", {}).get("parameters", {})
    color = params.get("color")
    max_price = params.get("price")

    results = df.copy()

    # Filter by color
    if color:
        results = results[results["color"].astype(str).str.contains(color, case=False, na=False)]

    # Filter by price
    if max_price:
        results = results[results["selling_price"] <= float(max_price)]

    # If nothing found
    if results.empty:
        return jsonify({
            "fulfillmentText": "No products found 😢"
        })

    # Take top 3 products
    top = results.head(3)

    response = "Here are some products:\n"

    for _, row in top.iterrows():
        response += f"- {row['name']} (${row['selling_price']})\n"

    return jsonify({
        "fulfillmentText": response
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
