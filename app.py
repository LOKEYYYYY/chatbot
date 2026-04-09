from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load CSV
df = pd.read_csv("adidas_usa.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
print("COLUMNS:", df.columns)

@app.route("/webhook", methods=["POST", "GET"])
def webhook():
    try:
        req = request.get_json()
        print("REQUEST:", req)

        intent = req["queryResult"]["intent"]["displayName"]
        params = req["queryResult"]["parameters"]

        if intent == "FindProduct":
            color = params.get("color")
            max_price = params.get("price")

            results = df.copy()

            # Try to find correct columns automatically
            color_col = next((c for c in df.columns if "color" in c), None)
            price_col = next((c for c in df.columns if "price" in c), None)
            name_col = next((c for c in df.columns if "name" in c), None)

            if color and color_col:
                results = results[results[color_col].astype(str).str.contains(color, case=False, na=False)]

            if max_price and price_col:
                results[price_col] = pd.to_numeric(results[price_col], errors='coerce')
                results = results[results[price_col] <= float(max_price)]

            if results.empty:
                return jsonify({
                    "fulfillmentText": "No products found 😢"
                })

            product = results.iloc[0]

            return jsonify({
                "fulfillmentText": f"Found {len(results)} products. Example: {product.get(name_col, 'Unknown')} - ${product.get(price_col, 'N/A')}"
            })

        return jsonify({
            "fulfillmentText": "Intent not matched"
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "fulfillmentText": f"Error: {str(e)}"
        })

if __name__ == "__main__":
    app.run(port=3000)
