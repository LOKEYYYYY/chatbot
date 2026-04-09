from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# 1. Load dataset
df = pd.read_csv("nike_products.csv")


# 2. Smart search function
def smart_search(query, params):
    results = df.copy()

    query = query.lower()

    # Search in ALL useful columns
    results = results[
        results["name"].str.lower().str.contains(query) |
        results["category"].str.lower().str.contains(query) |
        results["description"].str.lower().str.contains(query) |
        results["color"].str.lower().str.contains(query)
    ]

    return results.head(5)


# 3. Filter using entities (optional but useful)
def filter_products(results, params):
    if "category" in params and params["category"]:
        results = results[
            results["category"].str.contains(params["category"], case=False)
        ]

    if "color" in params and params["color"]:
        results = results[
            results["color"].str.contains(params["color"], case=False)
        ]

    return results


# 4. Format response
def format_response(results):
    if results.empty:
        return {"fulfillmentText": "Sorry 😢 I couldn't find anything matching that."}

    reply = "🔥 Here are some great options:\n\n"

    for _, row in results.iterrows():
        reply += f"{row['name']}\n"
        reply += f"💰 RM{row['price']}\n"
        reply += f"🎨 {row['color']}\n\n"

    return {"fulfillmentText": reply}


# 5. Webhook
@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()

    query = req["queryResult"]["queryText"]
    params = req["queryResult"]["parameters"]

    results = smart_search(query, params)
    results = filter_products(results, params)

    response = format_response(results)

    return jsonify(response)


if __name__ == "__main__":
    app.run(port=5000)
