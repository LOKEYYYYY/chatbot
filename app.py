from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# ==============================
# LOAD DATASET
# ==============================
def load_dataset():
    df = pd.read_csv("adidas_usa.csv")

    # Normalize text columns
    for col in ['name', 'category', 'color', 'description']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    # Ensure numeric price
    if "selling_price" in df.columns:
        df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")

    return df

df = load_dataset()


# ==============================
# EXTRACT INTENT + PARAMETERS
# ==============================
def extract_intent_data(req):
    intent = req['queryResult']['intent']['displayName']
    params = req['queryResult'].get('parameters', {})
    user_text = req['queryResult'].get("queryText", "").lower()

    return intent, params, user_text


# ==============================
# SMART KEYWORD DETECTION
# ==============================
def extract_keywords(text):
    usage_map = {
        "running": ["running", "jogging"],
        "hiking": ["hiking", "trekking", "outdoor"],
        "casual": ["casual", "daily", "everyday"],
        "training": ["gym", "training", "workout"]
    }

    keywords = []

    for key, values in usage_map.items():
        if any(v in text for v in values):
            keywords.append(key)

    return keywords


# ==============================
# SCORING SYSTEM (CORE AI LOGIC)
# ==============================
def compute_score(row, params, user_text):
    score = 0

    # CATEGORY MATCH
    if params.get("category"):
        if params["category"].lower() in str(row.get("category", "")):
            score += 3

    # COLOR MATCH
    if params.get("color"):
        if params["color"].lower() in str(row.get("color", "")):
            score += 2

    # PRICE MATCH (closer = better)
    if params.get("price"):
        try:
            target_price = float(params["price"])
            actual_price = row.get("selling_price", 0)

            if pd.notna(actual_price):
                diff = abs(actual_price - target_price)
                score += max(0, 3 - diff / 100)
        except:
            pass

    # DESCRIPTION / KEYWORD MATCH
    keywords = extract_keywords(user_text)

    description = str(row.get("description", ""))

    for k in keywords:
        if k in description:
            score += 4

    # NAME MATCH (bonus)
    if any(word in str(row.get("name", "")) for word in user_text.split()):
        score += 2

    return score


# ==============================
# FILTER + RANK PRODUCTS
# ==============================
def filter_and_rank_products(df, params, user_text):
    working_df = df.copy()

    # Compute score
    working_df["score"] = working_df.apply(
        lambda row: compute_score(row, params, user_text), axis=1
    )

    # Keep relevant results
    working_df = working_df[working_df["score"] > 1]

    # Sort best first
    working_df = working_df.sort_values(by="score", ascending=False)

    # Fallback if empty
    if working_df.empty:
        working_df = df.sample(min(5, len(df)))

    return working_df.head(5)


# ==============================
# FORMAT RICH RESPONSE (CARDS)
# ==============================
def format_rich_response(products):
    messages = []

    for _, row in products.iterrows():
        name = str(row.get("name", "Product")).title()
        price = row.get("selling_price", "N/A")
        image = row.get("image", "")  # MUST exist in dataset
        link = row.get("source", "")  # product URL

        card = {
            "card": {
                "title": name,
                "subtitle": f"💰 RM{price}",
                "imageUri": image,
                "buttons": [
                    {
                        "text": "View Product",
                        "postback": link
                    }
                ]
            }
        }

        messages.append(card)

    return messages


# ==============================
# TEXT RESPONSE (FALLBACK)
# ==============================
def format_text_response(products):
    if products.empty:
        return "😕 Sorry, I couldn’t find anything."

    reply = "🔥 Here are some recommendations:\n\n"

    for _, row in products.iterrows():
        reply += f"👟 {row.get('name', '').title()}\n"
        reply += f"💰 RM{row.get('selling_price', 'N/A')}\n\n"

    return reply


# ==============================
# WEBHOOK MAIN
# ==============================
@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()

    intent, params, user_text = extract_intent_data(req)

    # ==========================
    # PRODUCT SEARCH INTENT
    # ==========================
    if intent == "Product Search":
        products = filter_and_rank_products(df, params, user_text)

        return jsonify({
            "fulfillmentMessages": format_rich_response(products)
        })

    # ==========================
    # LIST CATEGORIES
    # ==========================
    elif intent == "List Categories":
        categories = df['category'].dropna().unique()

        reply = "🛍️ Available categories:\n\n"
        reply += "\n".join([f"• {c.title()}" for c in categories])

        return jsonify({
            "fulfillmentText": reply
        })

    # ==========================
    # FALLBACK
    # ==========================
    elif intent == "Default Fallback Intent":
        return jsonify({
            "fulfillmentText": "🤖 Try something like: 'running shoes under RM300'"
        })

    # ==========================
    # DEFAULT
    # ==========================
    else:
        return jsonify({
            "fulfillmentText": "🤖 How can I help you today?"
        })


# ==============================
# HEALTH CHECK
# ==============================
from flask import send_file

@app.route("/", methods=["GET"])
def home():
    return send_file("index.html")


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
