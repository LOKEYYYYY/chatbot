from flask import Flask, request, jsonify
import pandas as pd
import re
import os

app = Flask(__name__)

# Load dataset once
df = pd.read_csv("adidas_usa.csv")

# Clean up dataset
df = df[df['selling_price'].notna()]
df['selling_price'] = df['selling_price'].astype(float)
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0).astype(int)
df['availability'] = df['availability'].fillna('Unknown')
df['color'] = df['color'].fillna('').str.strip()
df['category'] = df['category'].fillna('').str.strip()
df['breadcrumbs'] = df['breadcrumbs'].fillna('').str.strip()

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
        "popular":      ["top selling", "best selling", "trending", "top popularity", "most reviewed"],
        "discount":     ["sale", "deals", "discounted", "on sale"],
        "best":         ["top", "highest", "best rated", "best value"],
        "high rating":  ["rating", "high rating", "top rated"],
        "cheap":        ["cheap", "cheapest", "lowest price", "budget", "affordable"],
        "expensive":    ["expensive", "most expensive", "premium", "high price", "luxury"]
    }

    for key, values in mapping.items():
        if pref in values or pref == key:
            return key

    return pref


def detect_category(query_text):
    """Detect product category from query text."""
    keyword_map = {
        "Shoes": [
            "shoes", "sneakers", "boots", "sandals", "running shoes",
            "bike shoes", "cleats", "slip-on", "loafers", "trainers",
            "footwear", "kicks"
        ],
        "Clothing": [
            "clothing", "shirt", "t-shirt", "jacket", "pants", "shorts",
            "jersey", "dress", "hoodie", "sweater", "tracksuit",
            "leggings", "top", "apparel", "outfit", "clothes"
        ],
        "Accessories": [
            "accessories", "bag", "backpack", "cap", "hat", "socks",
            "gloves", "belt", "wallet", "sunglasses", "watch",
            "headband", "wristband"
        ]
    }

    query_text = query_text.lower()
    for category, keywords in keyword_map.items():
        for word in keywords:
            if word in query_text:
                return category
    return ""


def detect_color(query_text):
    """Detect color preference from query text."""
    colors = [
        "black", "grey", "gray", "white", "blue", "purple", "pink",
        "green", "yellow", "red", "multicolor", "gold", "burgundy", "beige"
    ]
    query_lower = query_text.lower()
    for color in colors:
        if color in query_lower:
            # Normalize "gray" → "Grey" to match dataset
            if color == "gray":
                return "Grey"
            return color.capitalize()
    return ""


def detect_usage(query_text):
    """Detect usage/sub-category from breadcrumbs context."""
    usage_map = {
        "Men":        ["men", "men's", "mens", "male", "guy", "guys"],
        "Women":      ["women", "women's", "womens", "female", "lady", "ladies"],
        "Kids":       ["kids", "kid", "children", "child", "boys", "girls", "junior"],
        "Soccer":     ["soccer", "football"],
        "Training":   ["training", "gym", "workout", "fitness"],
        "Originals":  ["originals", "lifestyle", "casual", "retro", "classic"],
        "Essentials": ["essentials", "basics", "everyday"],
        "Swim":       ["swim", "swimming", "pool", "beach"],
        "Five Ten":   ["five ten", "510", "mountain bike", "cycling", "bike"]
    }

    query_lower = query_text.lower()
    for usage, keywords in usage_map.items():
        for word in keywords:
            if word in query_lower:
                return usage
    return ""


def extract_max_price(query_text):
    """Extract explicit max price from query."""
    match = re.search(r"(under|below|less than)\s*\$?\s*(\d+)", query_text)
    if match:
        return float(match.group(2))
    return None


def detect_relative_price_shift(query_text):
    """Detect if user wants relatively higher or lower priced items."""
    query_text = query_text.lower()

    higher_phrases = [
        "more expensive", "higher price", "pricier", "something more expensive",
        "a bit more expensive", "slightly expensive", "higher end", "cost more"
    ]
    lower_phrases = [
        "cheaper", "less expensive", "lower price", "something cheaper",
        "a bit cheaper", "slightly cheaper", "lower end", "cost less"
    ]

    for phrase in higher_phrases:
        if phrase in query_text:
            return "higher"
    for phrase in lower_phrases:
        if phrase in query_text:
            return "lower"
    return None


def get_price_anchor(session_id):
    if session_id in session_memory:
        return session_memory[session_id].get("last_max_shown_price", None)
    return None


def apply_filters(df, category, color, usage, price_range, max_price, query_text=""):
    """Apply all filters to the dataframe."""
    filtered = df.copy()

    # Filter out-of-stock unless user explicitly asks for all
    if "out of stock" not in query_text.lower() and "unavailable" not in query_text.lower():
        in_stock = filtered[filtered['availability'] == 'InStock']
        if not in_stock.empty:
            filtered = in_stock

    # Category filter
    if category:
        cat_filtered = filtered[filtered['category'].str.contains(category, case=False, na=False)]
        if not cat_filtered.empty:
            filtered = cat_filtered

    # Color filter
    if color:
        color_filtered = filtered[filtered['color'].str.contains(color, case=False, na=False)]
        if not color_filtered.empty:
            filtered = color_filtered

    # Usage/breadcrumb filter (Men, Women, Kids, Soccer, etc.)
    if usage:
        usage_filtered = filtered[filtered['breadcrumbs'].str.contains(usage, case=False, na=False)]
        if not usage_filtered.empty:
            filtered = usage_filtered

    # Price range filter
    if price_range == "cheap":
        filtered = filtered[filtered['selling_price'] <= 40]
    elif price_range == "mid range":
        filtered = filtered[(filtered['selling_price'] > 40) & (filtered['selling_price'] <= 100)]
    elif price_range == "expensive":
        filtered = filtered[filtered['selling_price'] > 100]

    # Absolute max price filter
    if max_price:
        filtered = filtered[filtered['selling_price'] <= max_price]

    # Keyword refinement on product name
    for word in query_text.split():
        if len(word) > 3 and word.lower() not in [
            "show", "find", "want", "need", "give", "tell", "look", "what",
            "with", "that", "have", "some", "most", "best", "under", "below"
        ]:
            refined = filtered[filtered['name'].str.contains(word, case=False, na=False)]
            if not refined.empty:
                filtered = refined

    return filtered


def apply_relative_price_filter(filtered, direction, price_anchor):
    if price_anchor is None:
        return filtered

    if direction == "higher":
        shifted = filtered[filtered['selling_price'] > price_anchor]
    elif direction == "lower":
        shifted = filtered[filtered['selling_price'] < price_anchor]
    else:
        return filtered

    return shifted if not shifted.empty else filtered


def apply_sorting(filtered, preference):
    if preference == "popular":
        return filtered.sort_values(by='reviews_count', ascending=False)
    elif preference == "discount":
        # Products with original_price available (discounted)
        has_discount = filtered[filtered['original_price'].notna()]
        if not has_discount.empty:
            return has_discount.sort_values(by='selling_price', ascending=True)
        return filtered.sort_values(by='selling_price', ascending=True)
    elif preference in ("best", "high rating"):
        return filtered.sort_values(by='average_rating', ascending=False)
    elif preference == "cheap":
        return filtered.sort_values(by='selling_price', ascending=True)
    elif preference == "expensive":
        return filtered.sort_values(by='selling_price', ascending=False)
    return filtered.sort_values(by='reviews_count', ascending=False)


def format_reply(results, category=""):
    if results.empty:
        return format_no_results_reply(category)

    lines = []
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        price_line = f"💲 ${row['selling_price']:.2f}"
        if pd.notna(row.get('original_price')) and row['original_price'] > row['selling_price']:
            price_line += f" ~~${row['original_price']:.2f}~~ 🔖 On Sale!"

        rating_line = ""
        if pd.notna(row['average_rating']):
            rating_line = f"\n   ⭐ {row['average_rating']:.1f}/5.0 ({int(row['reviews_count'])} reviews)"

        color_line = f"🎨 {row['color']}" if row['color'] else ""
        avail = "✅ In Stock" if row['availability'] == "InStock" else "❌ Out of Stock"

        entry = (
            f"{i}) {row['name']}\n"
            f"   {price_line}\n"
            f"   {avail}"
        )
        if rating_line:
            entry += rating_line
        if color_line:
            entry += f"\n   {color_line}"
        entry += f"\n   🏷️ {row['category']} | {row['breadcrumbs']}"

        lines.append(entry)

    return "👟 Adidas Products for You ✨\n\n" + "\n\n".join(lines)


def format_no_results_reply(category=""):
    related_map = {
        "Shoes":       ["Clothing", "Accessories"],
        "Clothing":    ["Shoes", "Accessories"],
        "Accessories": ["Shoes", "Clothing"]
    }
    related = related_map.get(category, ["Shoes", "Clothing", "Accessories"])
    reply = f"😢 No results found"
    if category:
        reply += f" for *{category}*"
    reply += ".\n\nYou might also like:\n"
    reply += "\n".join([f"• {r}" for r in related])
    reply += "\n\nOr try adjusting your color, price, or usage filters 🔍"
    return reply


def save_last_shown_price(session_id, results):
    if not results.empty and session_id in session_memory:
        session_memory[session_id]["last_max_shown_price"] = float(results['selling_price'].max())


def has_more_results(session_id, filtered_total):
    if session_id not in session_memory:
        return False
    page = session_memory[session_id].get("page", 1)
    return (page * 3) < filtered_total


def append_more_prompt(reply, session_id, filtered_total):
    if has_more_results(session_id, filtered_total):
        reply += "\n\n👀 Want to see more options? Just say 'show more'!"
    else:
        reply += "\n\n✅ That's all the results for this search."
    return reply


def detect_yes_intent(query_text):
    query_text = query_text.lower().strip()
    yes_phrases = [
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "show more",
        "more", "next", "continue", "go on", "show me more", "see more",
        "please", "yes please", "of course", "why not"
    ]
    return any(phrase == query_text or query_text.startswith(phrase) for phrase in yes_phrases)


def detect_no_intent(query_text):
    query_text = query_text.lower().strip()
    no_phrases = [
        "no", "nope", "nah", "no thanks", "no thank you", "that's enough",
        "stop", "done", "enough", "i'm good", "im good", "no more"
    ]
    return any(phrase == query_text or query_text.startswith(phrase) for phrase in no_phrases)


def get_summary_label(category, color, usage):
    parts = []
    if usage:
        parts.append(usage)
    if color:
        parts.append(color)
    if category:
        parts.append(category)
    return " ".join(parts) if parts else "products"


# -------------------------------
# Main Webhook
# -------------------------------

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    params = req['queryResult']['parameters']
    query_text = req['queryResult']['queryText']
    query_lower = query_text.lower()

    intent_name = req['queryResult']['intent']['displayName']
    session_id = req['session']

    # --------------------------------------------------
    # LIST ALL CATEGORIES
    # --------------------------------------------------
    if intent_name == "List Categories":
        categories = df['category'].dropna().unique()
        reply = "🛍️ Adidas USA carries these product categories:\n\n"
        reply += "\n".join([f"• {c}" for c in categories if c])
        reply += "\n\nYou can also filter by gender (Men, Women, Kids), sport (Soccer, Training, Swimming), or style (Originals, Essentials)!"
        return jsonify({"fulfillmentText": reply})

    # --------------------------------------------------
    # NEGATIVE INTENT — user says no to "show more?"
    # --------------------------------------------------
    if intent_name == "Negative Intent" or detect_no_intent(query_lower):
        if session_id in session_memory and session_memory[session_id].get("awaiting_more_confirm"):
            session_memory[session_id]["awaiting_more_confirm"] = False
            mem = session_memory[session_id]
            label = get_summary_label(
                mem.get("category", ""),
                mem.get("color", ""),
                mem.get("usage", "")
            )
            return jsonify({
                "fulfillmentText": f"No problem! Hope you found your perfect Adidas {label} 😊\nFeel free to search again anytime!"
            })

    # --------------------------------------------------
    # SHOW MORE INTENT
    # --------------------------------------------------
    if intent_name == "Show More" or (
        session_id in session_memory
        and session_memory[session_id].get("awaiting_more_confirm")
        and detect_yes_intent(query_lower)
    ):
        if session_id in session_memory:
            memory = session_memory[session_id]

            category   = memory.get("category", "")
            color      = memory.get("color", "")
            usage      = memory.get("usage", "")
            preference = memory.get("preference", "popular")
            price_range = memory.get("price_range", "")
            max_price  = memory.get("max_price", None)
            saved_query = memory.get("query_text", "")
            page = memory.get("page", 1) + 1

            filtered = apply_filters(df, category, color, usage, price_range, max_price, saved_query)

            if filtered.empty:
                return jsonify({"fulfillmentText": "😢 No more results found. Try a different search!"})

            filtered = apply_sorting(filtered, preference)
            start = (page - 1) * 3
            results = filtered.iloc[start:start + 3]

            if results.empty:
                return jsonify({"fulfillmentText": "😢 No more results.\n\nTry a different search to discover more Adidas products! 🔍"})

            session_memory[session_id]["page"] = page
            session_memory[session_id]["awaiting_more_confirm"] = False
            save_last_shown_price(session_id, results)

            reply = format_reply(results, category)
            reply = append_more_prompt(reply, session_id, len(filtered))
            session_memory[session_id]["awaiting_more_confirm"] = has_more_results(session_id, len(filtered))

            return jsonify({"fulfillmentText": reply})

        else:
            return jsonify({"fulfillmentText": "Please search for something first 😊"})

    # --------------------------------------------------
    # PRODUCT SEARCH (main intent)
    # --------------------------------------------------

    # --- Extract parameters from Dialogflow entities ---
    category    = params.get('products') or ""       # @products entity
    color       = params.get('color') or ""           # @color entity
    usage       = params.get('usage') or ""           # @usage entity
    brand       = params.get('brand') or ""           # @brand entity (always adidas here)
    price_range = params.get('price_range') or ""     # @price_range entity
    max_price   = params.get('max_price') or None     # @max_price entity
    preference  = normalize_preference(params.get('preference'))  # @preference entity

    # --- Normalize list params from Dialogflow ---
    if isinstance(category, list):
        category = category[0] if category else ""
    if isinstance(color, list):
        color = color[0] if color else ""
    if isinstance(usage, list):
        usage = usage[0] if usage else ""
    if isinstance(price_range, list):
        price_range = price_range[0] if price_range else ""

    # --- Force override preference from query text ---
    if any(w in query_lower for w in ["cheap", "cheapest", "lowest price", "budget", "affordable"]):
        preference = "cheap"
    elif any(w in query_lower for w in ["expensive", "most expensive", "premium", "luxury"]):
        preference = "expensive"
    elif any(w in query_lower for w in ["best rated", "top rated", "highest rated"]):
        preference = "high rating"
    elif any(w in query_lower for w in ["on sale", "discount", "deal"]):
        preference = "discount"

    # --- Smart detection from raw query text ---
    detected_category = detect_category(query_lower)
    detected_color    = detect_color(query_lower)
    detected_usage    = detect_usage(query_lower)
    extracted_price   = extract_max_price(query_lower)

    if detected_category and not category:
        category = detected_category
    if detected_color and not color:
        color = detected_color
    if detected_usage and not usage:
        usage = detected_usage
    if extracted_price:
        max_price = extracted_price

    # --- Carry over context from previous session if not re-specified ---
    if not category and session_id in session_memory:
        category = session_memory[session_id].get("category", "")
    if not color and session_id in session_memory:
        color = session_memory[session_id].get("color", "")
    if not usage and session_id in session_memory:
        usage = session_memory[session_id].get("usage", "")

    # --- Relative price shift ("more expensive" / "cheaper") ---
    relative_shift = detect_relative_price_shift(query_lower)
    price_anchor   = get_price_anchor(session_id)

    # --- Apply filters ---
    filtered = apply_filters(df, category, color, usage, price_range, max_price, query_lower)

    if relative_shift and price_anchor and not extracted_price:
        filtered = apply_relative_price_filter(filtered, relative_shift, price_anchor)
        if relative_shift == "higher":
            filtered = filtered.sort_values(by='selling_price', ascending=True)
        elif relative_shift == "lower":
            filtered = filtered.sort_values(by='selling_price', ascending=False)
    else:
        filtered = apply_sorting(filtered, preference)

    # --- Show all vs paginated ---
    if "all" in query_lower or intent_name == "Show All":
        results = filtered
    else:
        results = filtered.head(3)

    reply = format_reply(results, category)

    # --- Save session context ---
    session_memory[session_id] = {
        "category":    category,
        "color":       color,
        "usage":       usage,
        "preference":  preference,
        "price_range": price_range,
        "max_price":   max_price,
        "query_text":  query_lower,
        "page":        1,
        "awaiting_more_confirm": False
    }

    save_last_shown_price(session_id, results)

    # --- Append "show more?" nudge ---
    if "all" not in query_lower and intent_name != "Show All" and not results.empty:
        reply = append_more_prompt(reply, session_id, len(filtered))
        session_memory[session_id]["awaiting_more_confirm"] = has_more_results(session_id, len(filtered))

    return jsonify({"fulfillmentText": reply})


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
