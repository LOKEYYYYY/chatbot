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
                return category

    return ""


def extract_max_price(query_text):
    match = re.search(r"(under|below|less than)\s*(\d+)", query_text)
    if match:
        return float(match.group(2))
    return None


def detect_relative_price_shift(query_text):
    """
    Detects if the user wants relatively more expensive or cheaper products
    compared to what was last shown, without meaning the absolute most/least.
    Returns: 'higher', 'lower', or None
    """
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
    """
    Returns the max price of the last shown results stored in session,
    used as a reference point for relative price shifts.
    """
    if session_id in session_memory:
        return session_memory[session_id].get("last_max_shown_price", None)
    return None


def apply_filters(df, product, price_range, max_price, query_text=""):
    filtered = df.copy()

    if product:
        category_map = {
            "electronics": "Electronics",
            "footwear": "Footwear",
            "books": "Books",
            "appliances": "Appliances",
            "clothing": "Apparel"
        }

        mapped_category = category_map.get(product, product)

        filtered = filtered[
            filtered['Category'].str.contains(mapped_category, case=False, na=False)
        ]

        for word in query_text.split():
            if len(word) > 3:
                refined = filtered[
                    filtered['Product Name'].str.contains(word, case=False, na=False)
                ]
                # Only apply keyword refinement if it doesn't wipe all results
                if not refined.empty:
                    filtered = refined

    if price_range == "cheap":
        filtered = filtered[filtered['Price'] <= 700]
    elif price_range == "mid range":
        filtered = filtered[(filtered['Price'] > 700) & (filtered['Price'] <= 1400)]
    elif price_range == "expensive":
        filtered = filtered[filtered['Price'] > 1400]

    if max_price:
        filtered = filtered[filtered['Price'] <= max_price]

    return filtered


def apply_relative_price_filter(filtered, direction, price_anchor):
    """
    Filters results to be relatively higher or lower than the price anchor.
    'higher' -> shows products strictly above the anchor price
    'lower'  -> shows products strictly below the anchor price
    Falls back to full filtered set if no anchor or result would be empty.
    """
    if price_anchor is None:
        return filtered

    if direction == "higher":
        shifted = filtered[filtered['Price'] > price_anchor]
    elif direction == "lower":
        shifted = filtered[filtered['Price'] < price_anchor]
    else:
        return filtered

    # Only apply if it doesn't wipe all results
    if not shifted.empty:
        return shifted

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
        return filtered.sort_values(by='Price', ascending=True)

    elif preference == "expensive":
        return filtered.sort_values(by='Price', ascending=False)

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


def save_last_shown_price(session_id, results):
    """
    Saves the max price of the currently shown results into session memory
    so relative price shifts ('more expensive', 'cheaper') have a reference point.
    """
    if not results.empty and session_id in session_memory:
        session_memory[session_id]["last_max_shown_price"] = float(results['Price'].max())


def has_more_results(session_id, filtered_total):
    """
    Checks whether there are more results beyond the current page.
    Returns True if there are unseen results remaining.
    """
    if session_id not in session_memory:
        return False
    page = session_memory[session_id].get("page", 1)
    shown_so_far = page * 3
    return shown_so_far < filtered_total


def append_more_prompt(reply, session_id, filtered_total):
    """
    Appends a friendly 'want to see more?' nudge to the reply
    if there are more results available beyond the current page.
    """
    if has_more_results(session_id, filtered_total):
        reply += "\n\n👀 Want to see more options? Just say *'show more'*!"
    else:
        reply += "\n\n✅ That's all the results for this search."
    return reply


def detect_yes_intent(query_text):
    """
    Detects if the user is saying yes/confirm in response to the 'want more?' prompt.
    Returns True if the user is affirming, False otherwise.
    """
    query_text = query_text.lower().strip()
    yes_phrases = [
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "show more",
        "more", "next", "continue", "go on", "show me more", "see more",
        "please", "yes please", "of course", "why not"
    ]
    return any(phrase == query_text or query_text.startswith(phrase) for phrase in yes_phrases)


def detect_no_intent(query_text):
    """
    Detects if the user is declining / saying no more.
    Returns True if the user is declining.
    """
    query_text = query_text.lower().strip()
    no_phrases = [
        "no", "nope", "nah", "no thanks", "no thank you", "that's enough",
        "stop", "done", "enough", "i'm good", "im good", "no more"
    ]
    return any(phrase == query_text or query_text.startswith(phrase) for phrase in no_phrases)


def get_summary_label(preference, product):
    """
    Builds a short human-readable label of the current search context
    for use in follow-up prompts (e.g. 'cheap socks', 'popular laptops').
    """
    parts = []
    if preference and preference not in ("popular", "best"):
        parts.append(preference)
    if product:
        parts.append(product)
    return " ".join(parts) if parts else "products"


def suggest_related_categories(product):
    """
    Suggests related categories when a search returns no results,
    giving the user a smarter fallback than a generic list.
    """
    related_map = {
        "electronics": ["appliances", "gadgets"],
        "footwear": ["clothing", "accessories"],
        "books": ["magazines", "stationery"],
        "appliances": ["electronics", "home goods"],
        "clothing": ["footwear", "accessories"]
    }
    return related_map.get(product, [])


def format_no_results_reply(product):
    """
    Returns a smarter no-results message that includes related category suggestions
    when available, otherwise falls back to the top 3 available categories.
    """
    related = suggest_related_categories(product)

    if related:
        reply = f"😢 No results found for *{product}*.\n\nYou might also like:\n"
        reply += "\n".join([f"• {r.title()}" for r in related])
        reply += "\n\nOr type a category to search again 🔍"
    else:
        suggestions = df['Category'].dropna().unique()[:3]
        reply = "😢 No exact match found.\n\nTry searching for:\n"
        reply += "\n".join([f"• {s}" for s in suggestions])

    return reply


def format_reply(results, product=""):
    """
    Formats the product list reply. Uses smarter no-results messaging
    when a product context is available.
    """
    if results.empty:
        return format_no_results_reply(product)

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

    # LIST ALL CATEGORIES
    if intent_name == "List Categories":
        categories = df['Category'].dropna().unique()

        reply = "🛍️ We have these product categories:\n\n"
        reply += "\n".join([f"• {c}" for c in categories])

        return jsonify({"fulfillmentText": reply})

    # HANDLE USER SAYING "NO" after a 'want more?' prompt
    # Gracefully ends the browsing session and thanks the user.
    if intent_name == "Negative Intent" or detect_no_intent(query_text):
        if session_id in session_memory and session_memory[session_id].get("awaiting_more_confirm"):
            session_memory[session_id]["awaiting_more_confirm"] = False
            label = get_summary_label(
                session_memory[session_id].get("preference", ""),
                session_memory[session_id].get("product", "")
            )
            return jsonify({"fulfillmentText": f"No problem! Hope you found what you needed 😊\nFeel free to search for more {label} anytime!"})

    # HANDLE SHOW MORE — triggered by intent OR by user saying yes to the 'want more?' nudge
    if intent_name == "Show More Intent" or (
        session_id in session_memory
        and session_memory[session_id].get("awaiting_more_confirm")
        and detect_yes_intent(query_text)
    ):
        if session_id in session_memory:
            memory = session_memory[session_id]

            product = memory.get("product", "")
            preference = memory["preference"]
            price_range = memory["price_range"]
            max_price = memory["max_price"]
            # Use the saved query_text from the original search for consistent filtering
            saved_query_text = memory.get("query_text", "")
            page = memory.get("page", 1) + 1

            filtered = apply_filters(df, product, price_range, max_price, saved_query_text)

            if filtered.empty:
                return jsonify({"fulfillmentText": "No more results 😢 (filter returned empty)"})

            filtered = apply_sorting(filtered, preference)

            start = (page - 1) * 3
            end = start + 3
            results = filtered.iloc[start:end]

            if results.empty:
                return jsonify({"fulfillmentText": "No more results 😢\n\nTry a different search to discover more products! 🔍"})

            session_memory[session_id]["page"] = page
            session_memory[session_id]["awaiting_more_confirm"] = False
            save_last_shown_price(session_id, results)

            reply = format_reply(results, product)
            reply = append_more_prompt(reply, session_id, len(filtered))

            # Flag that we're waiting for user to confirm they want the next page
            session_memory[session_id]["awaiting_more_confirm"] = has_more_results(session_id, len(filtered))

            return jsonify({"fulfillmentText": reply})

        else:
            return jsonify({"fulfillmentText": "Please search for something first 😊"})

    product = params.get('product') or ""
    price_range = params.get('price_range') or ""
    preference = normalize_preference(params.get('preference'))
    max_price = params.get('max_price')

    # FORCE override using user text
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
        # Carry over the product from previous search if user didn't specify a new one
        product = session_memory[session_id].get("product", "")

    if extracted_price:
        max_price = extracted_price

    # --- Relative price shift: "more expensive" / "cheaper" ---
    # Kicks in when no absolute price override was found in the query,
    # and there is a previous result to anchor from.
    relative_shift = detect_relative_price_shift(query_text)
    price_anchor = get_price_anchor(session_id)

    filtered = apply_filters(df, product, price_range, max_price, query_text)

    if relative_shift and price_anchor and not extracted_price:
        # Apply relative filter on top of existing category/preference filters
        filtered = apply_relative_price_filter(filtered, relative_shift, price_anchor)
        # Sort direction matches user intent: pricier options start from just above anchor;
        # cheaper options start from just below anchor
        if relative_shift == "higher":
            filtered = filtered.sort_values(by='Price', ascending=True)
        elif relative_shift == "lower":
            filtered = filtered.sort_values(by='Price', ascending=False)
    else:
        filtered = apply_sorting(filtered, preference)

    # SHOW ALL support
    if "all" in query_text or intent_name == "Show All":
        results = filtered
    else:
        results = filtered.head(3)

    reply = format_reply(results, product)

    # SAVE CONTEXT — includes query_text so Show More replays the exact same search
    session_memory[session_id] = {
        "product": product,
        "preference": preference,
        "price_range": price_range,
        "max_price": max_price,
        "query_text": query_text,
        "page": 1,
        "awaiting_more_confirm": False
    }

    save_last_shown_price(session_id, results)

    # Append 'want more?' nudge only when showing a partial list (not show-all)
    if "all" not in query_text and intent_name != "Show All" and not results.empty:
        reply = append_more_prompt(reply, session_id, len(filtered))
        session_memory[session_id]["awaiting_more_confirm"] = has_more_results(session_id, len(filtered))

    return jsonify({"fulfillmentText": reply})


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
