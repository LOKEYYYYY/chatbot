from flask import Flask, request, jsonify
import pandas as pd
import re
import os

app = Flask(__name__)

# Load and clean dataset once
df = pd.read_csv("adidas_usa.csv")
df = df[df['selling_price'].notna()].copy()
df['selling_price']  = df['selling_price'].astype(float)
df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['reviews_count']  = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0).astype(int)
df['availability']   = df['availability'].fillna('Unknown')
df['color']          = df['color'].fillna('')
df['category']       = df['category'].fillna('')
df['breadcrumbs']    = df['breadcrumbs'].fillna('')
df['description']    = df['description'].fillna('')

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
        "popular":     ["top selling", "best selling", "trending", "top popularity"],
        "discount":    ["sale", "deals", "discounted"],
        "best":        ["top", "highest"],
        "high rating": ["rating", "high rating"],
        "cheap":       ["cheap", "cheapest", "lowest price", "budget"],
        "expensive":   ["expensive", "most expensive", "premium", "high price"]
    }

    for key, values in mapping.items():
        if pref in values:
            return key

    return pref


def detect_product(query_text):
    """Maps query keywords to dataset categories: Shoes, Clothing, Accessories."""
    keyword_map = {
        "Shoes":       ["shoes", "sneakers", "boots", "sandals", "trainers",
                        "footwear", "kicks", "cleats", "loafers"],
        "Clothing":    ["clothing", "clothes", "shirt", "t-shirt", "jacket",
                        "pants", "shorts", "jersey", "dress", "hoodie",
                        "sweater", "tracksuit", "leggings", "apparel", "outfit"],
        "Accessories": ["accessories", "bag", "backpack", "cap", "hat", "socks",
                        "gloves", "belt", "wallet", "headband", "wristband"]
    }

    query_text = query_text.lower()
    for category, keywords in keyword_map.items():
        for word in keywords:
            if word in query_text:
                return category
    return ""


def detect_color(query_text):
    """Detects color from query text and returns capitalised form matching the dataset."""
    colors = ["black", "grey", "gray", "white", "blue", "purple", "pink",
              "green", "yellow", "red", "multicolor", "gold", "burgundy", "beige"]
    query_lower = query_text.lower()
    for color in colors:
        if color in query_lower:
            return "Grey" if color == "gray" else color.capitalize()
    return ""


def detect_usage(query_text):
    """Detects gender/sport/style keywords that map to the breadcrumbs column."""
    usage_map = {
        "Men":       ["men", "men's", "mens", "male"],
        "Women":     ["women", "women's", "womens", "female", "ladies"],
        "Kids":      ["kids", "kid", "children", "child", "boys", "girls", "junior"],
        "Soccer":    ["soccer", "football"],
        "Training":  ["training", "gym", "workout", "fitness"],
        "Running":   ["running", "run", "jogging"],
        "Originals": ["originals", "lifestyle", "casual", "retro", "classic"],
        "Swim":      ["swim", "swimming", "pool"],
        "Five Ten":  ["five ten", "mountain bike", "cycling", "bike"]
    }

    query_lower = query_text.lower()
    for usage, keywords in usage_map.items():
        for word in keywords:
            if word in query_lower:
                return usage
    return ""


def extract_max_price(query_text):
    match = re.search(r"(under|below|less than)\s*\$?\s*(\d+)", query_text)
    return float(match.group(2)) if match else None


def detect_relative_price_shift(query_text):
    """
    Detects if the user wants relatively more expensive or cheaper products
    compared to what was last shown, without meaning the absolute most/least.
    Returns: 'higher', 'lower', or None
    """
    query_text = query_text.lower()

    higher_phrases = ["more expensive", "higher price", "pricier",
                      "something more expensive", "higher end", "cost more"]
    lower_phrases  = ["cheaper", "less expensive", "lower price",
                      "something cheaper", "lower end", "cost less"]

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


def apply_filters(df, product, price_range, max_price, query_text="",
                  color="", usage=""):
    filtered = df.copy()

    # Show only in-stock products by default
    in_stock = filtered[filtered['availability'] == 'InStock']
    if not in_stock.empty:
        filtered = in_stock

    # Category filter
    if product:
        cat_filtered = filtered[filtered['category'].str.contains(
            product, case=False, na=False)]
        if not cat_filtered.empty:
            filtered = cat_filtered

    # Color filter
    if color:
        color_filtered = filtered[filtered['color'].str.contains(
            color, case=False, na=False)]
        if not color_filtered.empty:
            filtered = color_filtered

    # Usage / breadcrumb filter — falls back to description if no breadcrumb match
    if usage:
        usage_filtered = filtered[filtered['breadcrumbs'].str.contains(
            usage, case=False, na=False)]
        if not usage_filtered.empty:
            filtered = usage_filtered
        else:
            desc_filtered = filtered[filtered['description'].str.contains(
                usage, case=False, na=False)]
            if not desc_filtered.empty:
                filtered = desc_filtered

    # Price range filter  (cheap ≤ $35 | mid $35–$80 | expensive > $80)
    if price_range == "cheap":
        filtered = filtered[filtered['selling_price'] <= 35]
    elif price_range == "mid range":
        filtered = filtered[(filtered['selling_price'] > 35) &
                            (filtered['selling_price'] <= 80)]
    elif price_range == "expensive":
        filtered = filtered[filtered['selling_price'] > 80]

    if max_price:
        filtered = filtered[filtered['selling_price'] <= max_price]

    # Keyword refinement on product name
    stopwords = {"show", "find", "want", "need", "give", "tell", "look",
                 "what", "with", "that", "have", "some", "most", "best",
                 "under", "below", "adidas"}
    for word in query_text.split():
        if len(word) > 3 and word.lower() not in stopwords:
            refined = filtered[filtered['name'].str.contains(
                word, case=False, na=False)]
            if not refined.empty:
                filtered = refined

    return filtered


def apply_relative_price_filter(filtered, direction, price_anchor):
    """
    Filters results to be relatively higher or lower than the price anchor.
    Falls back to full filtered set if result would be empty.
    """
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
        # Prioritise products that actually have a marked-down original price
        on_sale = filtered[filtered['original_price'].notna()]
        if not on_sale.empty:
            return on_sale.sort_values(by='selling_price', ascending=True)
        return filtered.sort_values(by='selling_price', ascending=True)
    elif preference in ("best", "high rating"):
        return filtered.sort_values(by='average_rating', ascending=False)
    elif preference == "cheap":
        return filtered.sort_values(by='selling_price', ascending=True)
    elif preference == "expensive":
        return filtered.sort_values(by='selling_price', ascending=False)
    return filtered.sort_values(by='reviews_count', ascending=False)


def save_last_shown_price(session_id, results):
    """
    Saves the max price of the currently shown results into session memory
    so relative price shifts ('more expensive', 'cheaper') have a reference point.
    """
    if not results.empty and session_id in session_memory:
        session_memory[session_id]["last_max_shown_price"] = float(
            results['selling_price'].max())


def has_more_results(session_id, filtered_total):
    """
    Checks whether there are more results beyond the current page.
    Returns True if there are unseen results remaining.
    """
    if session_id not in session_memory:
        return False
    page = session_memory[session_id].get("page", 1)
    return (page * 3) < filtered_total


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
    return any(phrase == query_text or query_text.startswith(phrase)
               for phrase in yes_phrases)


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
    return any(phrase == query_text or query_text.startswith(phrase)
               for phrase in no_phrases)


def get_summary_label(preference, product):
    """
    Builds a short human-readable label of the current search context
    for use in follow-up prompts (e.g. 'cheap Shoes', 'popular Clothing').
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
        "Shoes":       ["Clothing", "Accessories"],
        "Clothing":    ["Shoes", "Accessories"],
        "Accessories": ["Shoes", "Clothing"]
    }
    return related_map.get(product, [])


def format_no_results_reply(product):
    """
    Returns a smarter no-results message that includes related category suggestions
    when available, otherwise falls back to the top 3 available categories.
    """
    related = suggest_related_categories(product)

    if related:
        reply  = f"😢 No results found for *{product}*.\n\nYou might also like:\n"
        reply += "\n".join([f"• {r}" for r in related])
        reply += "\n\nOr type a category to search again 🔍"
    else:
        suggestions = [c for c in df['category'].dropna().unique() if c][:3]
        reply  = "😢 No exact match found.\n\nTry searching for:\n"
        reply += "\n".join([f"• {s}" for s in suggestions])

    return reply


def format_reply(results, product=""):
    """
    Formats the product list reply. Uses smarter no-results messaging
    when a product context is available.
    """
    if results.empty:
        return format_no_results_reply(product)

    lines = []
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        # Price line — show original + sale flag when discounted
        price_line = f"💲 ${row['selling_price']:.2f}"
        if pd.notna(row['original_price']) and row['original_price'] > row['selling_price']:
            price_line += f"  ~~${row['original_price']:.2f}~~  🔖 On Sale!"

        entry = f"{i}) {row['name']}\n   {price_line}"

        if pd.notna(row['average_rating']):
            entry += (f"\n   ⭐ {row['average_rating']:.1f}/5"
                      f"  ({int(row['reviews_count'])} reviews)")
        if row['color']:
            entry += f"\n   🎨 {row['color']}"

        entry += f"\n   🏷️ {row['category']}  |  {row['breadcrumbs']}"
        lines.append(entry)

    return "👟 Adidas Products for You ✨\n\n" + "\n\n".join(lines)


# -------------------------------
# Main Webhook
# -------------------------------

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    params     = req['queryResult']['parameters']
    query_text = req['queryResult']['queryText'].lower()
    intent_name = req['queryResult']['intent']['displayName']
    session_id  = req['session']

    # --------------------------------------------------
    # LIST ALL CATEGORIES
    # --------------------------------------------------
    if intent_name == "List Categories":
        categories = [c for c in df['category'].dropna().unique() if c]
        reply  = "🛍️ Adidas USA carries these categories:\n\n"
        reply += "\n".join([f"• {c}" for c in categories])
        reply += ("\n\nYou can also filter by gender (Men / Women / Kids), "
                  "sport (Soccer / Training / Running / Swimming), "
                  "or style (Originals)!")
        return jsonify({"fulfillmentText": reply})

    # --------------------------------------------------
    # NEGATIVE INTENT — user says no to "show more?"
    # --------------------------------------------------
    if intent_name == "Negative Intent" or detect_no_intent(query_text):
        if (session_id in session_memory
                and session_memory[session_id].get("awaiting_more_confirm")):
            session_memory[session_id]["awaiting_more_confirm"] = False
            label = get_summary_label(
                session_memory[session_id].get("preference", ""),
                session_memory[session_id].get("product", "")
            )
            return jsonify({"fulfillmentText":
                f"No problem! Hope you found what you needed 😊\n"
                f"Feel free to search for more {label} anytime!"})

    # --------------------------------------------------
    # SHOW MORE INTENT
    # --------------------------------------------------
    if intent_name == "Show More Intent" or (
        session_id in session_memory
        and session_memory[session_id].get("awaiting_more_confirm")
        and detect_yes_intent(query_text)
    ):
        if session_id in session_memory:
            memory      = session_memory[session_id]
            product     = memory.get("product", "")
            preference  = memory["preference"]
            price_range = memory["price_range"]
            max_price   = memory["max_price"]
            color       = memory.get("color", "")
            usage       = memory.get("usage", "")
            # Use the saved query_text from the original search for consistent filtering
            saved_query_text = memory.get("query_text", "")
            page = memory.get("page", 1) + 1

            filtered = apply_filters(df, product, price_range, max_price,
                                     saved_query_text, color, usage)

            if filtered.empty:
                return jsonify({"fulfillmentText":
                    "No more results 😢 (filter returned empty)"})

            filtered = apply_sorting(filtered, preference)
            start    = (page - 1) * 3
            results  = filtered.iloc[start:start + 3]

            if results.empty:
                return jsonify({"fulfillmentText":
                    "No more results 😢\n\n"
                    "Try a different search to discover more products! 🔍"})

            session_memory[session_id]["page"] = page
            session_memory[session_id]["awaiting_more_confirm"] = False
            save_last_shown_price(session_id, results)

            reply = format_reply(results, product)
            reply = append_more_prompt(reply, session_id, len(filtered))

            # Flag that we're waiting for user to confirm they want the next page
            session_memory[session_id]["awaiting_more_confirm"] = \
                has_more_results(session_id, len(filtered))

            return jsonify({"fulfillmentText": reply})

        else:
            return jsonify({"fulfillmentText": "Please search for something first 😊"})

    # --------------------------------------------------
    # PRODUCT SEARCH
    # --------------------------------------------------
    product     = params.get('product') or ""
    price_range = params.get('price_range') or ""
    preference  = normalize_preference(params.get('preference'))
    max_price   = params.get('max_price')
    color       = params.get('color') or ""
    usage       = params.get('usage') or ""

    # Normalise Dialogflow list params
    if isinstance(product, list):     product     = product[0]     if product     else ""
    if isinstance(color, list):       color       = color[0]       if color       else ""
    if isinstance(usage, list):       usage       = usage[0]       if usage       else ""
    if isinstance(price_range, list): price_range = price_range[0] if price_range else ""

    # FORCE override preference from query text
    if any(w in query_text for w in ["cheap", "cheapest", "lowest price", "budget"]):
        preference = "cheap"
    elif any(w in query_text for w in ["expensive", "most expensive", "highest price", "premium"]):
        preference = "expensive"

    # Smart extraction from text
    detected_product = detect_product(query_text)
    detected_color   = detect_color(query_text)
    detected_usage   = detect_usage(query_text)
    extracted_price  = extract_max_price(query_text)

    if detected_product:
        product = detected_product
    elif not product and session_id in session_memory:
        # Carry over the product from previous search if user didn't specify a new one
        product = session_memory[session_id].get("product", "")

    if detected_color and not color:
        color = detected_color
    if detected_usage and not usage:
        usage = detected_usage
    if extracted_price:
        max_price = extracted_price

    # --- Relative price shift: "more expensive" / "cheaper" ---
    # Kicks in when no absolute price override was found in the query,
    # and there is a previous result to anchor from.
    relative_shift = detect_relative_price_shift(query_text)
    price_anchor   = get_price_anchor(session_id)

    filtered = apply_filters(df, product, price_range, max_price,
                             query_text, color, usage)

    if relative_shift and price_anchor and not extracted_price:
        # Apply relative filter on top of existing category/preference filters
        filtered = apply_relative_price_filter(filtered, relative_shift, price_anchor)
        # Sort direction matches user intent: pricier options start from just above anchor;
        # cheaper options start from just below anchor
        if relative_shift == "higher":
            filtered = filtered.sort_values(by='selling_price', ascending=True)
        elif relative_shift == "lower":
            filtered = filtered.sort_values(by='selling_price', ascending=False)
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
        "product":     product,
        "preference":  preference,
        "price_range": price_range,
        "max_price":   max_price,
        "color":       color,
        "usage":       usage,
        "query_text":  query_text,
        "page":        1,
        "awaiting_more_confirm": False
    }

    save_last_shown_price(session_id, results)

    # Append 'want more?' nudge only when showing a partial list (not show-all)
    if "all" not in query_text and intent_name != "Show All" and not results.empty:
        reply = append_more_prompt(reply, session_id, len(filtered))
        session_memory[session_id]["awaiting_more_confirm"] = \
            has_more_results(session_id, len(filtered))

    return jsonify({"fulfillmentText": reply})


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
