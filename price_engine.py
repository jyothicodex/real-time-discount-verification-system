# utils/price_engine.py
# ─────────────────────────────────────────────
# Core pricing logic for DeciBuy
# Handles: unit price, market comparison, deal scoring
# ─────────────────────────────────────────────

import pandas as pd
import os

# Path to the product database
DB_PATH = os.path.join(os.path.dirname(__file__), "product_db.csv")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "user_history.csv")


def load_product_db() -> pd.DataFrame:
    """Load the market average price database."""
    try:
        return pd.read_csv(DB_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def load_user_history() -> pd.DataFrame:
    """Load the user's purchase history."""
    try:
        df = pd.read_csv(HISTORY_PATH)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "timestamp", "product_name", "price", "quantity_g_ml",
            "unit_price_per_100", "store", "deal_score", "verdict"
        ])


def save_to_history(product_name: str, price: float, quantity_g_ml: float,
                    unit_price: float, store: str, deal_score: int, verdict: str):
    """Save a purchase record to user history CSV."""
    from datetime import datetime
    df = load_user_history()
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "product_name": product_name,
        "price": price,
        "quantity_g_ml": quantity_g_ml,
        "unit_price_per_100": round(unit_price, 2),
        "store": store,
        "deal_score": deal_score,
        "verdict": verdict
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)


def normalize_quantity(quantity: float, unit: str) -> float:
    """
    Convert any unit to grams or ml (base unit).
    kg → g (× 1000), L → ml (× 1000), g/ml stays as is.
    """
    if unit in ["kg", "L"]:
        return quantity * 1000
    return quantity


def calculate_unit_price(price: float, quantity_g_ml: float) -> float:
    """
    Calculate cost per 100 grams or 100 ml.
    Formula: (price / quantity) × 100
    """
    if quantity_g_ml <= 0:
        return 0.0
    return (price / quantity_g_ml) * 100


def get_market_average(product_name: str) -> dict:
    """
    Lookup the market average price for a product from the database.
    Returns a dict with avg price and metadata, or None if not found.
    Uses fuzzy string matching (case-insensitive substring).
    """
    db = load_product_db()
    if db.empty:
        return None

    # Try exact match first, then fuzzy
    product_lower = product_name.lower().strip()
    match = db[db["product_name"].str.lower() == product_lower]

    if match.empty:
        # Fuzzy: check if product name is contained in DB names or vice versa
        match = db[
            db["product_name"].str.lower().str.contains(product_lower, na=False, regex=False) |
            pd.Series([product_lower] * len(db)).str.contains(
                db["product_name"].str.lower(), na=False, regex=False
            )
        ]

    if match.empty:
        return None

    row = match.iloc[0]
    return {
        "product_name": row["product_name"],
        "category": row["category"],
        "brand": row["brand"],
        "avg_price_per_100": row["avg_price_per_100g_ml"],
        "unit": row["unit"]
    }


def get_last_purchase(product_name: str) -> dict:
    """
    Retrieve the most recent purchase of a product from history.
    Used for auto-filling repeat-buyer mode.
    """
    history = load_user_history()
    if history.empty:
        return None

    product_lower = product_name.lower().strip()
    matches = history[history["product_name"].str.lower().str.contains(product_lower, na=False)]

    if matches.empty:
        return None

    last = matches.sort_values("timestamp", ascending=False).iloc[0]
    return last.to_dict()


def calculate_deal_score(unit_price: float, market_avg: float,
                          discount_pct: float = 0.0,
                          fake_discount_flag: bool = False,
                          shrinkflation_flag: bool = False) -> int:
    """
    Calculate a Deal Score from 0–100.

    Scoring breakdown:
    - Base score from price vs market average (0–70 points)
    - Discount bonus (0–20 points)
    - Penalty for fake discount or shrinkflation (-20 points each)

    Returns an integer score 0–100.
    """
    if market_avg <= 0:
        # No market data: score based on discount only
        base_score = 50
    else:
        # Price below market → high score; above market → low score
        ratio = unit_price / market_avg  # 1.0 = exactly at market price
        if ratio <= 0.7:
            base_score = 70         # 30%+ cheaper than market
        elif ratio <= 0.85:
            base_score = 60         # 15–30% cheaper
        elif ratio <= 1.0:
            base_score = 50         # at or slightly below market
        elif ratio <= 1.15:
            base_score = 35         # slightly above market
        elif ratio <= 1.30:
            base_score = 20         # clearly overpriced
        else:
            base_score = 10         # very overpriced

    # Discount bonus: genuine discount adds up to 20 points
    discount_bonus = min(20, int(discount_pct * 0.8)) if not fake_discount_flag else 0

    # Penalties
    fake_penalty   = 25 if fake_discount_flag else 0
    shrink_penalty = 15 if shrinkflation_flag else 0

    raw_score = base_score + discount_bonus - fake_penalty - shrink_penalty
    return max(0, min(100, raw_score))


def score_to_verdict(score: int) -> tuple:
    """
    Convert numeric score to a label and emoji.
    Returns (label, emoji, color).
    """
    if score >= 80:
        return "Excellent Deal", "🏆", "green"
    elif score >= 60:
        return "Good Deal", "✅", "lightgreen"
    elif score >= 40:
        return "Fair Price", "⚖️", "orange"
    elif score >= 20:
        return "Bad Deal", "❌", "red"
    else:
        return "Fake Discount", "🚨", "darkred"
