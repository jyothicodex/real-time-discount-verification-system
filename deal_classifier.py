# models/deal_classifier.py
# ─────────────────────────────────────────────
# ML-based Deal Classifier for DeciBuy
# Uses a Random Forest trained on synthetic deal data
# Predicts: Excellent / Good / Fair / Bad / Fake
# ─────────────────────────────────────────────

import os
import pickle
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "deal_classifier.pkl")

# Label mapping (same order as training)
LABELS = ["Fake Discount", "Bad Deal", "Fair Price", "Good Deal", "Excellent Deal"]


def _generate_synthetic_data(n_samples: int = 1000):
    """
    Generate synthetic training data for the deal classifier.

    Features:
    - price_ratio: unit_price / market_avg (lower = better)
    - discount_pct: actual discount percentage (0–60)
    - shrinkflation: 1 if detected, else 0
    - fake_pct_flag: 1 if fake discount label, else 0
    - inflated_mrp_flag: 1 if inflated MRP, else 0

    Target: deal quality label (0=Fake, 1=Bad, 2=Fair, 3=Good, 4=Excellent)
    """
    np.random.seed(42)
    X, y = [], []

    for _ in range(n_samples):
        price_ratio       = np.random.uniform(0.5, 1.8)
        discount_pct      = np.random.uniform(0, 60)
        shrinkflation     = np.random.choice([0, 1], p=[0.8, 0.2])
        fake_pct_flag     = np.random.choice([0, 1], p=[0.85, 0.15])
        inflated_mrp_flag = np.random.choice([0, 1], p=[0.80, 0.20])

        # Determine label based on logic (mirrors the scoring function)
        flags = shrinkflation + fake_pct_flag + inflated_mrp_flag

        if flags >= 2:
            label = 0  # Fake
        elif price_ratio > 1.3 or flags >= 1:
            label = 1  # Bad
        elif price_ratio > 1.0:
            label = 2  # Fair
        elif price_ratio > 0.85 and discount_pct < 10:
            label = 3  # Good
        elif price_ratio <= 0.85 or discount_pct >= 20:
            label = 4  # Excellent
        else:
            label = 3  # Good

        X.append([price_ratio, discount_pct, shrinkflation,
                   fake_pct_flag, inflated_mrp_flag])
        y.append(label)

    return np.array(X), np.array(y)


def train_and_save_model():
    """
    Train the Random Forest classifier on synthetic data and save to disk.
    Called once on first run if no model file exists.
    """
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn not installed."

    X, y = _generate_synthetic_data(n_samples=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))

    # Persist model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    return clf, f"Model trained. Accuracy: {acc*100:.1f}%"


def load_model():
    """Load saved model from disk, or train a new one if not found."""
    if not SKLEARN_AVAILABLE:
        return None

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        clf, _ = train_and_save_model()
        return clf


def ml_predict(unit_price: float, market_avg: float,
                discount_pct: float = 0,
                shrinkflation: int = 0,
                fake_pct_flag: int = 0,
                inflated_mrp_flag: int = 0) -> dict:
    """
    Predict deal quality using the trained ML model.

    Args:
        unit_price: Price per 100g/ml
        market_avg: Market average per 100g/ml
        discount_pct: Actual discount % (0 if unknown)
        shrinkflation: 1 if shrinkflation detected
        fake_pct_flag: 1 if fake discount % label
        inflated_mrp_flag: 1 if inflated MRP

    Returns:
        dict with: label (str), confidence (float), available (bool)
    """
    if not SKLEARN_AVAILABLE:
        return {"label": "N/A", "confidence": 0.0, "available": False}

    clf = load_model()
    if clf is None:
        return {"label": "N/A", "confidence": 0.0, "available": False}

    price_ratio = (unit_price / market_avg) if market_avg > 0 else 1.0
    features = np.array([[price_ratio, discount_pct, shrinkflation,
                           fake_pct_flag, inflated_mrp_flag]])

    pred_class = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    confidence = round(float(probabilities[pred_class]) * 100, 1)

    return {
        "label":      LABELS[pred_class],
        "confidence": confidence,
        "available":  True
    }
