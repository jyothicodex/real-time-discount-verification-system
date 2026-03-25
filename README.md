# 🛍️ DeciBuy — Smart Price Auditor

## 📌 Overview

DeciBuy is a lightweight, intelligent price analysis system designed to help users make smarter shopping decisions.
It evaluates product prices, detects misleading discounts, and recommends the best value options.

---

## 🚀 Key Features

* 🔍 **Single Item Analysis**
  Check product price against market trends and get a deal score.

* ⚠️ **Fake Discount Detection**
  Detects:

  * Inflated MRP
  * Shrinkflation (less quantity, same price)
  * Misleading discount percentages

* 📊 **Unit Price Calculation**
  Converts all products into ₹ per 100g/ml for fair comparison.

* 📦 **Pack Comparison**
  Compare up to 3 pack sizes to find the best value.

* 🧾 **Bill OCR Scanner**
  Upload receipt images to extract and analyze items automatically.

* 📈 **Purchase History**
  Save and review past purchases with CSV export support.

* 🤖 **AI Explanations (Optional)**
  Uses Ollama for structured insights (fallback available if not installed).

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

### 3. Open in browser

```
http://localhost:8501
```

---

## 🧠 Optional AI Setup

For enhanced explanations:

```bash
# Install Ollama and run:
ollama pull qwen3-vl:4b
```

> If not installed, the system automatically uses rule-based logic.

---

## 🖥️ Application Tabs

| Tab              | Description                        |
| ---------------- | ---------------------------------- |
| 🔍 Single Item   | Analyze individual product pricing |
| 🛒 Cart Check    | Evaluate multiple items together   |
| 🧾 Bill OCR      | Scan and analyze receipt images    |
| 📈 My History    | Track and review past purchases    |
| 📊 Compare Packs | Find best value across pack sizes  |
| ℹ️ About         | Project details and explanation    |

---

## 📂 Project Structure

```
DeciBuy/
├── app.py
├── price_engine.py
├── discount_detector.py
├── ai_engine.py
├── ocr_engine.py
├── deal_classifier.py
├── product_db.csv
├── user_history.csv
└── requirements.txt
```

---

## 💡 How to Use

1. Enter product details (price, quantity, unit)
2. Click **Analyze**
3. View:

   * Deal Score
   * Discount flags
   * Smart recommendation

👉 Use **Compare Packs** for better quantity decisions
👉 Use **History** to track spending patterns

---

## 📝 Notes

* Data is stored locally using CSV (no database required)
* Ensure no other Streamlit app is running on the same port
* Designed for simplicity and quick decision-making

---

## 🎯 Design Philosophy

* Minimal and clean UI
* Focus on real-world usefulness
* Actionable insights over complex analytics

---

## 🔭 Future Scope

* Live price integration (Blinkit, BigBasket APIs)
* Mobile app with barcode scanning
* Crowdsourced pricing data
* Price prediction using ML

---

**DeciBuy — Smart Shopping, Real Decisions**

