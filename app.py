# app.py
# ═══════════════════════════════════════════════════════════
#  DeciBuy — Real-Time Discount Verification System
#  Version 2.0 | Research-Grade Implementation
#  Run with: streamlit run app.py
# ═══════════════════════════════════════════════════════════

import sys
import os

# Add project root to path so utils/models can be imported
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Internal modules ──
from price_engine      import (normalize_quantity, calculate_unit_price,
                                get_market_average, get_last_purchase,
                                calculate_deal_score, score_to_verdict,
                                load_user_history, save_to_history)
from discount_detector import full_discount_audit
from ai_engine         import get_structured_verdict, get_cart_verdict
from ocr_engine        import extract_text_from_image, parse_bill_items, ocr_status
from deal_classifier   import ml_predict, train_and_save_model

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="DeciBuy | Real-Time Discount Verification",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #f8f9fa; }

/* Score card */
.score-card {
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    font-size: 2.2em;
    font-weight: bold;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}
.score-excellent { background: linear-gradient(135deg, #11998e, #38ef7d); }
.score-good      { background: linear-gradient(135deg, #56ab2f, #a8e063); }
.score-fair      { background: linear-gradient(135deg, #f7971e, #ffd200); }
.score-bad       { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
.score-fake      { background: linear-gradient(135deg, #1a1a2e, #e94560); }

/* Verdict box */
.verdict-box {
    padding: 16px 20px;
    border-radius: 10px;
    border-left: 5px solid #333;
    background: #fff;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* Flag badge */
.flag-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 4px 4px 4px 0;
}
.flag-red   { background: #ffe0e0; color: #c0392b; }
.flag-green { background: #e0ffe0; color: #27ae60; }

/* Sidebar header */
.sidebar-title {
    font-size: 1.3em;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 10px;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-size: 16px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# HELPER: render deal score card
# ═══════════════════════════════════════════════════════════
def render_score_card(score: int, label: str, emoji: str):
    css_class = {
        "Excellent Deal": "score-excellent",
        "Good Deal":      "score-good",
        "Fair Price":     "score-fair",
        "Bad Deal":       "score-bad",
        "Fake Discount":  "score-fake"
    }.get(label, "score-fair")

    st.markdown(
        f'<div class="score-card {css_class}">'
        f'{emoji} Deal Score: {score}/100<br>'
        f'<span style="font-size:0.55em">{label}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.progress(score / 100)


def render_ai_verdict(verdict: dict):
    """Render the structured AI verdict in a clean 3-section card."""
    ai_label = "🧠 AI Analysis" if verdict.get("ai_available") else "📊 Rule-Based Analysis"
    st.subheader(ai_label)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**📋 Summary**\n\n{verdict['summary']}")
    with col2:
        st.warning(f"**🔍 Reason**\n\n{verdict['reason']}")
    with col3:
        st.success(f"**💡 Suggestion**\n\n{verdict['suggestion']}")


def render_flags(audit: dict):
    """Display discount audit flags visually."""
    st.subheader("🚩 Discount Audit")
    checks = [
        ("Inflated MRP",        audit["inflated_mrp"]),
        ("Fake % Label",        audit["fake_percentage"]),
        ("Shrinkflation",       audit["shrinkflation"]),
    ]
    for name, result in checks:
        if result["detected"]:
            st.markdown(
                f'<span class="flag-badge flag-red">🚨 {name}: DETECTED</span>',
                unsafe_allow_html=True
            )
            st.caption(result["reason"])
        else:
            st.markdown(
                f'<span class="flag-badge flag-green">✅ {name}: Clean</span>',
                unsafe_allow_html=True
            )


def price_comparison_chart(product_name: str, unit_price: float, market_avg: float,
                             prev_unit: float = None):
    """Plotly bar chart: your price vs market average vs last purchase."""
    labels  = ["Your Price", "Market Avg"]
    values  = [unit_price, market_avg]
    colors  = ["#e74c3c", "#2ecc71"]

    if prev_unit and prev_unit > 0:
        labels.append("Your Last Purchase")
        values.append(prev_unit)
        colors.append("#3498db")

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"₹{v:.2f}" for v in values],
        textposition="outside"
    ))
    fig.update_layout(
        title=f"Price Comparison — {product_name} (per 100g/ml)",
        yaxis_title="₹ per 100g/ml",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛍️ DeciBuy v2.0</div>', unsafe_allow_html=True)
    st.caption("Real-Time Discount Verification System")
    st.divider()

    st.markdown("**🔧 System Status**")
    ocr_stat = ocr_status()
    st.markdown(f"{'✅' if ocr_stat['easyocr'] else '❌'} EasyOCR")
    st.markdown(f"{'✅' if ocr_stat['pytesseract'] else '❌'} Pytesseract")

    try:
        import sklearn
        st.markdown("✅ scikit-learn (ML)")
    except ImportError:
        st.markdown("❌ scikit-learn (ML)")

    try:
        import plotly
        st.markdown("✅ Plotly (Charts)")
    except ImportError:
        st.markdown("❌ Plotly (Charts)")

    st.divider()

    # User Shopping Statistics
    # Store selector (for labeling history entries)
    store_name = st.text_input("🏪 Store Name (optional)", placeholder="e.g., D-Mart")

    st.divider()
    st.markdown("**📖 How to use:**")
    st.caption("1. Enter product details in any tab\n"
               "2. Click Analyze\n"
               "3. View score, flags & AI verdict\n"
               "4. Check History tab to track purchases")

    st.divider()


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.title("🛍️ DeciBuy — Real-Time Discount Verification System")
st.markdown("*Detect fake discounts • Compare market prices • Shop smarter*")
st.divider()


# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Single Item",
    "🛒 Cart Check",
    "🧾 Bill OCR",
    "📈 My History",
    "ℹ️ About",
    "📊 Compare Packs"
])


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 1 — SINGLE ITEM CHECK                          ║
# ╚═══════════════════════════════════════════════════════╝
with tab1:
    st.subheader("🔍 Single Item Price Analyzer")

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        # ── Product info ──
        product = st.text_input("🧾 Product Name",
                                 placeholder="e.g., Fortune Sunflower Oil, Amul Milk")

        # Auto-lookup market data when product name entered
        market_data = None
        if product.strip():
            market_data = get_market_average(product)
            if market_data:
                st.success(f"📦 Found in database: **{market_data['product_name']}** "
                           f"({market_data['category']}) — "
                           f"Market avg: ₹{market_data['avg_price_per_100']:.2f}/100g")
            else:
                st.info("🔎 Product not in database. Market comparison will use manual data.")

        # ── Current price inputs ──
        st.markdown("**🏪 Current Store Price**")
        c1, c2, c3 = st.columns(3)
        with c1:
            curr_price = st.number_input("Price (₹)", min_value=0.1, step=1.0, key="s_price")
        with c2:
            curr_qty   = st.number_input("Quantity", min_value=0.1, value=500.0, key="s_qty")
        with c3:
            curr_unit  = st.selectbox("Unit", ["g", "ml", "kg", "L"], key="s_unit")

        # ── MRP and discount ──
        st.markdown("**🏷️ Label Details (optional but recommended)**")
        d1, d2 = st.columns(2)
        with d1:
            mrp = st.number_input("MRP (₹)", min_value=0.0, step=1.0,
                                   help="Maximum Retail Price printed on pack")
        with d2:
            claimed_disc = st.number_input("Claimed Discount %", min_value=0.0,
                                            max_value=99.0,
                                            help="Discount % shown on offer/label")

        # ── Buyer type (first-time vs repeat buyer) ──
        buyer_type = st.radio(
            "Buyer Type",
            ["First-time buyer", "Repeat buyer"],
            index=1,
            horizontal=True
        )

        if buyer_type == "First-time buyer":
            st.info("First-time buyer selected: previous purchase data will be omitted.")
            prev_price = 0.0
            prev_qty = 0.0
        else:
            st.markdown("**📜 Previous Purchase (for repeat buyer)**")
            last_purchase = None
            if product.strip():
                last_purchase = get_last_purchase(product)

            if last_purchase:
                st.success(f"✅ Auto-filled from history: "
                           f"₹{last_purchase['price']} for {last_purchase['quantity_g_ml']}g/ml "
                           f"on {last_purchase['timestamp'][:10]}")
                prev_price = float(last_purchase["price"])
                prev_qty   = float(last_purchase["quantity_g_ml"])
            else:
                p1, p2 = st.columns(2)
                with p1:
                    prev_price = st.number_input("Last Price (₹)", min_value=0.0, step=1.0)
                with p2:
                    prev_qty   = st.number_input("Last Quantity (g/ml)", min_value=0.0,
                                                  value=500.0, step=10.0)

    # ── ANALYZE BUTTON ──────────────────────────────────
    if st.button("🔍 ANALYZE DEAL", use_container_width=True, type="primary"):

        if not product.strip():
            st.warning("⚠️ Please enter a product name.")
            st.stop()

        # ── Step 1: Unit price calculation ──
        norm_qty   = normalize_quantity(curr_qty, curr_unit)
        unit_price = calculate_unit_price(curr_price, norm_qty)

        # ── Step 2: Market average lookup ──
        market_avg = market_data["avg_price_per_100"] if market_data else 0.0

        # ── Step 3: Discount audit ──
        sell_price = curr_price if mrp == 0 else min(curr_price, mrp)
        audit = full_discount_audit(
            product_name        = product,
            mrp                 = mrp if mrp > 0 else curr_price,
            selling_price       = sell_price,
            claimed_discount_pct= claimed_disc,
            curr_qty            = norm_qty,
            prev_qty            = prev_qty,
            prev_price          = prev_price,
            market_avg_unit     = market_avg
        )

        # ── Step 4: Deal score ──
        score = calculate_deal_score(
            unit_price          = unit_price,
            market_avg          = market_avg,
            discount_pct        = claimed_disc,
            fake_discount_flag  = audit["any_fake_flag"],
            shrinkflation_flag  = audit["shrinkflation"]["detected"]
        )
        label, emoji, _ = score_to_verdict(score)

        # ── Step 5: ML prediction ──
        ml_result = ml_predict(
            unit_price        = unit_price,
            market_avg        = market_avg if market_avg > 0 else unit_price,
            discount_pct      = claimed_disc,
            shrinkflation     = int(audit["shrinkflation"]["detected"]),
            fake_pct_flag     = int(audit["fake_percentage"]["detected"]),
            inflated_mrp_flag = int(audit["inflated_mrp"]["detected"])
        )

        # ── RESULTS ─────────────────────────────────────
        st.divider()
        res_col1, res_col2 = st.columns([1, 1.5])

        with res_col1:
            render_score_card(score, label, emoji)

            st.markdown("**📊 Quick Numbers**")
            st.metric("Your Unit Price", f"₹{unit_price:.2f}/100g")
            if market_avg > 0:
                delta_pct = ((unit_price - market_avg) / market_avg) * 100
                st.metric("Market Average",
                           f"₹{market_avg:.2f}/100g",
                           delta=f"{delta_pct:+.1f}% vs market",
                           delta_color="inverse")

            if ml_result["available"]:
                st.markdown(
                    f"**🤖 ML Prediction:** {ml_result['label']} "
                    f"({ml_result['confidence']:.0f}% confidence)"
                )

        with res_col2:
            render_flags(audit)

        # ── Price chart ──
        if market_avg > 0:
            prev_unit_price = calculate_unit_price(prev_price, prev_qty) if prev_price > 0 and prev_qty > 0 else None
            price_comparison_chart(product, unit_price, market_avg, prev_unit_price)

        # ── AI Verdict ──
        with st.spinner("Generating AI explanation..."):
            verdict = get_structured_verdict(
                product_name  = product,
                unit_price    = unit_price,
                market_avg    = market_avg,
                deal_score    = score,
                verdict_label = label,
                flags         = audit
            )
        render_ai_verdict(verdict)

        # ── Save to history ──
        save_to_history(
            product_name   = product,
            price          = curr_price,
            quantity_g_ml  = norm_qty,
            unit_price     = unit_price,
            store          = store_name or "Unknown",
            deal_score     = score,
            verdict        = label
        )
        st.caption("✅ Purchase saved to history.")


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 2 — CART CHECK                                 ║
# ╚═══════════════════════════════════════════════════════╝
with tab2:
    st.subheader("🛒 Smart Cart Analyzer")
    st.caption("Analyze multiple items at once and get a cart-wide shopping strategy.")

    num_items = st.number_input("Number of items", min_value=1, max_value=10, value=3)

    cart_items = []

    for i in range(num_items):
        with st.expander(f"📦 Item {i+1}", expanded=(i == 0)):
            a1, a2 = st.columns(2)
            with a1:
                name = st.text_input("Product Name", key=f"c_name{i}",
                                      placeholder="e.g., Dettol Soap")
            with a2:
                # Auto-lookup
                mkt = get_market_average(name) if name.strip() else None
                if mkt:
                    st.caption(f"Market avg: ₹{mkt['avg_price_per_100']:.2f}/100g")

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                cp = st.number_input("Price (₹)", min_value=0.1, key=f"c_cp{i}", step=1.0)
            with b2:
                cq = st.number_input("Qty (g/ml)", min_value=0.1, value=100.0, key=f"c_cq{i}")
            with b3:
                rp = st.number_input("Last Price", min_value=0.0, key=f"c_rp{i}", step=1.0)
            with b4:
                rq = st.number_input("Last Qty", min_value=0.0, value=100.0, key=f"c_rq{i}")

            if name.strip() and cp > 0:
                curr_u = calculate_unit_price(cp, cq)
                m_avg  = mkt["avg_price_per_100"] if mkt else 0
                ref_u  = calculate_unit_price(rp, rq) if rp > 0 and rq > 0 else m_avg

                s = calculate_deal_score(curr_u, m_avg if m_avg > 0 else ref_u)
                lbl, emj, _ = score_to_verdict(s)

                cart_items.append({
                    "name":       name,
                    "unit_price": curr_u,
                    "market_avg": m_avg,
                    "ref_unit":   ref_u,
                    "deal_score": s,
                    "verdict":    lbl,
                    "emoji":      emj
                })

    if st.button("🚀 ANALYZE CART", use_container_width=True, type="primary") and cart_items:
        st.divider()
        st.subheader("📊 Cart Score Overview")

        # Summary bar chart
        fig = px.bar(
            x=[it["name"] for it in cart_items],
            y=[it["deal_score"] for it in cart_items],
            color=[it["deal_score"] for it in cart_items],
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[0, 100],
            labels={"x": "Product", "y": "Deal Score", "color": "Score"},
            title="Deal Scores — Your Cart Items"
        )
        fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Per-item cards
        cols = st.columns(min(len(cart_items), 3))
        for idx, item in enumerate(cart_items):
            with cols[idx % 3]:
                render_score_card(item["deal_score"], item["verdict"], item["emoji"])
                st.caption(f"**{item['name']}**\n₹{item['unit_price']:.2f}/100g")

        # AI cart analysis
        with st.spinner("Generating cart insight..."):
            cart_ai = get_cart_verdict(cart_items)
        st.subheader("🧠 AI Cart Strategy")
        st.info(cart_ai)

        # Avg cart score
        avg_score = sum(it["deal_score"] for it in cart_items) // len(cart_items)
        st.metric("🎯 Average Cart Deal Score", f"{avg_score}/100")


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 3 — BILL OCR                                   ║
# ╚═══════════════════════════════════════════════════════╝
with tab3:
    st.subheader("🧾 Bill OCR — Auto-Extract & Analyze")
    st.caption("Upload a shopping bill image. DeciBuy will extract items and analyze every product automatically.")

    ocr_stat = ocr_status()
    if not ocr_stat["any_ocr"]:
        st.warning(
            "⚠️ No OCR library found. Please install one:\n\n"
            "```\npip install easyocr\n```\nor\n```\npip install pytesseract Pillow\n```"
        )
    else:
        engine = "EasyOCR" if ocr_stat["easyocr"] else "pytesseract"
        st.success(f"✅ OCR Engine: **{engine}** is ready.")

    uploaded_bill = st.file_uploader(
        "📷 Upload Bill / Receipt Image",
        type=["jpg", "jpeg", "png"],
        help="Supports most printed bill formats."
    )

    if uploaded_bill:
        st.image(uploaded_bill, caption="Uploaded Bill", use_column_width=True)

        if st.button("🔍 EXTRACT & ANALYZE", use_container_width=True, type="primary"):
            with st.spinner("Running OCR..."):
                image_bytes = uploaded_bill.read()
                raw_text = extract_text_from_image(image_bytes)

            if raw_text is None:
                st.error("OCR unavailable. Install easyocr or pytesseract.")
            elif not raw_text.strip():
                st.warning("No text extracted. Try a clearer, well-lit image.")
            else:
                with st.expander("📄 Raw OCR Text"):
                    st.text(raw_text)

                items = parse_bill_items(raw_text)

                if not items:
                    st.warning("Could not identify any priced items. "
                               "Try a cleaner bill image or manual entry.")
                else:
                    st.success(f"✅ Extracted **{len(items)} items** from bill!")
                    st.subheader("📦 Bill Analysis Results")

                    for item in items:
                        with st.expander(f"📦 {item['name']} — ₹{item['price']}"):
                            mkt = get_market_average(item["name"])
                            m_avg = mkt["avg_price_per_100"] if mkt else 0

                            if item["unit_price_per_100"]:
                                score = calculate_deal_score(item["unit_price_per_100"], m_avg)
                                lbl, emj, _ = score_to_verdict(score)
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("Unit Price", f"₹{item['unit_price_per_100']:.2f}/100g")
                                    if m_avg:
                                        st.metric("Market Avg", f"₹{m_avg:.2f}/100g")
                                with c2:
                                    render_score_card(score, lbl, emj)
                            else:
                                st.info(f"₹{item['price']} — Quantity not detected. "
                                        "Cannot calculate unit price.")


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 4 — PURCHASE HISTORY                           ║
# ╚═══════════════════════════════════════════════════════╝
with tab4:
    st.subheader("📈 Your Purchase History & Trends")

    history = load_user_history()

    if history.empty:
        st.info("No purchase history yet. Analyze items to start building your history.")
    else:
        # Filters
        f1, f2 = st.columns(2)
        with f1:
            search_term = st.text_input("🔎 Search product", placeholder="e.g., Amul")
        with f2:
            verdict_filter = st.multiselect(
                "Filter by verdict",
                ["Excellent Deal", "Good Deal", "Fair Price", "Bad Deal", "Fake Discount"],
                default=[]
            )

        filtered = history.copy()
        if search_term:
            filtered = filtered[
                filtered["product_name"].str.lower().str.contains(search_term.lower(), na=False)
            ]
        if verdict_filter:
            filtered = filtered[filtered["verdict"].isin(verdict_filter)]

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Purchases", len(filtered))
        m2.metric("Avg Deal Score", f"{filtered['deal_score'].mean():.0f}/100" if not filtered.empty else "—")
        m3.metric("Best Deal", filtered["deal_score"].max() if not filtered.empty else "—")
        m4.metric("Worst Deal", filtered["deal_score"].min() if not filtered.empty else "—")

        # History table
        st.dataframe(
            filtered[["timestamp","product_name","price","quantity_g_ml",
                       "unit_price_per_100","store","deal_score","verdict"]]
            .sort_values("timestamp", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

        # Optional analytics section (hidden by default for simple view)
        show_analytics = st.checkbox("Show deal score trends and verdict distribution", value=False)
        if show_analytics and len(filtered) >= 2:
            fig_trend = px.scatter(
                filtered,
                x="timestamp",
                y="deal_score",
                color="verdict",
                hover_name="product_name",
                title="Deal Score Trend Over Time",
                size="deal_score",
                color_discrete_map={
                    "Excellent Deal": "#2ecc71",
                    "Good Deal":      "#27ae60",
                    "Fair Price":     "#f39c12",
                    "Bad Deal":       "#e74c3c",
                    "Fake Discount":  "#8e44ad"
                }
            )
            fig_trend.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)

        if show_analytics and not filtered.empty:
            verdict_counts = filtered["verdict"].value_counts().reset_index()
            verdict_counts.columns = ["Verdict", "Count"]
            fig_pie = px.pie(verdict_counts, names="Verdict", values="Count",
                              title="Verdict Distribution",
                              color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Export
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv_data,
            file_name=f"decibuy_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 5 — ABOUT                                      ║
# ╚═══════════════════════════════════════════════════════╝
with tab5:
    st.subheader("ℹ️ About DeciBuy")

    st.markdown("""
    ### 📌 Project Abstract
    **DeciBuy** is a real-time discount verification and intelligent price analysis system
    designed to protect Indian consumers from misleading pricing tactics used by retailers
    and e-commerce platforms. The system computes unit prices, compares against a curated
    market database, detects fake discounts (inflated MRP, shrinkflation, false percentage
    labels), assigns a transparent Deal Score (0–100), and delivers structured AI-generated
    explanations. It also supports bill OCR for automatic multi-item extraction and an ML
    classifier for deal quality prediction.
    """)

    st.divider()

    col_inn, col_fut = st.columns(2)

    with col_inn:
        st.markdown("### 🚀 5 Key Innovation Points")
        st.markdown("""
1. **Real-Time Market Benchmarking** — Unit price is compared against a curated product
   database (market averages), exposing true over/under-pricing beyond surface-level
   discounts.

2. **Three-Layer Fake Discount Detection** — Simultaneously detects inflated MRP abuse,
   false percentage labels, and shrinkflation — a unique multi-vector fraud detection
   approach.

3. **Transparent Deal Score (0–100)** — A composite scoring model that weighs market
   position, discount authenticity, shrinkflation, and historical data into one
   interpretable score.

4. **Structured LLM Explanations** — Local AI (Ollama) generates consumer-grade reasoning
   in Summary → Reason → Suggestion format, with a deterministic rule-based fallback
   ensuring 100% availability.

5. **Bill OCR Auto-Analysis** — First-of-its-kind integration of OCR + unit price engine
   on uploaded receipts, enabling post-purchase review with zero manual entry.
        """)

    with col_fut:
        st.markdown("### 🔭 5 Future Scope Ideas")
        st.markdown("""
1. **Live Price Scraping** — Integrate real-time scraping from Blinkit, BigBasket, and
   Zepto APIs to replace static DB with live market prices.

2. **Mobile App (React Native)** — Port the engine to a native app with camera-based
   barcode scanning for instant shelf-side deal analysis.

3. **Federated Consumer Database** — Crowdsource price submissions from users across
   cities to build a dynamic, always-updated regional price graph.

4. **Time-Series Price Forecasting** — Use LSTM/Prophet to predict whether a product's
   price will drop in the next 7–30 days, guiding buy-now vs wait decisions.

5. **Retailer Compliance Dashboard** — A merchant-facing module that flags pricing
   violations, supporting consumer protection agencies with audit-ready reports.
        """)

    st.divider()
    st.markdown("### 🗂️ Project File Structure")
    st.code("""
DeciBuy/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── price_engine.py         # Unit price, market lookup, deal score
│   ├── discount_detector.py    # Fake discount, shrinkflation, MRP audit
│   ├── ai_engine.py            # Ollama LLM integration + rule fallback
│   └── ocr_engine.py          # EasyOCR / pytesseract bill extraction
├── models/
│   ├── __init__.py
│   ├── deal_classifier.py      # sklearn RandomForest deal predictor
│   └── deal_classifier.pkl     # Saved model (auto-generated on train)
└── data/
    ├── product_db.csv          # Market average price database (40 products)
    └── user_history.csv        # User purchase history (auto-updated)
    """, language="")

    st.divider()
    st.caption("DeciBuy v2.0 — Built for Academic Research | Powered by Python, Streamlit, Ollama, scikit-learn")


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 6 — COMPARE PACKS                              ║
# ╚═══════════════════════════════════════════════════════╝
with tab6:
    st.subheader("📊 Pack Size Comparison")
    st.caption("Compare multiple package sizes of the same product to find the best value.")

    product = st.text_input("Product Name", key="comp_product", placeholder="e.g., Fortune Oil")

    # ── Pack inputs ──
    st.markdown("**📦 Enter package details (up to 3 options)**")
    o1, o2, o3 = st.columns(3)
    with o1:
        pack1_qty = st.number_input("Pack1 quantity", min_value=0.0, value=500.0, step=1.0, key="comp_pack1_qty")
        pack1_unit = st.selectbox("Pack1 unit", ["g", "ml", "kg", "L"], key="comp_pack1_unit")
        pack1_price = st.number_input("Pack1 price (₹)", min_value=0.0, value=0.0, step=1.0, key="comp_pack1_price")
    with o2:
        pack2_qty = st.number_input("Pack2 quantity", min_value=0.0, value=1000.0, step=1.0, key="comp_pack2_qty")
        pack2_unit = st.selectbox("Pack2 unit", ["g", "ml", "kg", "L"], key="comp_pack2_unit")
        pack2_price = st.number_input("Pack2 price (₹)", min_value=0.0, value=0.0, step=1.0, key="comp_pack2_price")
    with o3:
        pack3_qty = st.number_input("Pack3 quantity", min_value=0.0, value=2000.0, step=1.0, key="comp_pack3_qty")
        pack3_unit = st.selectbox("Pack3 unit", ["g", "ml", "kg", "L"], key="comp_pack3_unit")
        pack3_price = st.number_input("Pack3 price (₹)", min_value=0.0, value=0.0, step=1.0, key="comp_pack3_price")

    # ── Compare button ──
    if st.button("📊 COMPARE PACKS", use_container_width=True, type="primary"):

        if not product.strip():
            st.warning("⚠️ Please enter a product name.")
            st.stop()

        # ── Process pack options ──
        pack_options = [
            (pack1_qty, pack1_unit, pack1_price),
            (pack2_qty, pack2_unit, pack2_price),
            (pack3_qty, pack3_unit, pack3_price),
        ]
        valid_options = []
        for pq, pu, pp in pack_options:
            if pq > 0 and pp > 0:
                norm = normalize_quantity(pq, pu)
                if norm > 0:
                    valid_options.append({
                        "size": f"{pq}{pu}",
                        "price": f"₹{pp}",
                        "unit_price": calculate_unit_price(pp, norm),
                        "raw_qty": norm
                    })

        if not valid_options:
            st.warning("⚠️ Please enter at least one valid pack option (quantity > 0 and price > 0).")
            st.stop()

        # ── Display results ──
        best = min(valid_options, key=lambda x: x["unit_price"])
        st.markdown("**📦 Pack Comparison Results (sorted by unit price)**")
        st.dataframe(
            pd.DataFrame(valid_options)
            .sort_values("unit_price")
            .reset_index(drop=True),
            use_container_width=True
        )
        st.success(
            f"🏆 Best value: {best['size']} at ₹{best['unit_price']:.2f} per 100g/ml "
            f"({best['price']})"
        )

        # ── Market comparison if available ──
        market_data = get_market_average(product)
        if market_data:
            market_avg = market_data["avg_price_per_100"]
            st.info(f"📊 Market average for {product}: ₹{market_avg:.2f}/100g/ml")
            if best["unit_price"] < market_avg:
                st.success("💰 This pack is below market average — great deal!")
            elif best["unit_price"] > market_avg:
                st.warning("⚠️ This pack is above market average — consider alternatives.")


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.divider()
st.caption("🛍️ DeciBuy | Real-Time Discount Verification System | "
           "Smart Shopping, Real Decisions | Research Prototype v2.0")
