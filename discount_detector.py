# utils/discount_detector.py
# ─────────────────────────────────────────────
# Smart discount analysis engine for DeciBuy
# Detects: fake discounts, shrinkflation, inflated MRP tricks
# ─────────────────────────────────────────────


def detect_inflated_mrp(mrp: float, avg_market_price: float) -> dict:
    """
    Detect if the MRP has been artificially inflated before applying discount.

    Logic: If MRP is significantly higher than the known market average,
    the discount shown is misleading — it's just going back to normal price.

    Args:
        mrp: The Maximum Retail Price printed on the product
        avg_market_price: Known fair market price per unit

    Returns:
        dict with 'detected' (bool) and 'reason' (str)
    """
    if avg_market_price <= 0:
        return {"detected": False, "reason": "No market data to compare."}

    inflation_ratio = mrp / avg_market_price

    if inflation_ratio > 1.35:
        return {
            "detected": True,
            "reason": (
                f"MRP (₹{mrp:.2f}) appears inflated by ~{((inflation_ratio-1)*100):.0f}% "
                f"above market average (₹{avg_market_price:.2f}). "
                "Discount shown is misleading — price is just returning to normal."
            )
        }
    return {"detected": False, "reason": "MRP appears reasonable vs market average."}


def detect_shrinkflation(prev_qty: float, curr_qty: float,
                          prev_price: float, curr_price: float) -> dict:
    """
    Detect shrinkflation: quantity reduced while price stays the same or increases.

    Classic trick: old pack was 500g, new pack is 450g at the same ₹120 label.

    Args:
        prev_qty: Previous quantity in grams/ml
        curr_qty: Current quantity in grams/ml
        prev_price: Previous price in ₹
        curr_price: Current price in ₹

    Returns:
        dict with 'detected' (bool), 'qty_change_pct' (float), 'price_change_pct' (float),
        'effective_price_hike_pct' (float), 'reason' (str)
    """
    if prev_qty <= 0 or prev_price <= 0:
        return {"detected": False, "reason": "Insufficient data for shrinkflation check."}

    qty_change_pct   = ((curr_qty - prev_qty) / prev_qty) * 100
    price_change_pct = ((curr_price - prev_price) / prev_price) * 100

    # Effective hike = unit price change
    prev_unit = (prev_price / prev_qty) * 100
    curr_unit = (curr_price / curr_qty) * 100
    effective_hike_pct = ((curr_unit - prev_unit) / prev_unit) * 100

    # Shrinkflation: quantity dropped ≥3% AND effective unit price went up
    if qty_change_pct <= -3 and effective_hike_pct > 0:
        return {
            "detected": True,
            "qty_change_pct": round(qty_change_pct, 1),
            "price_change_pct": round(price_change_pct, 1),
            "effective_price_hike_pct": round(effective_hike_pct, 1),
            "reason": (
                f"⚠️ SHRINKFLATION DETECTED! Quantity reduced by {abs(qty_change_pct):.1f}% "
                f"but effective unit price increased by {effective_hike_pct:.1f}%. "
                f"You are paying more for less product."
            )
        }

    return {
        "detected": False,
        "qty_change_pct": round(qty_change_pct, 1),
        "price_change_pct": round(price_change_pct, 1),
        "effective_price_hike_pct": round(effective_hike_pct, 1),
        "reason": "No shrinkflation detected."
    }


def detect_fake_percentage_discount(original_price: float,
                                     discounted_price: float,
                                     claimed_discount_pct: float) -> dict:
    """
    Verify if the discount percentage shown matches actual price reduction.

    Trick: A product says "30% OFF" but actual reduction is only 10%.

    Args:
        original_price: Price before discount (MRP)
        discounted_price: Price after discount (selling price)
        claimed_discount_pct: Discount % advertised on the product/app

    Returns:
        dict with 'detected' (bool), 'actual_discount_pct' (float), 'reason' (str)
    """
    if original_price <= 0:
        return {"detected": False, "actual_discount_pct": 0.0,
                "reason": "No original price provided."}

    actual_discount_pct = ((original_price - discounted_price) / original_price) * 100

    # Allow ±3% tolerance (rounding on labels is common)
    discrepancy = abs(actual_discount_pct - claimed_discount_pct)

    if claimed_discount_pct > 0 and discrepancy > 3:
        return {
            "detected": True,
            "actual_discount_pct": round(actual_discount_pct, 1),
            "reason": (
                f"Claimed discount: {claimed_discount_pct:.0f}% | "
                f"Actual discount: {actual_discount_pct:.1f}%. "
                f"Discrepancy of {discrepancy:.1f}% detected — misleading label!"
            )
        }

    return {
        "detected": False,
        "actual_discount_pct": round(actual_discount_pct, 1),
        "reason": f"Discount label is accurate. Actual discount: {actual_discount_pct:.1f}%"
    }


def full_discount_audit(
    product_name: str,
    mrp: float,
    selling_price: float,
    claimed_discount_pct: float,
    curr_qty: float,
    prev_qty: float = 0,
    prev_price: float = 0,
    market_avg_unit: float = 0
) -> dict:
    """
    Run all three discount checks and return a consolidated audit report.

    Returns:
        dict containing results from all three detectors + overall fake_flag
    """
    # 1. Inflated MRP check (compare MRP to market avg, both in unit price terms)
    # We compare per-unit prices here: mrp per 100g vs market avg per 100g
    mrp_unit = (mrp / curr_qty) * 100 if curr_qty > 0 else 0
    inflated = detect_inflated_mrp(mrp_unit, market_avg_unit) if market_avg_unit > 0 \
               else {"detected": False, "reason": "No market data available."}

    # 2. Fake percentage check
    fake_pct = detect_fake_percentage_discount(mrp, selling_price, claimed_discount_pct)

    # 3. Shrinkflation check
    shrink = detect_shrinkflation(prev_qty, curr_qty, prev_price, selling_price) \
             if prev_qty > 0 and prev_price > 0 \
             else {"detected": False, "reason": "No previous purchase data for comparison."}

    # Overall fake flag: if ANY red flag is found
    any_fake = inflated["detected"] or fake_pct["detected"] or shrink["detected"]

    return {
        "inflated_mrp": inflated,
        "fake_percentage": fake_pct,
        "shrinkflation": shrink,
        "any_fake_flag": any_fake,
        "flags_count": sum([inflated["detected"], fake_pct["detected"], shrink["detected"]])
    }
