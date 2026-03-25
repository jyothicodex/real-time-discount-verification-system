# utils/ai_engine.py
# ─────────────────────────────────────────────
# AI explanation engine for DeciBuy
# Generates structured verdicts using local Ollama LLM
# Output is always: Summary | Reason | Suggestion
# ─────────────────────────────────────────────

import subprocess
import json

# ── Change this to your installed Ollama model ──
MODEL_NAME = "qwen3-vl:4b"  # e.g., llama3.2, mistral, gemma3


def _run_ollama(prompt: str, timeout: int = 90) -> str:
    """
    Internal function: call Ollama CLI and return raw text output.
    Uses shell=True for Windows compatibility.
    """
    try:
        # Escape double quotes inside the prompt to avoid shell issues
        safe_prompt = prompt.replace('"', "'").replace('\n', ' ').strip()
        command = f'ollama run {MODEL_NAME} "{safe_prompt}"'

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0 and result.stderr.strip():
            return f"Ollama error: {result.stderr.strip()}"

        return result.stdout.strip() if result.stdout.strip() else "AI did not return a response."

    except subprocess.TimeoutExpired:
        return "AI request timed out. Please try again."
    except FileNotFoundError:
        return "Ollama not installed. Please install from https://ollama.com"
    except Exception as e:
        return f"AI service unavailable: {str(e)}"


def get_structured_verdict(
    product_name: str,
    unit_price: float,
    market_avg: float,
    deal_score: int,
    verdict_label: str,
    flags: dict = None,
    user_type: str = "buyer"
) -> dict:
    """
    Generate a structured AI verdict with three sections:
    - Summary: one-sentence overall assessment
    - Reason: why this deal is good/bad
    - Suggestion: what the buyer should do

    Returns a dict with keys: summary, reason, suggestion
    Falls back to rule-based text if AI is unavailable.
    """

    # Build a rich context string for the LLM
    flag_text = ""
    if flags:
        if flags.get("inflated_mrp", {}).get("detected"):
            flag_text += "⚠ Inflated MRP detected. "
        if flags.get("fake_percentage", {}).get("detected"):
            flag_text += "⚠ Fake discount percentage detected. "
        if flags.get("shrinkflation", {}).get("detected"):
            flag_text += "⚠ Shrinkflation detected (quantity reduced). "

    prompt = f"""
You are a smart shopping assistant in India. Analyze this product deal and respond ONLY with a JSON object with exactly 3 keys: "summary", "reason", "suggestion". No extra text.

Product: {product_name}
Unit Price: Rs {unit_price:.2f} per 100g/ml
Market Average: Rs {market_avg:.2f} per 100g/ml (0 means unknown)
Deal Score: {deal_score}/100
Verdict: {verdict_label}
Red Flags: {flag_text if flag_text else 'None'}
Buyer Type: {user_type}

Rules:
- summary: 1 sentence verdict in simple English
- reason: 2-3 sentences explaining why this score and verdict
- suggestion: 1-2 sentences on what the buyer should do

Respond ONLY with valid JSON, example:
{{"summary": "...", "reason": "...", "suggestion": "..."}}
"""

    raw = _run_ollama(prompt)

    # Try to parse JSON from LLM output
    try:
        # Find the first { and last } to extract JSON even if there's extra text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            return {
                "summary":    parsed.get("summary", "—"),
                "reason":     parsed.get("reason", "—"),
                "suggestion": parsed.get("suggestion", "—"),
                "ai_available": True
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Fallback: Rule-based structured response (no AI needed) ──
    return _rule_based_verdict(product_name, unit_price, market_avg,
                                deal_score, verdict_label, flags)


def _rule_based_verdict(product_name, unit_price, market_avg,
                         deal_score, verdict_label, flags) -> dict:
    """
    Deterministic fallback when AI is unavailable.
    Generates structured text from deal score and flags.
    """
    if deal_score >= 80:
        summary    = f"{product_name} is an excellent deal — don't miss it!"
        reason     = (f"At ₹{unit_price:.2f}/100g, it's well below the market average "
                      f"of ₹{market_avg:.2f}/100g. Score: {deal_score}/100.")
        suggestion = "Go ahead and buy. Consider buying in bulk if possible."

    elif deal_score >= 60:
        summary    = f"{product_name} is a decent deal worth buying."
        reason     = (f"Unit price ₹{unit_price:.2f}/100g is close to the market average "
                      f"₹{market_avg:.2f}/100g. Score: {deal_score}/100.")
        suggestion = "Buy it, but no need to stock up."

    elif deal_score >= 40:
        summary    = f"{product_name} is priced fairly but no real savings."
        reason     = (f"Unit price ₹{unit_price:.2f}/100g is at or slightly above market. "
                      f"Score: {deal_score}/100.")
        suggestion = "Buy only if needed. Check other stores or brands."

    elif deal_score >= 20:
        summary    = f"{product_name} is overpriced — avoid if possible."
        reason     = (f"At ₹{unit_price:.2f}/100g vs market avg ₹{market_avg:.2f}/100g, "
                      f"you're overpaying. Score: {deal_score}/100.")
        suggestion = "Avoid this deal. Try a different store or brand."

    else:
        flag_detail = ""
        if flags:
            if flags.get("inflated_mrp", {}).get("detected"):
                flag_detail += " MRP looks artificially inflated."
            if flags.get("shrinkflation", {}).get("detected"):
                flag_detail += " Quantity has been quietly reduced."
        summary    = f"{product_name} appears to be a fake or misleading discount."
        reason     = f"Multiple red flags detected.{flag_detail} Score: {deal_score}/100."
        suggestion = "Do NOT buy. Report the misleading offer to the store."

    return {
        "summary": summary,
        "reason": reason,
        "suggestion": suggestion,
        "ai_available": False  # indicates fallback was used
    }


def get_cart_verdict(cart_items: list) -> str:
    """
    Generate an overall cart analysis.

    Args:
        cart_items: list of dicts, each with:
                    {name, unit_price, market_avg, deal_score, verdict}

    Returns:
        Plain text cart summary from AI (or fallback text).
    """
    if not cart_items:
        return "No items to analyze."

    lines = []
    for item in cart_items:
        lines.append(
            f"- {item['name']}: ₹{item['unit_price']:.2f}/100g | "
            f"Score: {item['deal_score']}/100 | {item['verdict']}"
        )

    cart_text = "\n".join(lines)
    prompt = f"""
You are a shopping advisor. Here is a cart analysis:
{cart_text}

Give a short, helpful cart summary:
1. Which items are worth buying?
2. Which items to avoid?
3. One overall money-saving tip.

Keep it under 100 words. Be direct and friendly.
"""
    return _run_ollama(prompt)
