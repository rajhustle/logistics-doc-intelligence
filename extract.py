"""
extract.py — Structured field extraction for UltraShip Doc Intelligence

Uses regex patterns to extract shipment fields from document text.
Returns JSON with nulls for missing fields — no LLM or API key required.
"""

import re


def _find(pattern: str, text: str, flags=re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    if match:
        return match.group(1).strip()
    return None


def _extract_currency(rate_str: str) -> str | None:
    """Detect currency symbol from rate string."""
    if not rate_str:
        return None
    if "$" in rate_str:
        return "USD"
    if "£" in rate_str:
        return "GBP"
    if "€" in rate_str:
        return "EUR"
    if "aed" in rate_str.lower():
        return "AED"
    return None


def extract_from_text(text: str) -> dict:
    """
    Extract all required shipment fields from raw document text.
    Returns dict with nulls for any missing field.
    """

    result = {
        "shipment_id": None,
        "shipper": None,
        "consignee": None,
        "pickup_datetime": None,
        "delivery_datetime": None,
        "equipment_type": None,
        "mode": None,
        "rate": None,
        "currency": None,
        "weight": None,
        "carrier_name": None,
    }

    # Shipment ID — matches patterns like SHP-12345, SHIP-001, BOL#12345
    result["shipment_id"] = (
        _find(r"shipment\s*(?:id|#|no|number)[:\s#]+([A-Z0-9\-]+)", text)
        or _find(r"(?:bol|pro|order)\s*(?:#|no|number)?[:\s]+([A-Z0-9\-]+)", text)
    )

    # Shipper
    result["shipper"] = (
        _find(r"shipper[:\s]+([^\n,]+)", text)
        or _find(r"ship\s*from[:\s]+([^\n,]+)", text)
        or _find(r"origin\s*(?:company)?[:\s]+([^\n,]+)", text)
    )

    # Consignee
    result["consignee"] = (
        _find(r"consignee[:\s]+([^\n,]+)", text)
        or _find(r"ship\s*to[:\s]+([^\n,]+)", text)
        or _find(r"deliver\s*to[:\s]+([^\n,]+)", text)
        or _find(r"customer\s*(?:name)?[:\s]+([^\n,]+)", text)
    )

    # Pickup datetime
    result["pickup_datetime"] = (
        _find(r"pickup\s*(?:date(?:time)?|time)?[:\s]+([^\n]+)", text)
        or _find(r"pick\s*up\s*(?:date)?[:\s]+([^\n]+)", text)
        or _find(r"ship\s*date[:\s]+([^\n]+)", text)
    )

    # Delivery datetime
    result["delivery_datetime"] = (
        _find(r"delivery\s*(?:date(?:time)?|time)?[:\s]+([^\n]+)", text)
        or _find(r"deliver\s*(?:by|date)?[:\s]+([^\n]+)", text)
        or _find(r"estimated\s*(?:arrival|delivery)[:\s]+([^\n]+)", text)
    )

    # Equipment type
    result["equipment_type"] = (
        _find(r"equipment\s*(?:type)?[:\s]+([^\n,]+)", text)
        or _find(r"trailer\s*(?:type)?[:\s]+([^\n,]+)", text)
        or _find(r"\b(flatbed|reefer|dry\s*van|step\s*deck|tanker|box\s*truck)\b", text)
    )

    # Mode
    result["mode"] = (
        _find(r"\bmode[:\s]+([^\n,]+)", text)
        or _find(r"\b(ftl|ltl|partial|intermodal|air|ocean|rail)\b", text)
    )

    # Rate (capture number + possible symbol)
    rate_raw = (
        _find(r"rate[:\s]+(\$?[\d,]+(?:\.\d{1,2})?)", text)
        or _find(r"(?:total\s*)?(?:freight\s*)?charge[:\s]+(\$?[\d,]+(?:\.\d{1,2})?)", text)
        or _find(r"(?:amount|price)[:\s]+(\$?[\d,]+(?:\.\d{1,2})?)", text)
    )
    if rate_raw:
        result["rate"] = rate_raw.replace("$", "").strip()
        result["currency"] = _extract_currency(rate_raw) or _extract_currency(text[:500])

    # Weight
    result["weight"] = _find(
        r"(?:total\s*)?weight[:\s]+([\d,]+(?:\.\d+)?\s*(?:lbs?|kg|pounds?|kilograms?)?)",
        text
    )

    # Carrier name
    result["carrier_name"] = (
        _find(r"carrier\s*(?:name)?[:\s]+([^\n,]+)", text)
        or _find(r"(?:trucking|transport(?:ation)?)\s*(?:company)?[:\s]+([^\n,]+)", text)
    )

    # Clean up whitespace in all values
    for key, val in result.items():
        if isinstance(val, str):
            result[key] = val.strip()

    return result


def extract_structured_data(vectorstore) -> dict:
    """
    Pull all document chunks from vectorstore and run extraction.
    Called from app.py when user clicks Extract button.
    """
    try:
        results = vectorstore.similarity_search(
            "shipment carrier rate pickup delivery consignee shipper weight", k=20
        )
        full_text = "\n".join([doc.page_content for doc in results])
    except Exception as e:
        return {"error": f"Could not retrieve document: {str(e)}"}

    if not full_text.strip():
        return {"error": "Document appears to be empty."}

    return extract_from_text(full_text)
