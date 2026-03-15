"""
rag.py — Retrieval + Guardrails for UltraShip Doc Intelligence
"""

import re

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "and", "or", "are", "was", "were", "has", "have", "how", "when", "where",
    "who", "which", "does", "do", "it", "this", "that", "me", "my", "tell",
    "give", "about", "with", "from", "be", "by", "as", "its", "their", "name"
}

# Words that MUST appear in the question for it to be domain-relevant
# If question contains only generic words + one of these, it should still be blocked
NOISE_KEYWORDS = {
    "gold", "silver", "bitcoin", "crypto", "stock", "sensex", "nifty",
    "weather", "temperature", "news", "today", "tomorrow", "yesterday",
    "price", "market", "forex", "dollar", "rupee", "euro", "petrol"
}

# Core logistics domain keywords — question must overlap with these
DOMAIN_KEYWORDS = {
    "carrier", "shipment", "shipper", "consignee", "pickup", "delivery",
    "freight", "cargo", "truck", "trailer", "bol", "rate", "weight",
    "origin", "destination", "equipment", "mode", "ftl", "ltl",
    "dispatch", "logistics", "invoice", "payment", "surcharge"
}

FIELD_MAP = {
    "carrier":      ["carrier name", "carrier"],
    "rate":         ["total rate", "rate", "freight charge", "base rate", "amount"],
    "pickup":       ["pickup date", "pick up date", "ship date"],
    "delivery":     ["delivery date", "deliver by", "estimated arrival"],
    "consignee":    ["consignee", "ship to", "deliver to", "customer name"],
    "shipper":      ["shipper", "ship from", "origin company"],
    "shipment":     ["shipment id", "shipment no", "bol number", "bol", "pro number"],
    "weight":       ["weight"],
    "origin":       ["origin"],
    "destination":  ["destination"],
    "equipment":    ["equipment type", "trailer type"],
    "mode":         ["mode"],
    "email":        ["contact email", "email"],
    "currency":     ["currency"],
    "payment":      ["payment terms"],
}


def _tokenize(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _token_overlap(question: str, document_text: str) -> int:
    return len(_tokenize(question) & _tokenize(document_text))


def _is_domain_relevant(question: str) -> bool:
    """
    Two-layer check:
    1. If question contains any noise keyword (gold, weather, etc) → block
    2. If question has no overlap with core logistics domain keywords → block
    """
    q_tokens = _tokenize(question)

    # Block if noise keyword present
    if q_tokens & NOISE_KEYWORDS:
        return False

    # Block if no logistics domain keyword present
    if not (q_tokens & DOMAIN_KEYWORDS):
        return False

    return True


def _parse_all_fields(text: str) -> dict:
    """
    Parse ALL 'Field Name: Value' pairs from document.
    """
    pattern = r"^([A-Za-z][A-Za-z\s]{1,30}?)\s*[:\-]\s*(.+)$"
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            label = match.group(1).strip().lower()
            value = match.group(2).strip()
            fields[label] = value
    return fields


def _extract_value(question: str, chunk: str, all_doc_text: str = "") -> str:
    """Extract specific field value. Search chunk first, then full doc."""
    q_lower = question.lower()

    chunk_fields = _parse_all_fields(chunk)
    doc_fields   = _parse_all_fields(all_doc_text) if all_doc_text else {}

    for keyword, labels in FIELD_MAP.items():
        if keyword in q_lower:
            for label in labels:
                if label.lower() in chunk_fields:
                    return chunk_fields[label.lower()]
                if label.lower() in doc_fields:
                    return doc_fields[label.lower()]

    return chunk.strip().splitlines()[0]


def ask_question(question: str, vectorstore, all_doc_text: str = "") -> dict:

    if not question or not question.strip():
        return {"answer": "Please enter a question.", "source": None, "confidence": 0.0, "guardrail": "empty_question"}

    results = vectorstore.similarity_search_with_score(question, k=3)

    if not results:
        return {"answer": "Not found in document.", "source": None, "confidence": 0.0, "guardrail": "no_results"}

    best_doc, best_score = results[0]
    confidence = round(1 / (1 + best_score), 3)

    # Layer 1: Domain relevance check
    if not _is_domain_relevant(question):
        return {
            "answer": "This question is not related to the uploaded document.",
            "source": None,
            "confidence": confidence,
            "guardrail": "domain_check"
        }

    # Layer 2: Token overlap with document
    if all_doc_text:
        overlap = _token_overlap(question, all_doc_text)
        if overlap < 2:
            return {
                "answer": "This question is not related to the uploaded document.",
                "source": None,
                "confidence": confidence,
                "guardrail": "token_overlap"
            }

    answer = _extract_value(question, best_doc.page_content, all_doc_text)

    return {
        "answer": answer,
        "source": best_doc.page_content.strip(),
        "confidence": confidence,
        "guardrail": None
    }
