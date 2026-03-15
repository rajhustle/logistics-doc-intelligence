"""
rag.py — Retrieval + Guardrails for UltraShip Doc Intelligence
"""

import re

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "and", "or", "are", "was", "were", "has", "have", "how", "when", "where",
    "who", "which", "does", "do", "it", "this", "that", "me", "my", "tell",
    "give", "about", "with", "from", "be", "by", "as", "its", "their", "name",
    "current", "today", "now", "latest", "per", "any", "all", "get", "find",
    "show", "list", "please", "can", "you", "much", "many", "some"
}

# Hard block — if any of these appear in the question, reject immediately.
# Currency names included because "dollar rate" / "rupee price" are never logistics.
NOISE_KEYWORDS = {
    "gold", "silver", "bitcoin", "crypto", "stock", "sensex", "nifty",
    "weather", "temperature", "news", "tomorrow", "yesterday",
    "market", "forex", "petrol", "diesel", "inflation", "gdp",
    "president", "prime", "minister", "election", "cricket", "movie",
    "film", "song", "recipe", "food", "restaurant", "hotel", "flight",
    "visa", "passport", "health", "disease", "covid", "vaccine",
    "python", "javascript", "programming", "code", "error", "bug",
    "dollar", "rupee", "euro", "yen", "pound", "aed", "dinar", "franc",
    "ruble", "yuan", "ringgit", "baht", "krona", "peso", "real", "lira"
}

# Core logistics domain keywords
DOMAIN_KEYWORDS = {
    "carrier", "shipment", "shipper", "consignee", "pickup", "delivery",
    "freight", "cargo", "truck", "trailer", "bol", "rate", "weight",
    "origin", "destination", "equipment", "mode", "ftl", "ltl",
    "dispatch", "logistics", "invoice", "payment", "surcharge",
    "pallet", "commodity", "hazmat", "liftgate", "appointment",
    "confirmed", "coordinator", "reference", "scac", "driver", "currency"
}

# Generic words that need a logistics anchor to be accepted
# NOTE: "rate" is intentionally excluded — "what is the rate?" is valid,
# and "gold rate" is already caught by NOISE_KEYWORDS.
AMBIGUOUS_KEYWORDS = {"price", "amount", "charge", "cost", "fee", "value"}

# Confirms logistics intent when only an ambiguous word is present
LOGISTICS_CONTEXT_WORDS = {
    "total", "freight", "base", "fuel", "accessorial", "shipment",
    "carrier", "delivery", "pickup", "shipping", "transport"
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


def _has_field_map_hit(question: str) -> bool:
    """Returns True if the question directly names a known shipment field."""
    q_lower = question.lower()
    return any(keyword in q_lower for keyword in FIELD_MAP)


def _is_domain_relevant(question: str) -> bool:
    """
    Three-layer guardrail — no LLM required.

    Layer A — Noise block:
        Hard reject on noise/currency keywords.
        Catches: "gold rate", "dollar rate", "petrol price", "sensex" etc.

    Layer B — Ambiguity check:
        "price", "cost", "charge" etc. need a logistics anchor word.
        Blocks "price of oil", "cost of living" etc.

    Layer C — Domain keyword required:
        Question must contain at least one logistics domain keyword.
    """
    q_lower = question.lower()
    q_tokens = _tokenize(question)
    raw_words = set(re.findall(r"[a-z]+", q_lower))

    # Layer A
    if raw_words & NOISE_KEYWORDS:
        return False

    # Layer B
    unambiguous_domain_hits = q_tokens & (DOMAIN_KEYWORDS - AMBIGUOUS_KEYWORDS)
    ambiguous_hits = q_tokens & AMBIGUOUS_KEYWORDS
    if ambiguous_hits and not unambiguous_domain_hits:
        if not (raw_words & LOGISTICS_CONTEXT_WORDS):
            return False

    # Layer C
    if not (q_tokens & DOMAIN_KEYWORDS):
        return False

    return True


def _parse_all_fields(text: str) -> dict:
    """Parse all 'Field Name: Value' pairs from document text."""
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

    # Layer 2: Token overlap check
    # SKIPPED if question directly names a known field (carrier, rate, pickup, etc.)
    # WHY: short precise questions like "who is the carrier?" tokenize to just
    # {'carrier'} — 1 token — which would false-block at threshold < 2.
    # A direct FIELD_MAP hit is stronger proof of relevance than token count.
    if all_doc_text and not _has_field_map_hit(question):
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
