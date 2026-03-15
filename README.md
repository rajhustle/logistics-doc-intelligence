# UltraShip Doc Intelligence

**Upload a logistics document. Ask questions about it. Extract structured shipment data. No API key required.**

🟢 **[Click here to try the live demo](YOUR_STREAMLIT_URL_HERE)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)

> App may take 30–60 seconds to wake if inactive. Upload `sample_document.txt` included in this repo to test immediately.

---

## What This Is

A document intelligence system built for logistics documents — Rate Confirmations, Bills of Lading (BOLs), and shipping orders.

Upload a document and the system does two things:

1. **Answer questions** in plain English — with confidence scoring and a guardrail layer that blocks off-topic questions
2. **Extract structured shipment data** automatically — carrier, shipper, consignee, rate, pickup/delivery dates, weight, mode, and more

No OpenAI key. No Anthropic key. No internet required after setup. Runs entirely on local HuggingFace embeddings.

---

## Why This Is Different From a Generic RAG Chatbot

Most RAG demos answer anything — even questions completely unrelated to the uploaded document. Ask a generic bot "what is the gold rate today?" and it returns a hallucinated answer from whatever chunk it retrieved.

This system has a **deterministic guardrail layer** — no LLM judge, no API call, zero cost — that detects when a question cannot be answered from the uploaded document and blocks it cleanly.

Try it: upload the sample document, ask `what is the gold rate?` — blocked. Ask `what is the bronze rate?` — blocked. Ask `who is the carrier?` — answered precisely.

The guardrail works by grounding every question in the actual document. Every content word in your question must exist in the document. Words after "of/for/about" must exist as actual field labels. If anything is foreign to the document — it blocks.

This is the same reliability thinking used in production AI systems where wrong answers have real consequences.

---

## Features

### 💬 Question Answering with Guardrails
- Ask any question about your uploaded document
- **Document-grounded guardrail** — blocks questions whose content words don't exist in the document
- Confidence scoring on every answer:
  - 🟢 Green — high confidence (≥ 0.65)
  - 🟡 Orange — medium confidence (≥ 0.45)
  - 🔴 Red — low confidence, treat with caution
- Source chunk display — see exactly which part of the document the answer came from

### 🗂️ Structured Data Extraction
Automatically extracts 11 shipment fields from any logistics document:

| Field | Example |
|-------|---------|
| Shipment ID | SHP-45892 |
| Shipper | Global Parts Manufacturing Inc. |
| Consignee | ABC Logistics & Distribution |
| Pickup Date | 10 Feb 2026, 08:00 AM |
| Delivery Date | 14 Feb 2026, 05:00 PM |
| Equipment Type | Dry Van 53ft |
| Mode | FTL |
| Rate | $1,500.00 |
| Currency | USD |
| Weight | 18,500 lbs |
| Carrier Name | FedEx Freight |

Shows found fields and missing fields explicitly — no silent failures.

### 📄 Document Support
- PDF
- DOCX (Word documents)
- TXT

---

## Guardrail — How It Works

The guardrail is entirely deterministic. No LLM. No API. Pure logic.

```
Question enters
      ↓
Rule 1: Every content word in the question must exist in the document
        "gold rate" → "gold" not in document → BLOCKED
        "bronze rate" → "bronze" not in document → BLOCKED
        "carrier pigeon" → "pigeon" not in document → BLOCKED
      ↓
Rule 2: Words after "of/for/about" must be actual field labels in the document
        "weight of a truck" → "truck" is not a queryable field → BLOCKED
        "delivery for pizza" → "pizza" not in document → BLOCKED
        "weight of the shipment" → "shipment" IS a field → PASSES
      ↓
Answer returned with confidence score + source chunk
```

---

## Architecture

```
Document Upload
      ↓
Text Extraction (pdfplumber / python-docx / plain text)
      ↓
Chunking (RecursiveCharacterTextSplitter — 500 chars, 50 overlap)
      ↓
Embedding (HuggingFace all-MiniLM-L6-v2 — runs locally, no API key)
      ↓
FAISS Vector Store (in-memory)
      ↓
      ├── Q&A Tab
      │     ↓
      │   Similarity Search (top 3 chunks)
      │     ↓
      │   Document-Grounded Guardrail (rag.py)
      │     ↓
      │   Field Extraction (FIELD_MAP pattern matching)
      │     ↓
      │   Confidence Score + Answer + Source Chunk
      │
      └── Extraction Tab
            ↓
          Regex Engine (extract.py — 11 fields, multi-pattern fallbacks)
            ↓
          Structured JSON output — found fields + missing fields
```

---

## Quick Test

Upload `sample_document.txt` (included) and try:

**These should be answered:**
- `who is the carrier?`
- `what is the total rate?`
- `when is pickup?`
- `who is the consignee?`
- `what is the weight?`

**These should be blocked by the guardrail:**
- `what is the gold rate today?`
- `what is the bronze rate?`
- `what is the carrier pigeon?`
- `who is the prime minister?`

Then click **Extract Structured Data** — all 11 fields pulled automatically.

---

## How to Run Locally

**Step 1 — Clone or download**
```bash
git clone https://github.com/rajhustle/logistics-doc-intelligence
cd logistics-doc-intelligence
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501` automatically.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS (in-memory) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| PDF extraction | pdfplumber |
| DOCX extraction | python-docx |
| Field extraction | Custom regex engine (no LLM) |
| Guardrails | Document-grounded deterministic logic (no LLM, no API) |
| Confidence | FAISS distance → normalized score |

**No OpenAI. No Anthropic. No API keys. No internet after setup.**

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app — upload, Q&A tab, extraction tab |
| `rag.py` | Retrieval engine — document-grounded guardrail, confidence scoring, field extraction |
| `extract.py` | Regex-based structured data extractor — 11 shipment fields |
| `requirements.txt` | Dependencies |
| `sample_document.txt` | Sample Rate Confirmation for immediate testing |

---

## Author

**Kaushal Raj** — Production Voice & Agentic AI Engineer
[GitHub](https://github.com/rajhustle) · [LinkedIn](https://linkedin.com/in/kaushal-raj-a83603380)
