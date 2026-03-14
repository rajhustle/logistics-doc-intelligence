# UltraShip Doc Intelligence

**Upload a logistics document. Ask questions about it. Extract structured shipment data. All offline — no API key required.**

🟢 **[Click here to try the live demo](#)** ← *(update this link after Streamlit deployment)*

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#)

---

## What This Is

A document intelligence system built specifically for logistics documents — Rate Confirmations, Bills of Lading (BOLs), and shipping orders.

Upload a document and the system does two things:

1. **Answer questions** about the document in plain English — with confidence scoring and guardrails that block off-topic questions
2. **Extract structured shipment data** automatically — carrier, shipper, consignee, rate, pickup/delivery dates, weight, mode, and more

No OpenAI key. No Anthropic key. No internet connection required after setup. Runs entirely on your machine using local HuggingFace embeddings.

---

## Why This Is Different From a Generic RAG Chatbot

Most RAG demos have a critical flaw — they answer anything, even questions completely unrelated to the uploaded document. Ask "what is the Sensex today?" and a generic RAG bot will hallucinate an answer from whatever it retrieved.

This system has a **guardrail layer** that detects when a question is not related to the uploaded document and blocks it — returning a clear message instead of a hallucinated answer.

Additionally most RAG systems return raw chunk text as the answer. This system has a **field extraction engine** that parses structured fields (carrier name, rate, dates) and returns clean precise values — not paragraph blobs.

These are the same reliability principles used in production AI systems where wrong answers have real consequences.

---

## Features

### 💬 Question Answering with Guardrails
- Ask any question about your uploaded document
- Token overlap guardrail — blocks questions not related to the document
- Confidence scoring — every answer shows a confidence level
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

Shows which fields were found and which are missing from the document — no silent failures.

### 📄 Document Support
- PDF
- DOCX (Word documents)
- TXT

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
      │   Guardrail Check (token overlap — blocks off-topic questions)
      │     ↓
      │   Field Extraction (FIELD_MAP pattern matching)
      │     ↓
      │   Confidence Score + Answer
      │
      └── Extraction Tab
            ↓
          Regex Engine (extract.py — 11 fields, multi-pattern fallbacks)
            ↓
          Structured JSON output
```

---

## How to Run

**Step 1 — Download the project**
- Click the green **Code** button on this GitHub page
- Click **Download ZIP**
- Unzip the folder on your computer

**Step 2 — Install Python**
- Go to [python.org](https://www.python.org/downloads/) and download Python 3.8 or higher
- During installation check **"Add Python to PATH"**

**Step 3 — Open terminal in the project folder**
- **Windows:** open the unzipped folder → click the address bar → type `cmd` → press Enter
- **Mac:** right click the folder → New Terminal at Folder

**Step 4 — Install dependencies**
```bash
pip install -r requirements.txt
pip install pdfplumber python-docx
```

**Step 5 — Run**
```bash
streamlit run app.py
```

**Step 6 — Open in browser**
- Streamlit opens `http://localhost:8501` automatically
- Upload the included `sample_document.txt` to test immediately

---

## Quick Test

A sample Rate Confirmation document is included — `sample_document.txt`.

Upload it and try these questions:
- `who is the carrier?`
- `what is the total rate?`
- `when is pickup?`
- `who is the consignee?`
- `what is the weight?`

Then click **Extract Structured Data** to see all 11 fields pulled automatically.

Try an off-topic question like `what is the gold rate today?` — watch the guardrail block it.

---

## Deploy Your Own — Free

Deploy your own instance on Streamlit Cloud in 5 minutes:

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Select your forked repo
5. Set main file path: `app.py`
6. Click Deploy

Free public URL. No server. No monthly cost.

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
| Guardrails | Token overlap scoring |
| Confidence | FAISS distance → normalized score |

**No OpenAI. No Anthropic. No API keys. No internet after setup.**

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app — upload, Q&A tab, extraction tab |
| `rag.py` | Retrieval engine — guardrails, confidence scoring, field extraction |
| `extract.py` | Regex-based structured data extractor — 11 shipment fields |
| `requirements.txt` | Dependencies |
| `sample_document.txt` | Sample Rate Confirmation for testing |

---

## Author

**Kaushal Raj** — Production Voice & Agentic AI Engineer
[GitHub](https://github.com/rajhustle) · [LinkedIn](https://linkedin.com/in/kaushal-raj-a83603380)
