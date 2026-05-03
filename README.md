# 🇲🇾 VLM Document Intelligence — Malaysian Documents

> Extract structured data from Malaysian documents using **Qwen2.5-VL** and **EasyOCR**. The only open-source document intelligence pipeline built specifically for Malaysian document formats.

[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL-blue)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/aliyaalias19/my-document-intelligence)

---

## Why This Project Exists

Every Malaysian fintech, insurtech, bank, and government agency processes the same set of documents — MyKad, SSM certificates, LHDN EA forms, payslips, bank statements — and most do it manually or with expensive proprietary OCR solutions.

Existing document intelligence tools (AWS Textract, Azure Form Recognizer, Google Document AI) are:
- Expensive at scale
- Not fine-tuned for Malaysian document formats
- Unaware of Malaysian-specific fields (IC number structure, SSM registration format, MYR amounts)
- Unable to handle Bahasa Malaysia + English code-switching

This project fills that gap with a fully open-source, locally-runnable pipeline.

---

## Supported Document Types

| Document | Key Fields Extracted | Language |
|---|---|---|
| 🪪 **MyKad** (Malaysian IC) | IC number, name, DOB, gender, state, religion, address | BM |
| 🪪 **Singapore NRIC** | NRIC number, name, DOB, sex, race | EN |
| 📋 **SSM Business Registration** | Reg number, business name, type, owner, MSIC code, status | BM/EN |
| 🧾 **Tax Invoice** | Invoice no, seller, buyer, line items, SST, total (MYR) | BM/EN |
| 📑 **LHDN EA Form** | Assessment year, employer, gross salary, PCB, EPF | BM/EN |
| 💰 **Payslip** | Basic salary, allowances, EPF, SOCSO, PCB, net pay | BM/EN |
| 🏦 **Bank Statement** | Bank, account holder, account no, balances, transactions | EN |
| 💡 **Utility Bill** | Provider (TNB/Unifi/Maxis), account, amounts, kWh | BM/EN |
| 💼 **EPF/KWSP Statement** | Member, Account 1 & 2 balances, dividends | BM/EN |

---

## Quick Start

```bash
git clone https://github.com/aliyaalias19/vlm-document-intelligence
cd vlm-document-intelligence
pip install -r requirements.txt

# Verify setup (< 5 seconds, no GPU needed)
python quickstart.py
```

### Run the Demo (Gradio)

```bash
pip install easyocr gradio pillow
python demo/app.py
# Open: http://localhost:7860
```

### Start the API

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### API Usage

```bash
# Extract a document
curl -X POST http://localhost:8000/extract \
  -F "file=@mykad.jpg" \
  | python -m json.tool
```

```python
import requests

with open("invoice.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/extract",
        files={"file": f},
    )

result = response.json()
print(f"Type: {result['document_type']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Data: {result['extracted_data']}")
```

---

## Sample Output

### MyKad (Malaysian IC)
```json
{
  "document_type": "mykad",
  "confidence": 0.94,
  "extracted_data": {
    "ic_number": "900101-14-5123",
    "full_name": "ALIYA BINTI ALIAS",
    "date_of_birth": "01/01/1990",
    "gender": "female",
    "nationality": "WARGANEGARA",
    "religion": "Islam",
    "state_of_birth": "Selangor",
    "age": 35
  },
  "model_used": "Qwen2.5-VL-7B",
  "latency_ms": 1840
}
```

### SSM Business Registration
```json
{
  "document_type": "ssm_registration",
  "confidence": 0.91,
  "extracted_data": {
    "registration_number": "202401234567",
    "business_name": "ALIYA TECH ENTERPRISE",
    "business_type": "Sole Proprietorship",
    "registration_date": "15/03/2024",
    "owner_name": "ALIYA BINTI ALIAS",
    "owner_ic": "900101-14-5123",
    "business_activity": "IT Consulting and AI Development",
    "msic_code": "62090",
    "status": "Active"
  },
  "model_used": "Qwen2.5-VL-7B",
  "latency_ms": 2100
}
```

---

## Architecture

```
Document Image (JPG/PNG/PDF)
        ↓
┌───────────────────────────────────────────────────────┐
│               STAGE 1: TEXT EXTRACTION                │
│                                                       │
│  Primary: Qwen2.5-VL (GPU)                           │
│  ├── Understands document layout                     │
│  ├── Reads BM + EN + mixed text                      │
│  └── Outputs structured JSON directly                │
│                                                       │
│  Fallback: EasyOCR (CPU, no GPU needed)              │
│  ├── BM + EN language models                         │
│  └── Raw text → Stage 2 extraction                  │
└───────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────┐
│           STAGE 2: DOCUMENT CLASSIFICATION            │
│                                                       │
│  DocumentClassifier (keyword + regex)                │
│  ├── 9 document types                                │
│  ├── Malaysian-specific signatures                   │
│  └── Confidence scoring                             │
└───────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────┐
│           STAGE 3: FIELD EXTRACTION                  │
│                                                       │
│  VLM path: Structured prompt → JSON → parse          │
│  OCR path: Regex extractors per document type        │
│  ├── IC number parsing (YYMMDD-SS-GGGG)              │
│  ├── MYR amount detection                            │
│  ├── SSM registration format                        │
│  └── Malaysian date formats                         │
└───────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────┐
│           STAGE 4: PYDANTIC VALIDATION               │
│                                                       │
│  Schema per document type                            │
│  ├── Type coercion (str → float for amounts)         │
│  ├── IC number normalisation                         │
│  └── Null handling for missing fields                │
└───────────────────────────────────────────────────────┘
        ↓
   Structured JSON Output
```

---

## Malaysian-Specific Features

### IC Number Parsing
```python
# Auto-detects format YYMMDD-SS-GGGG
# Extracts: DOB, gender, state of birth
ic = "901201-14-5121"
# → DOB: 01/12/1990
# → Gender: male (odd last digit)
# → State: Selangor (code 14)
# → Age: 35 (calculated)
```

### SSM Registration Format
```python
# Handles both old and new formats
old_format = "123456-V"         # Pre-2016
new_format = "202401234567"     # Post-2016 (12-digit: YYYYMMXXXXXXX)
```

### MYR Amount Extraction
```python
# Detects Malaysian Ringgit amounts in various formats
"RM 1,500.00"   → 1500.0
"Rm1500"        → 1500.0
"RM1,500"       → 1500.0
```

---

## Model Details

### Qwen2.5-VL (Primary — GPU Recommended)

Qwen2.5-VL provides robust structured data extraction from invoices, forms, and tables, excelling in document and diagram understanding. The 7B model matches GPT-4o in document understanding benchmarks.

| Size | VRAM Required | Latency | Use Case |
|---|---|---|---|
| 3B | ~4GB | ~5s/page | CPU / laptop GPU |
| 7B | ~8GB | ~2s/page | T4 GPU (Colab free) |
| 72B | ~40GB | ~1s/page | A100 (Colab Pro+) |

### EasyOCR (Fallback — CPU)
- Languages: Malay (`ms`) + English (`en`)
- No GPU required
- ~3-5s per page on CPU
- Lower accuracy than VLM but fully free

---

## Project Structure

```
vlm-document-intelligence/
│
├── extractors/
│   ├── classifier.py        ← Document type detection (keyword + regex)
│   ├── vlm_extractor.py     ← Qwen2.5-VL extraction pipeline
│   └── regex_extractors.py  ← Per-type regex fallback extractors
│
├── models/
│   └── document_schemas.py  ← Pydantic schemas for all 9 document types
│
├── api/
│   └── main.py              ← FastAPI: POST /extract, GET /supported-types
│
├── demo/
│   └── app.py               ← Gradio demo for Hugging Face Spaces
│
├── quickstart.py            ← Verify setup (no GPU needed)
└── requirements.txt
```

---

## Privacy Notice

This system is designed for processing documents in **enterprise / internal systems**. When deploying:

- Do not log raw document content to external services
- Use the local/self-hosted deployment for sensitive documents
- The demo on Hugging Face Spaces uses **dummy/sample documents only**
- Real personal documents (MyKad, bank statements) should only be processed in your own controlled infrastructure

---

## Citation

```bibtex
@misc{alias2026vlmdocument,
  title  = {VLM Document Intelligence: Malaysian Document Extraction Pipeline},
  author = {Alias, Aliya},
  year   = {2026},
  url    = {https://github.com/aliyaalias19/vlm-document-intelligence}
}
```

---

## 👤 About

Built by **Aliya Alias** — AI Engineer, Kuala Lumpur, Malaysia.
MSc Artificial Intelligence, University of Malaya (CGPA 3.73).
Specialising in production LLM systems and computer vision.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aliyaalias-blue)](https://linkedin.com/in/aliyaalias)
[![GitHub](https://img.shields.io/badge/GitHub-aliyaalias19-black)](https://github.com/aliyaalias19)

*MIT License.*
