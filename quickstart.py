"""
Quick Start — VLM Document Intelligence
=========================================
Verify setup using the regex-only pipeline (no GPU needed).

Usage:
    python quickstart.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("🇲🇾 VLM Document Intelligence — Quick Start")
print("=" * 55)

# ── Step 1: Imports ───────────────────────────────────────────────────────────
print("\n📦 Checking imports...")
try:
    from models.document_schemas import (
        MyKadResult, MalaysianInvoiceResult, SSMRegistrationResult,
        DOCUMENT_SCHEMAS, DOCUMENT_DESCRIPTIONS
    )
    from extractors.classifier import DocumentClassifier
    from extractors.regex_extractors import extract_with_regex
    print(f"   ✅ All imports successful")
    print(f"   ✅ {len(DOCUMENT_SCHEMAS)} document schemas loaded")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# ── Step 2: Document classifier ───────────────────────────────────────────────
print("\n🔍 Testing document classifier...")
classifier = DocumentClassifier()

test_texts = [
    ("900101-14-5123\nWARGANEGARA\nALIYA BINTI ALIAS",
     "mykad", "Malaysian IC text"),

    ("SURUHANJAYA SYARIKAT MALAYSIA\nSijil Pendaftaran Perniagaan\n"
     "Nama Perniagaan: ALIYA TECH ENTERPRISE\n201901234567",
     "ssm_registration", "SSM certificate text"),

    ("INVOICE NO: INV-2024-001\nTax Invoice\nRM 1,500.00\nSST @ 6%: RM 90.00\n"
     "TOTAL: RM 1,590.00",
     "invoice", "Malaysian invoice text"),

    ("SLIP GAJI\nGaji Pokok: RM 5,500\nEPF: RM 605\nPCB: RM 380\nGaji Bersih: RM 4,515",
     "payslip", "Payslip text"),

    ("BORANG EA\nTahun Taksiran 2024\nLembaga Hasil Dalam Negeri\n"
     "Jumlah Pendapatan: RM 66,000",
     "ea_form", "EA Form text"),

    ("TNB ELECTRICITY BILL\nAmount Due: RM 245.60\nDue Date: 15/11/2024\n"
     "Units Consumed: 412 kWh",
     "utility_bill", "TNB utility bill"),

    ("What's for lunch today?", "unknown", "Unrelated text"),
]

classifier_pass = 0
for text, expected, label in test_texts:
    result = classifier.classify_from_text(text)
    ok = result.document_type == expected
    icon = "✅" if ok else "❌"
    print(f"   {icon} [{label}] detected={result.document_type} "
          f"(conf={result.confidence:.0%}) expected={expected}")
    if ok:
        classifier_pass += 1

print(f"\n   Classifier: {classifier_pass}/{len(test_texts)} correct")

# ── Step 3: Regex extraction ──────────────────────────────────────────────────
print("\n📋 Testing regex extraction...")

test_cases = [
    {
        "doc_type": "mykad",
        "text": "900101-14-5123\nALIYA BINTI ALIAS\nWARGANEGARA\nISLAM",
        "expected_field": "ic_number",
        "expected_value": "900101-14-5123",
        "label": "MyKad IC number"
    },
    {
        "doc_type": "invoice",
        "text": "INVOICE NO: INV-001\nTotal Amount: RM 1,500.00\nSST @ 6%: RM 90.00",
        "expected_field": "total_amount",
        "expected_value": 1500.0,
        "label": "Invoice total"
    },
    {
        "doc_type": "payslip",
        "text": "Basic Salary: RM 5,500.00\nGross Pay: RM 6,200.00\nNet Pay: RM 5,215.00",
        "expected_field": "net_pay",
        "expected_value": 5215.0,
        "label": "Payslip net pay"
    },
    {
        "doc_type": "utility_bill",
        "text": "TNB ELECTRICITY BILL\nAmount Due: RM 245.60\n412 kWh consumed",
        "expected_field": "provider",
        "expected_value": "TNB (Tenaga Nasional Berhad)",
        "label": "TNB utility provider"
    },
    {
        "doc_type": "ea_form",
        "text": "BORANG EA\nTahun Taksiran 2024\nEmployee: Aliya Alias\n"
                "Gross Salary: RM 66,000\nPCB: RM 5,400",
        "expected_field": "assessment_year",
        "expected_value": "2024",
        "label": "EA Form year"
    },
]

regex_pass = 0
for case in test_cases:
    extracted = extract_with_regex(case["doc_type"], case["text"])
    value = extracted.get(case["expected_field"])
    ok = value == case["expected_value"]
    icon = "✅" if ok else "❌"
    print(f"   {icon} [{case['label']}] "
          f"extracted='{value}' expected='{case['expected_value']}'")
    if ok:
        regex_pass += 1

print(f"\n   Regex extraction: {regex_pass}/{len(test_cases)} correct")

# ── Step 4: Pydantic schema validation ────────────────────────────────────────
print("\n✅ Testing Pydantic schema validation...")

test_schemas = [
    ("mykad", {
        "ic_number": "900101-14-5123",
        "full_name": "ALIYA BINTI ALIAS",
        "gender": "female",
        "nationality": "WARGANEGARA",
    }),
    ("invoice", {
        "invoice_number": "INV-2024-001",
        "total_amount": 1590.0,
        "sst_rate": "6%",
        "sst_amount": 90.0,
    }),
    ("ssm_registration", {
        "registration_number": "201901234567",
        "business_name": "ALIYA TECH ENTERPRISE",
        "business_type": "Sole Proprietorship",
        "status": "Active",
    }),
]

schema_pass = 0
for doc_type, data in test_schemas:
    schema_class = DOCUMENT_SCHEMAS[doc_type]
    try:
        validated = schema_class(confidence=0.90, **data)
        out = validated.model_dump(exclude_none=True)
        icon = "✅"
        schema_pass += 1
        key_count = len([k for k in out if k not in ("document_type", "confidence")])
        print(f"   ✅ [{doc_type}] validated OK | {key_count} fields")
    except Exception as e:
        print(f"   ❌ [{doc_type}] validation failed: {e}")

print(f"\n   Schema validation: {schema_pass}/{len(test_schemas)} correct")

# ── Step 5: Check VLM availability ───────────────────────────────────────────
print("\n🤖 Checking VLM/OCR availability...")

try:
    import torch
    has_gpu = torch.cuda.is_available()
    print(f"   GPU available: {'✅ ' + torch.cuda.get_device_name(0) if has_gpu else '❌ (CPU only)'}")
except ImportError:
    has_gpu = False
    print("   PyTorch: ❌ not installed")

try:
    import easyocr
    print("   EasyOCR: ✅ available (fast fallback)")
except ImportError:
    print("   EasyOCR: ❌ not installed (run: pip install easyocr)")

try:
    import transformers
    print(f"   Transformers: ✅ v{transformers.__version__}")
    if has_gpu:
        print("   Qwen2.5-VL: ✅ will use GPU for best quality")
    else:
        print("   Qwen2.5-VL: ⚠️ will use CPU (slower)")
except ImportError:
    print("   Transformers: ❌ not installed (optional for VLM)")

# ── Summary ───────────────────────────────────────────────────────────────────
total_pass = classifier_pass + regex_pass + schema_pass
total_tests = len(test_texts) + len(test_cases) + len(test_schemas)

print(f"\n{'='*55}")
print(f"📊 RESULTS: {total_pass}/{total_tests} tests passed")

if total_pass == total_tests:
    print("✅ All tests passed! Core pipeline working.")
else:
    print(f"⚠️  {total_tests - total_pass} tests failed.")

print(f"\nNext steps:")
print(f"  1. Run the demo:")
print(f"       python demo/app.py")
print(f"  2. Start the API:")
print(f"       uvicorn api.main:app --reload --port 8000")
print(f"  3. For VLM (best quality):")
print(f"       pip install transformers easyocr pillow torch")
print(f"       # Then rerun quickstart — VLM will be loaded automatically")
print(f"{'='*55}\n")
