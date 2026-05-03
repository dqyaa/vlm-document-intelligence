"""
VLM Document Extractor
=======================
Uses Qwen2.5-VL (Vision-Language Model) to extract structured data
from document images. Falls back to EasyOCR + regex when VLM is unavailable.

Architecture (two-stage):
  Stage 1: OCR / VLM → raw text extraction from image
  Stage 2: LLM prompt → structured JSON extraction from text

This separation means:
  - Fast path: EasyOCR (no GPU needed) → regex extraction
  - Accurate path: Qwen2.5-VL → JSON extraction

Why Qwen2.5-VL?
  - Free, Apache 2.0 license
  - Native Bahasa Malaysia + English support
  - Excellent document understanding (tables, forms, invoices)
  - Runs on free Colab T4 GPU (7B param version)
  - 3B version runs on CPU with ~10s latency

Usage:
    # Full VLM pipeline (needs GPU)
    extractor = VLMExtractor(model="qwen2.5-vl")
    result = extractor.extract("mykad.jpg")

    # Fast OCR-only pipeline (no GPU needed)
    extractor = VLMExtractor(model="easyocr")
    result = extractor.extract("invoice.jpg")

    # Auto-detect best available
    extractor = VLMExtractor()
    result = extractor.extract("statement.jpg")
"""

import json
import time
import base64
import logging
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Complete result from document extraction."""
    document_type: str
    confidence: float
    extracted_data: dict        # Validated Pydantic model as dict
    raw_text: str               # Full OCR text
    model_used: str             # Which model was used
    latency_ms: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Extraction prompts per document type ─────────────────────────────────────

EXTRACTION_PROMPTS = {
    "mykad": """Extract ALL text from this Malaysian MyKad (IC) image.
Then return a JSON object with these fields (use null for missing fields):
{
  "ic_number": "YYMMDD-SS-GGGG format (12 digits with dashes)",
  "full_name": "Full name in ALL CAPS as shown on card",
  "date_of_birth": "DD/MM/YYYY or YYMMDD",
  "gender": "male or female (odd last 4 digits = male, even = female)",
  "nationality": "WARGANEGARA or PEMASTAUTIN TETAP or other",
  "religion": "Islam/Buddhist/etc if visible",
  "race": "Malay/Chinese/Indian/etc if visible",
  "address": "Full address from back of card"
}
Return ONLY the JSON object, nothing else.""",

    "sg_nric": """Extract ALL text from this Singapore NRIC (identity card) image.
Return a JSON object:
{
  "nric_number": "Letter + 7 digits + Letter (e.g. S1234567A)",
  "full_name": "Full name as shown",
  "date_of_birth": "DD/MM/YYYY",
  "sex": "MALE or FEMALE",
  "race": "Race/ethnicity as shown",
  "country_of_birth": "Country",
  "address": "Full address if visible"
}
Return ONLY the JSON object.""",

    "ssm_registration": """Extract ALL text from this SSM (Suruhanjaya Syarikat Malaysia)
business registration certificate.
Return a JSON object:
{
  "registration_number": "Business registration number",
  "business_name": "Nama Perniagaan / Business name",
  "business_type": "Sole Proprietorship / Sdn Bhd / Partnership / LLP",
  "registration_date": "DD/MM/YYYY",
  "commencement_date": "DD/MM/YYYY",
  "registered_address": "Full registered address",
  "owner_name": "Owner name (for sole proprietorships)",
  "owner_ic": "Owner IC number",
  "business_activity": "Description of business activity",
  "msic_code": "5-digit MSIC code",
  "status": "Active / Wound Up / Struck Off",
  "expiry_date": "Certificate expiry / renewal date"
}
Return ONLY the JSON object.""",

    "invoice": """Extract ALL text from this Malaysian invoice/receipt image.
Return a JSON object:
{
  "invoice_number": "Invoice or receipt number",
  "invoice_date": "Date of invoice",
  "seller_name": "Seller/company name",
  "seller_ssm": "SSM registration number if shown",
  "seller_address": "Seller address",
  "buyer_name": "Customer/buyer name",
  "buyer_address": "Buyer address if shown",
  "line_items": [
    {"description": "Item name", "quantity": 1, "unit_price": 0.0, "amount": 0.0}
  ],
  "subtotal": 0.0,
  "sst_rate": "6% or 8% or null",
  "sst_amount": 0.0,
  "discount": 0.0,
  "total_amount": 0.0,
  "payment_method": "Cash/Card/Online Transfer"
}
All amounts in MYR as numbers (not strings). Return ONLY the JSON object.""",

    "ea_form": """Extract ALL text from this LHDN EA Form (Borang EA).
Return a JSON object:
{
  "assessment_year": "Year e.g. 2025",
  "employer_name": "Company/employer name",
  "employer_e_number": "E number / employer reference",
  "employee_name": "Employee full name",
  "employee_ic": "Employee IC number",
  "employee_tax_file": "Employee tax file number (SG/OG)",
  "gross_salary": 0.0,
  "bonus": 0.0,
  "allowances": 0.0,
  "total_gross_income": 0.0,
  "epf_employee": 0.0,
  "socso_employee": 0.0,
  "pcb_deducted": 0.0,
  "net_income": 0.0
}
All monetary values as numbers in MYR. Return ONLY the JSON object.""",

    "payslip": """Extract ALL text from this Malaysian salary payslip (slip gaji).
Return a JSON object:
{
  "employer_name": "Company name",
  "employee_name": "Employee name",
  "employee_id": "Staff ID",
  "position": "Job title",
  "pay_period": "Month Year e.g. October 2024",
  "basic_salary": 0.0,
  "overtime": 0.0,
  "allowances": 0.0,
  "gross_pay": 0.0,
  "epf_employee": 0.0,
  "socso_employee": 0.0,
  "income_tax_pcb": 0.0,
  "total_deductions": 0.0,
  "net_pay": 0.0
}
All amounts as numbers in MYR. Return ONLY the JSON object.""",

    "bank_statement": """Extract ALL information from this Malaysian bank statement image.
Return a JSON object:
{
  "bank_name": "Bank name",
  "account_holder": "Account holder name",
  "account_number": "Account number (may be masked)",
  "account_type": "Savings / Current / Islamic",
  "statement_period_from": "DD/MM/YYYY",
  "statement_period_to": "DD/MM/YYYY",
  "opening_balance": 0.0,
  "closing_balance": 0.0,
  "total_credits": 0.0,
  "total_debits": 0.0,
  "transactions": [
    {"date": "DD/MM", "description": "Txn description",
     "debit": null, "credit": 0.0, "balance": 0.0}
  ]
}
Amounts in MYR as numbers. Include up to 10 transactions. Return ONLY the JSON.""",

    "utility_bill": """Extract ALL text from this Malaysian utility bill.
Return a JSON object:
{
  "provider": "TNB / Unifi / Maxis / Syabas / etc",
  "account_number": "Account number",
  "account_holder": "Customer name",
  "service_address": "Service address",
  "bill_date": "DD/MM/YYYY",
  "due_date": "DD/MM/YYYY",
  "billing_period_from": "DD/MM/YYYY",
  "billing_period_to": "DD/MM/YYYY",
  "amount_due": 0.0,
  "total_amount": 0.0,
  "units_consumed": null
}
Return ONLY the JSON object.""",

    "epf_statement": """Extract ALL information from this EPF/KWSP statement.
Return a JSON object:
{
  "member_name": "Member name",
  "member_ic": "IC number",
  "membership_number": "EPF membership number",
  "statement_year": "Year",
  "account1_opening": 0.0,
  "account1_contributions": 0.0,
  "account1_dividends": 0.0,
  "account1_closing": 0.0,
  "account2_opening": 0.0,
  "account2_contributions": 0.0,
  "account2_dividends": 0.0,
  "account2_closing": 0.0,
  "total_savings": 0.0,
  "dividend_rate": "5.50%"
}
Return ONLY the JSON object.""",

    "unknown": """Extract ALL text from this document image.
Identify what type of document it is, then extract key information.
Return a JSON object with:
{
  "document_type_detected": "What type of document this appears to be",
  "key_fields": {}
}
Return ONLY the JSON object.""",
}


class VLMExtractor:
    """
    Vision-Language Model document extractor.

    Supports multiple backends:
    - Qwen2.5-VL (best quality, needs GPU)
    - EasyOCR + regex (fast, no GPU)
    - Claude/GPT-4o API (cloud, highest accuracy)

    Auto-detects best available backend if not specified.
    """

    def __init__(
        self,
        model: str = "auto",          # "qwen2.5-vl" | "easyocr" | "claude" | "auto"
        model_size: str = "7b",       # "3b" | "7b" | "72b"
        api_key: Optional[str] = None,
        device: str = "auto",         # "cuda" | "cpu" | "auto"
    ):
        self.model_name = model
        self.model_size = model_size
        self.api_key = api_key
        self.device = device
        self._vlm = None
        self._ocr = None
        self._processor = None
        self._active_backend = None

        if model == "auto":
            self._active_backend = self._detect_best_backend()
        else:
            self._active_backend = model

        logger.info(f"VLMExtractor initialised | backend={self._active_backend}")

    def _detect_best_backend(self) -> str:
        """Auto-detect the best available extraction backend."""
        # Check GPU availability
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            has_gpu = False

        # Check if transformers is available (for Qwen2.5-VL)
        try:
            import transformers
            has_transformers = True
        except ImportError:
            has_transformers = False

        # Check if EasyOCR is available
        try:
            import easyocr
            has_easyocr = True
        except ImportError:
            has_easyocr = False

        if has_transformers and has_gpu:
            logger.info("GPU detected — using Qwen2.5-VL for best quality")
            return "qwen2.5-vl"
        elif has_transformers:
            logger.info("No GPU — using Qwen2.5-VL-3B on CPU (slower)")
            self.model_size = "3b"
            return "qwen2.5-vl"
        elif has_easyocr:
            logger.info("Using EasyOCR (no transformers available)")
            return "easyocr"
        else:
            logger.warning(
                "No VLM or OCR backend available. "
                "Install: pip install transformers easyocr"
            )
            return "regex_only"

    def _load_vlm(self):
        """Lazy-load Qwen2.5-VL model."""
        if self._vlm is not None:
            return

        model_id = {
            "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        }.get(self.model_size, "Qwen/Qwen2.5-VL-7B-Instruct")

        logger.info(f"Loading {model_id}...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch

            device_map = "auto" if self.device == "auto" else self.device
            load_in_4bit = torch.cuda.is_available()

            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self._vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map=device_map,
                )
            else:
                self._vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map=device_map,
                )

            self._processor = AutoProcessor.from_pretrained(model_id)
            logger.info(f"✅ {model_id} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            logger.info("Falling back to EasyOCR")
            self._active_backend = "easyocr"

    def _load_easyocr(self):
        """Lazy-load EasyOCR."""
        if self._ocr is not None:
            return
        try:
            import easyocr
            self._ocr = easyocr.Reader(
                ["ms", "en"],    # Malay + English
                gpu=False,       # CPU for portability
                verbose=False,
            )
            logger.info("✅ EasyOCR loaded (ms + en)")
        except Exception as e:
            logger.error(f"EasyOCR load failed: {e}")

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _ocr_with_easyocr(self, image_path: str) -> str:
        """Extract text using EasyOCR."""
        self._load_easyocr()
        if self._ocr is None:
            return ""
        try:
            results = self._ocr.readtext(image_path, detail=0, paragraph=True)
            return "\n".join(results)
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""

    def _extract_with_vlm(self, image_path: str, doc_type: str) -> tuple[str, str]:
        """
        Extract structured data using Qwen2.5-VL.
        Returns (raw_text, json_string).
        """
        self._load_vlm()
        if self._vlm is None:
            return "", "{}"

        try:
            from PIL import Image
            import torch

            image = Image.open(image_path).convert("RGB")
            prompt = EXTRACTION_PROMPTS.get(doc_type, EXTRACTION_PROMPTS["unknown"])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            from qwen_vl_utils import process_vision_info
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._vlm.device)

            with torch.no_grad():
                generated_ids = self._vlm.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=True,
                )

            output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
            raw_output = self._processor.decode(output_ids, skip_special_tokens=True)

            # Also do an OCR-only pass for raw text
            raw_text_prompt = "Extract all text from this document image. Return only the text, preserving layout."
            messages_ocr = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": raw_text_prompt},
                    ],
                }
            ]
            text_ocr = self._processor.apply_chat_template(
                messages_ocr, tokenize=False, add_generation_prompt=True)
            image_inputs_ocr, _ = process_vision_info(messages_ocr)
            inputs_ocr = self._processor(
                text=[text_ocr], images=image_inputs_ocr,
                padding=True, return_tensors="pt"
            ).to(self._vlm.device)

            with torch.no_grad():
                gen_ocr = self._vlm.generate(**inputs_ocr, max_new_tokens=512)
            raw_text = self._processor.decode(
                gen_ocr[0][inputs_ocr.input_ids.shape[1]:], skip_special_tokens=True
            )

            return raw_text, raw_output

        except Exception as e:
            logger.error(f"VLM extraction error: {e}")
            return "", "{}"

    def _extract_json_from_text(self, text: str) -> dict:
        """Parse JSON from model output, handling markdown code blocks."""
        # Remove markdown code blocks
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_match = re.search(r"\{[\s\S]+\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from: {text[:200]}")
        return {}

    def _validate_and_build(self, doc_type: str, raw_data: dict,
                             raw_text: str, confidence: float) -> dict:
        """Validate extracted data against Pydantic schema."""
        from models.document_schemas import DOCUMENT_SCHEMAS

        schema_class = DOCUMENT_SCHEMAS.get(doc_type)
        if not schema_class:
            return {**raw_data, "document_type": doc_type, "confidence": confidence}

        try:
            validated = schema_class(
                confidence=confidence,
                raw_text=raw_text[:2000] if raw_text else None,
                **{k: v for k, v in raw_data.items()
                   if k not in ("document_type", "confidence", "raw_text")}
            )
            result = validated.model_dump(exclude_none=True)
            return result

        except Exception as e:
            logger.warning(f"Pydantic validation warning for {doc_type}: {e}")
            # Return raw data with base fields
            return {
                "document_type": doc_type,
                "confidence": confidence,
                "raw_text": raw_text[:2000] if raw_text else None,
                **raw_data,
            }

    def extract(
        self,
        image_path: str,
        doc_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from a document image.

        Args:
            image_path: Path to image file (JPG, PNG, PDF)
            doc_type:   Override document type classification
            filename:   Original filename (used as classification hint)

        Returns:
            ExtractionResult with extracted_data, raw_text, model_used, latency
        """
        t0 = time.time()
        errors = []
        warnings = []
        image_path = str(image_path)

        # ── Stage 1: Extract raw text ────────────────────────────────────────
        if self._active_backend == "qwen2.5-vl":
            raw_text, json_output = self._extract_with_vlm(
                image_path, doc_type or "unknown"
            )
            model_used = f"Qwen2.5-VL-{self.model_size.upper()}"

        elif self._active_backend == "easyocr":
            raw_text = self._ocr_with_easyocr(image_path)
            json_output = None
            model_used = "EasyOCR"

        else:
            # No backend — try PIL for basic image reading
            try:
                from PIL import Image
                img = Image.open(image_path)
                raw_text = f"Image loaded: {img.size} pixels. No OCR backend available."
            except Exception:
                raw_text = ""
            json_output = None
            model_used = "none"
            warnings.append("No OCR backend available. Install easyocr or transformers.")

        # ── Stage 2: Classify document type ─────────────────────────────────
        from extractors.classifier import DocumentClassifier
        classifier = DocumentClassifier()

        if doc_type is None:
            classification = classifier.classify(raw_text, filename)
            doc_type = classification.document_type
            confidence = classification.confidence
        else:
            confidence = 0.90  # User-specified = high confidence

        # ── Stage 3: Parse structured data ──────────────────────────────────
        if json_output:
            # VLM already gave us JSON
            raw_data = self._extract_json_from_text(json_output)
        else:
            # EasyOCR path: use regex extractors
            from extractors.regex_extractors import extract_with_regex
            raw_data = extract_with_regex(doc_type, raw_text)

        # ── Stage 4: Validate against schema ─────────────────────────────────
        extracted_data = self._validate_and_build(doc_type, raw_data, raw_text, confidence)

        latency = (time.time() - t0) * 1000

        logger.info(
            f"Extraction complete | type={doc_type} conf={confidence:.0%} "
            f"backend={model_used} latency={latency:.0f}ms"
        )

        return ExtractionResult(
            document_type=doc_type,
            confidence=confidence,
            extracted_data=extracted_data,
            raw_text=raw_text,
            model_used=model_used,
            latency_ms=round(latency, 1),
            errors=errors,
            warnings=warnings,
        )
