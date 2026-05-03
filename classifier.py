"""
Document Classifier
====================
Determines the type of document from an image before extraction.
Fast, runs with zero ML models — pure regex and keyword matching.

Two classification modes:
1. Pattern-based (fast, deterministic, 0ms) — primary
2. VLM-based (accurate, ~1s) — fallback for ambiguous documents

Usage:
    classifier = DocumentClassifier()
    doc_type, confidence = classifier.classify_from_text("extracted OCR text")
    doc_type, confidence = classifier.classify_from_filename("invoice_april.pdf")
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Keyword signatures per document type ─────────────────────────────────────

DOCUMENT_SIGNATURES = {
    "mykad": {
        "required": ["mykad", "kad pengenalan", "warganegara", "pemastautin"],
        "patterns": [
            r"\b\d{6}-\d{2}-\d{4}\b",              # IC number format
            r"jabatan\s+pendaftaran\s+negara",
            r"malaysia\b.*\bidentity",
        ],
        "score_bonus": 15,
    },
    "sg_nric": {
        "required": ["nric", "singapore", "identity card"],
        "patterns": [
            r"\b[STFGM]\d{7}[A-Z]\b",              # NRIC format
            r"republic\s+of\s+singapore",
            r"immigration\s+&\s+checkpoints",
        ],
        "score_bonus": 15,
    },
    "ssm_registration": {
        "required": [],
        "keywords": ["suruhanjaya syarikat", "ssm", "pendaftaran perniagaan",
                     "companies commission", "certificate of registration",
                     "sijil pendaftaran", "nombor pendaftaran",
                     "registration of businesses", "companies act"],
        "patterns": [
            r"\b(201[0-9]|202[0-9])\d{8}\b",       # New 12-digit SSM format
            r"\b\d{6,7}-[A-Z]\b",                   # Old SSM format
            r"borang\s+[abc]\b",
            r"registration\s+fee\s+rm",
        ],
        "score_bonus": 10,
    },
    "invoice": {
        "required": [],
        "keywords": ["invoice", "invois", "tax invoice", "invois cukai",
                     "receipt", "resit", "total amount", "jumlah",
                     "subtotal", "sst", "service tax", "sales tax"],
        "patterns": [
            r"invoice\s+(no|number|#)",
            r"bil\s+(no|nombor|#)",
            r"rm\s*[\d,]+\.\d{2}",                  # Malaysian Ringgit amount
            r"sst\s*@?\s*\d+\s*%",
        ],
        "score_bonus": 5,
    },
    "ea_form": {
        "required": [],
        "keywords": ["borang ea", "ea form", "lembaga hasil dalam negeri",
                     "lhdn", "hasil", "pcb", "potongan cukai bulanan",
                     "monthly tax deduction", "income tax", "cukai pendapatan",
                     "assessment year", "tahun taksiran"],
        "patterns": [
            r"borang\s+ea\b",
            r"e[\.\s]*?number\s*:",
            r"pcb\s+deducted",
            r"tahun\s+taksiran\s+\d{4}",
        ],
        "score_bonus": 15,
    },
    "payslip": {
        "required": [],
        "keywords": ["payslip", "salary slip", "slip gaji", "gaji",
                     "basic salary", "gaji pokok", "epf", "kwsp",
                     "socso", "perkeso", "pcb", "net pay", "gaji bersih",
                     "gross pay", "gaji kasar"],
        "patterns": [
            r"pay\s*period\s*:",
            r"tempoh\s+gaji\s*:",
            r"employee\s+(id|no)\s*:",
            r"basic\s+salary\s*:?\s*rm",
        ],
        "score_bonus": 10,
    },
    "bank_statement": {
        "required": [],
        "keywords": ["statement", "penyata", "account statement",
                     "penyata akaun", "transaction", "transaksi",
                     "balance", "baki", "debit", "credit", "kredit",
                     "maybank", "cimb", "public bank", "rhb", "hlb",
                     "hong leong", "bank islam", "affin"],
        "patterns": [
            r"account\s+(no|number)\s*:?\s*[\d\-\s]+",
            r"penyata\s+akaun",
            r"closing\s+balance\s*:?\s*rm",
            r"statement\s+period\s*:",
        ],
        "score_bonus": 5,
    },
    "utility_bill": {
        "required": [],
        "keywords": ["tnb", "tenaga nasional", "electricity", "elektrik",
                     "unifi", "streamyx", "tm", "maxis", "celcom",
                     "syabas", "water", "air", "utility", "utiliti",
                     "amount due", "amaun perlu dibayar"],
        "patterns": [
            r"account\s+number\s*:?\s*[\d\-]+",
            r"bill\s+date\s*:",
            r"due\s+date\s*:",
            r"kwh\s+consumed",
            r"unit\s+consumed",
        ],
        "score_bonus": 8,
    },
    "epf_statement": {
        "required": [],
        "keywords": ["kwsp", "epf", "kumpulan wang simpanan pekerja",
                     "employees provident fund", "akaun 1", "akaun 2",
                     "account 1", "account 2", "dividend", "dividen",
                     "tabung haji"],
        "patterns": [
            r"membership\s+number\s*:",
            r"akaun\s+[12]\b",
            r"dividen\s+@\s*\d+",
            r"dividend\s+rate\s*:",
        ],
        "score_bonus": 12,
    },
}

FILENAME_HINTS = {
    "mykad": ["mykad", "ic", "identity", "kad_pengenalan", "nric_my"],
    "sg_nric": ["nric", "sg_ic", "singapore_id"],
    "ssm_registration": ["ssm", "business_cert", "sijil", "registration_cert"],
    "invoice": ["invoice", "invois", "receipt", "resit", "bill", "tax_inv"],
    "ea_form": ["ea_form", "borang_ea", "ea_2", "lhdn"],
    "payslip": ["payslip", "salary", "gaji", "slip_gaji"],
    "bank_statement": ["statement", "penyata", "bank_stmt"],
    "utility_bill": ["tnb", "unifi", "maxis", "utility", "electricity", "water_bill"],
    "epf_statement": ["epf", "kwsp", "provident"],
}


@dataclass
class ClassificationResult:
    document_type: str
    confidence: float
    method: str          # "keyword" | "pattern" | "filename" | "vlm"
    scores: dict         # All document type scores for debugging


class DocumentClassifier:
    """
    Fast document type classifier.

    Uses keyword matching + regex patterns on OCR text.
    Returns document_type and confidence score.

    Usage:
        classifier = DocumentClassifier()

        # From OCR text (primary usage)
        result = classifier.classify_from_text(ocr_text)

        # From filename (quick pre-filter)
        result = classifier.classify_from_filename("invoice_march_2024.pdf")

        # Get human-readable description
        desc = classifier.describe(result.document_type)
    """

    def __init__(self):
        self._compiled_patterns = {}
        for doc_type, sig in DOCUMENT_SIGNATURES.items():
            self._compiled_patterns[doc_type] = [
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in sig.get("patterns", [])
            ]

    def classify_from_text(self, text: str) -> ClassificationResult:
        """
        Classify document type from OCR-extracted text.
        Returns the most likely document type and confidence.
        """
        if not text or not text.strip():
            return ClassificationResult(
                document_type="unknown", confidence=0.0,
                method="keyword", scores={}
            )

        text_lower = text.lower()
        scores = {}

        for doc_type, sig in DOCUMENT_SIGNATURES.items():
            score = 0.0

            # Required keyword check (must have at least one)
            required = sig.get("required", [])
            if required:
                if not any(kw in text_lower for kw in required):
                    scores[doc_type] = 0.0
                    continue
                # Found a required keyword — bonus
                score += 20

            # Optional keywords
            keywords = sig.get("keywords", [])
            for kw in keywords:
                if kw in text_lower:
                    score += 8

            # Regex patterns
            for pattern in self._compiled_patterns.get(doc_type, []):
                if pattern.search(text):
                    score += 12

            # Score bonus per document type
            score_bonus = sig.get("score_bonus", 0)
            if score > 0:
                score += score_bonus

            scores[doc_type] = score

        if not scores or max(scores.values()) == 0:
            return ClassificationResult(
                document_type="unknown", confidence=0.0,
                method="keyword", scores=scores
            )

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        second_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

        # Confidence = how dominant the winner is
        if best_score == 0:
            confidence = 0.0
        elif second_score == 0:
            confidence = min(0.95, best_score / 50)
        else:
            confidence = min(0.95, (best_score - second_score) / (best_score + second_score + 1) + 0.3)

        confidence = max(0.0, min(1.0, confidence))

        logger.debug(f"Document classification: {best_type} ({confidence:.0%}) | scores={dict(sorted(scores.items(), key=lambda x: -x[1])[:3])}")

        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            method="keyword",
            scores=scores,
        )

    def classify_from_filename(self, filename: str) -> ClassificationResult:
        """Quick classification from filename — use as pre-filter."""
        filename_lower = filename.lower()

        for doc_type, hints in FILENAME_HINTS.items():
            for hint in hints:
                if hint in filename_lower:
                    return ClassificationResult(
                        document_type=doc_type,
                        confidence=0.70,   # Filename hints are suggestive not definitive
                        method="filename",
                        scores={},
                    )

        return ClassificationResult(
            document_type="unknown", confidence=0.0,
            method="filename", scores={}
        )

    def classify(self, text: str, filename: str | None = None) -> ClassificationResult:
        """
        Combined classification — text first, filename as tiebreaker.
        """
        text_result = self.classify_from_text(text)

        if text_result.confidence >= 0.70:
            return text_result

        if filename:
            file_result = self.classify_from_filename(filename)
            if file_result.document_type != "unknown":
                # Use filename hint to boost the text result
                if (file_result.document_type == text_result.document_type or
                        text_result.document_type == "unknown"):
                    return ClassificationResult(
                        document_type=file_result.document_type,
                        confidence=max(text_result.confidence, 0.65),
                        method="combined",
                        scores=text_result.scores,
                    )

        return text_result

    def describe(self, doc_type: str) -> str:
        """Human-readable description of a document type."""
        from models.document_schemas import DOCUMENT_DESCRIPTIONS
        return DOCUMENT_DESCRIPTIONS.get(doc_type, f"Unknown document type: {doc_type}")
