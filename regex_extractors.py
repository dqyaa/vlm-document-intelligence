"""
Regex Extractors
================
Fast, deterministic pattern-based extraction from OCR text.
Used as fallback when VLM is unavailable, or for structured fields
that regex handles better than LLMs (dates, numbers, ICs).

These are also used to validate/correct VLM output.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Common patterns ───────────────────────────────────────────────────────────

IC_PATTERN = re.compile(r'\b(\d{6})[-\s]?(\d{2})[-\s]?(\d{4})\b')
SG_NRIC_PATTERN = re.compile(r'\b([STFGM]\d{7}[A-Z])\b', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.IGNORECASE)
PHONE_MY_PATTERN = re.compile(r'(\+?60|0)[-\s]?1[0-9][-\s]?\d{3,4}[-\s]?\d{3,4}')
DATE_PATTERN = re.compile(
    r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b|'
    r'\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b'
)
AMOUNT_PATTERN = re.compile(r'rm\s*[\d,]+\.\d{2}', re.IGNORECASE)
SSM_OLD_PATTERN = re.compile(r'\b\d{6,8}[-\s][A-Z]\b')
SSM_NEW_PATTERN = re.compile(r'\b(201[0-9]|202[0-9])\d{8}\b')


def _find_amount(text: str, label: str) -> Optional[float]:
    """Find a specific labelled amount in text."""
    pattern = re.compile(
        rf'{re.escape(label)}[\s:]*rm?\s*([\d,]+\.?\d*)',
        re.IGNORECASE
    )
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _find_first_amount(text: str) -> Optional[float]:
    """Find the first RM amount in text."""
    m = AMOUNT_PATTERN.search(text)
    if m:
        try:
            digits = re.sub(r'[^\d.]', '', m.group())
            return float(digits)
        except ValueError:
            pass
    return None


def _find_date(text: str, label: str) -> Optional[str]:
    """Find a date following a specific label."""
    pattern = re.compile(
        rf'{re.escape(label)}[\s:]*(\d{{1,2}}[/\-.]\d{{1,2}}[/\-.]\d{{2,4}})',
        re.IGNORECASE
    )
    m = pattern.search(text)
    return m.group(1) if m else None


def _extract_name_after_label(text: str, labels: list[str]) -> Optional[str]:
    """Extract a name appearing after a given label."""
    for label in labels:
        pattern = re.compile(
            rf'{re.escape(label)}\s*:?\s*([A-Z][A-Z\s/.\'-]{{3,60}})',
            re.IGNORECASE
        )
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


# ── Per-document extractors ───────────────────────────────────────────────────

def extract_mykad(text: str) -> dict:
    result = {}

    # IC number
    m = IC_PATTERN.search(text)
    if m:
        ic = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        result["ic_number"] = ic
        # Gender from last digit of sequence
        last_digit = int(m.group(3)[-1])
        result["gender"] = "male" if last_digit % 2 == 1 else "female"
        # Birth year
        yy = int(m.group(1)[:2])
        result["birth_year"] = 2000 + yy if yy <= 25 else 1900 + yy

    # Name (all caps line)
    name_match = re.search(r'\b([A-Z][A-Z\s/.\'-]{5,50}[A-Z])\b', text)
    if name_match:
        result["full_name"] = name_match.group(1).strip()

    # Nationality
    if "warganegara" in text.lower():
        result["nationality"] = "WARGANEGARA"
    elif "pemastautin tetap" in text.lower():
        result["nationality"] = "PEMASTAUTIN TETAP"

    # Religion
    religions = ["islam", "buddha", "hindu", "kristian", "sikh", "tiada agama"]
    for rel in religions:
        if rel in text.lower():
            result["religion"] = rel.title()
            break

    return result


def extract_sg_nric(text: str) -> dict:
    result = {}
    m = SG_NRIC_PATTERN.search(text)
    if m:
        result["nric_number"] = m.group(1).upper()

    name_match = re.search(r'name\s*:?\s*([A-Z][A-Z\s]+)', text, re.IGNORECASE)
    if name_match:
        result["full_name"] = name_match.group(1).strip()

    dob_m = _find_date(text, "date of birth")
    if dob_m:
        result["date_of_birth"] = dob_m

    sex_m = re.search(r'sex\s*:?\s*(male|female)', text, re.IGNORECASE)
    if sex_m:
        result["sex"] = sex_m.group(1).upper()

    return result


def extract_ssm_registration(text: str) -> dict:
    result = {}

    # Registration number
    m = SSM_NEW_PATTERN.search(text)
    if m:
        result["registration_number"] = m.group(0)
    else:
        m = SSM_OLD_PATTERN.search(text)
        if m:
            result["registration_number"] = m.group(0)

    # Business name — usually prominent all-caps line
    biz_match = re.search(
        r'(?:nama\s+perniagaan|business\s+name)\s*:?\s*([^\n]+)',
        text, re.IGNORECASE
    )
    if biz_match:
        result["business_name"] = biz_match.group(1).strip()

    # Business type
    if any(t in text.lower() for t in ["sole proprietorship", "enterprise", "perseorangan"]):
        result["business_type"] = "Sole Proprietorship"
    elif "sdn bhd" in text.lower() or "sendirian berhad" in text.lower():
        result["business_type"] = "Sdn Bhd"
    elif "berhad" in text.lower():
        result["business_type"] = "Berhad (Public)"
    elif "llp" in text.lower() or "perkongsian" in text.lower():
        result["business_type"] = "LLP"

    # Dates
    reg_date = _find_date(text, "registration date") or _find_date(text, "tarikh pendaftaran")
    if reg_date:
        result["registration_date"] = reg_date

    # Status
    if "active" in text.lower() or "aktif" in text.lower():
        result["status"] = "Active"
    elif "wound up" in text.lower():
        result["status"] = "Wound Up"

    # MSIC code
    msic = re.search(r'msic\s*:?\s*(\d{5})', text, re.IGNORECASE)
    if msic:
        result["msic_code"] = msic.group(1)

    return result


def extract_invoice(text: str) -> dict:
    result = {}

    # Invoice number
    inv_m = re.search(r'(?:invoice|invois|receipt|resit)\s*(?:no|#|number)?\s*:?\s*([A-Z0-9\-/]+)',
                      text, re.IGNORECASE)
    if inv_m:
        result["invoice_number"] = inv_m.group(1).strip()

    # Dates
    result["invoice_date"] = (
        _find_date(text, "invoice date") or
        _find_date(text, "tarikh") or
        _find_date(text, "date")
    )

    # Total
    for label in ["total amount", "jumlah", "total", "grand total", "amount due"]:
        amt = _find_amount(text, label)
        if amt:
            result["total_amount"] = amt
            break

    # SST
    sst_m = re.search(r'sst\s*@?\s*(\d+)\s*%?\s*:?\s*rm?\s*([\d,.]+)', text, re.IGNORECASE)
    if sst_m:
        result["sst_rate"] = f"{sst_m.group(1)}%"
        try:
            result["sst_amount"] = float(sst_m.group(2).replace(",", ""))
        except ValueError:
            pass

    # Seller name
    seller_m = _extract_name_after_label(text, ["sold by", "from", "company", "syarikat"])
    if seller_m:
        result["seller_name"] = seller_m

    result["currency"] = "MYR"
    return result


def extract_ea_form(text: str) -> dict:
    result = {}

    # Assessment year
    year_m = re.search(r'(?:assessment\s+year|tahun\s+taksiran)\s*:?\s*(\d{4})',
                       text, re.IGNORECASE)
    if year_m:
        result["assessment_year"] = year_m.group(1)

    # Employer
    emp_m = _extract_name_after_label(text, ["employer", "majikan", "company"])
    if emp_m:
        result["employer_name"] = emp_m

    # Employee
    emp_name = _extract_name_after_label(text, ["employee", "pekerja", "name"])
    if emp_name:
        result["employee_name"] = emp_name

    # IC
    ic_m = IC_PATTERN.search(text)
    if ic_m:
        result["employee_ic"] = f"{ic_m.group(1)}-{ic_m.group(2)}-{ic_m.group(3)}"

    # Key amounts
    for field, labels in [
        ("gross_salary", ["gross salary", "gaji kasar", "total gross"]),
        ("epf_employee", ["epf", "kwsp", "employee epf"]),
        ("pcb_deducted", ["pcb", "monthly tax", "potongan cukai"]),
        ("total_gross_income", ["total income", "jumlah pendapatan"]),
    ]:
        for label in labels:
            amt = _find_amount(text, label)
            if amt:
                result[field] = amt
                break

    return result


def extract_payslip(text: str) -> dict:
    result = {}

    # Company / employee
    result["employer_name"] = _extract_name_after_label(text, ["company", "employer", "syarikat"])
    result["employee_name"] = _extract_name_after_label(text, ["employee", "name", "nama"])

    # Pay period
    period_m = re.search(
        r'(?:pay\s+period|tempoh\s+gaji|period)\s*:?\s*([A-Za-z]+\s+\d{4}|\d{2}/\d{4})',
        text, re.IGNORECASE
    )
    if period_m:
        result["pay_period"] = period_m.group(1)

    # Amounts
    for field, labels in [
        ("basic_salary", ["basic salary", "gaji pokok", "basic"]),
        ("gross_pay", ["gross pay", "gaji kasar", "total earnings"]),
        ("epf_employee", ["epf", "kwsp employee"]),
        ("socso_employee", ["socso", "perkeso"]),
        ("income_tax_pcb", ["pcb", "income tax", "cukai pendapatan"]),
        ("net_pay", ["net pay", "gaji bersih", "take home"]),
    ]:
        for label in labels:
            amt = _find_amount(text, label)
            if amt:
                result[field] = amt
                break

    return result


def extract_bank_statement(text: str) -> dict:
    result = {}

    # Bank name
    banks = ["Maybank", "CIMB", "Public Bank", "RHB", "Hong Leong",
             "Bank Islam", "Affin", "AmBank", "Alliance", "BSN"]
    for bank in banks:
        if bank.lower() in text.lower():
            result["bank_name"] = bank
            break

    # Account number
    acc_m = re.search(r'account\s*(?:no|number)\s*:?\s*([\d\-\s\*]+)', text, re.IGNORECASE)
    if acc_m:
        result["account_number"] = acc_m.group(1).strip()

    # Account holder
    result["account_holder"] = _extract_name_after_label(
        text, ["account holder", "account name", "name"])

    # Balances
    result["opening_balance"] = _find_amount(text, "opening balance")
    result["closing_balance"] = _find_amount(text, "closing balance")

    # Account type
    if any(t in text.lower() for t in ["islamic", "akaun-i", "savings-i"]):
        result["account_type"] = "Islamic Savings"
    elif "current" in text.lower():
        result["account_type"] = "Current"
    elif "savings" in text.lower() or "simpanan" in text.lower():
        result["account_type"] = "Savings"

    return result


def extract_utility_bill(text: str) -> dict:
    result = {}

    providers = {
        "tnb": "TNB (Tenaga Nasional Berhad)",
        "tenaga nasional": "TNB (Tenaga Nasional Berhad)",
        "unifi": "Unifi (TM)",
        "streamyx": "Streamyx (TM)",
        "maxis": "Maxis",
        "celcom": "Celcom",
        "digi": "Digi",
        "syabas": "Syabas (Air Selangor)",
        "pengurusan air": "Air Selangor",
    }

    text_lower = text.lower()
    for key, name in providers.items():
        if key in text_lower:
            result["provider"] = name
            break

    result["account_number"] = None
    acc_m = re.search(r'account\s*(?:no|number)?\s*:?\s*([\d\-A-Z]+)', text, re.IGNORECASE)
    if acc_m:
        result["account_number"] = acc_m.group(1).strip()

    result["amount_due"] = (
        _find_amount(text, "amount due") or
        _find_amount(text, "amaun perlu dibayar") or
        _find_amount(text, "total amount") or
        _find_first_amount(text)
    )

    result["bill_date"] = _find_date(text, "bill date") or _find_date(text, "tarikh bil")
    result["due_date"] = _find_date(text, "due date") or _find_date(text, "tarikh akhir")

    # Units consumed (TNB)
    units_m = re.search(r'([\d,]+)\s*kwh', text, re.IGNORECASE)
    if units_m:
        try:
            result["units_consumed"] = float(units_m.group(1).replace(",", ""))
        except ValueError:
            pass

    return result


def extract_epf_statement(text: str) -> dict:
    result = {}

    result["member_name"] = _extract_name_after_label(text, ["name", "nama"])

    ic_m = IC_PATTERN.search(text)
    if ic_m:
        result["member_ic"] = f"{ic_m.group(1)}-{ic_m.group(2)}-{ic_m.group(3)}"

    year_m = re.search(r'(?:statement|year|tahun)\s*:?\s*(\d{4})', text, re.IGNORECASE)
    if year_m:
        result["statement_year"] = year_m.group(1)

    result["account1_closing"] = _find_amount(text, "account 1") or _find_amount(text, "akaun 1")
    result["account2_closing"] = _find_amount(text, "account 2") or _find_amount(text, "akaun 2")
    result["total_savings"] = _find_amount(text, "total savings") or _find_amount(text, "jumlah simpanan")

    div_m = re.search(r'dividend\s*(?:rate)?\s*:?\s*([\d.]+)\s*%', text, re.IGNORECASE)
    if div_m:
        result["dividend_rate"] = f"{div_m.group(1)}%"

    return result


# ── Router ────────────────────────────────────────────────────────────────────

EXTRACTORS = {
    "mykad": extract_mykad,
    "sg_nric": extract_sg_nric,
    "ssm_registration": extract_ssm_registration,
    "invoice": extract_invoice,
    "ea_form": extract_ea_form,
    "payslip": extract_payslip,
    "bank_statement": extract_bank_statement,
    "utility_bill": extract_utility_bill,
    "epf_statement": extract_epf_statement,
}


def extract_with_regex(doc_type: str, text: str) -> dict:
    """Route to the appropriate regex extractor for a document type."""
    extractor = EXTRACTORS.get(doc_type)
    if extractor is None:
        logger.debug(f"No regex extractor for doc_type={doc_type}")
        return {}
    try:
        return extractor(text)
    except Exception as e:
        logger.error(f"Regex extraction error for {doc_type}: {e}")
        return {}
