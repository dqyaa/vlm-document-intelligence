"""
Document Schemas — Pydantic Models for Malaysian Documents
==========================================================
Structured output schemas for every supported document type.
Every field extracted from a document maps to one of these schemas.

Pydantic ensures:
- Type validation on every field
- Optional fields that return None gracefully
- Serialisation to JSON for API responses
- Clear contracts for downstream systems

Supported document types:
  Malaysian IC (MyKad)          — National identity card
  Singapore NRIC                — Singapore identity card
  SSM Business Registration     — Sole proprietor / Sdn Bhd certificate
  Malaysian Invoice             — Tax invoice / receipt
  LHDN EA Form                  — Employee tax form (from employer)
  Bank Statement                — Malaysian bank account statement
  Tabung Haji Statement         — Haji savings statement
  EPF Statement                 — Employee Provident Fund
  Payslip                       — Monthly salary slip
  Utility Bill                  — TNB / Unifi / Maxis bill
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import date
import re


# ── Base ──────────────────────────────────────────────────────────────────────

class DocumentBase(BaseModel):
    """Base class for all document extractions."""
    document_type: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    language: Optional[str] = None    # "BM" | "EN" | "mixed"
    extraction_notes: Optional[str] = None
    raw_text: Optional[str] = None    # Full OCR text (for debugging)


# ── 1. Malaysian IC / MyKad ───────────────────────────────────────────────────

class MyKadResult(DocumentBase):
    """
    Malaysian National Identity Card (MyKad / IC).

    Format: YYMMDD-SS-GGGG
    - YYMMDD: Date of birth
    - SS: State of birth (01-16 for MY states, 71-74 for territories, 82-83 for foreign-born)
    - GGGG: Random sequence number (odd = male, even = female)
    """
    document_type: Literal["mykad"] = "mykad"

    # Core fields (front of card)
    ic_number: Optional[str] = Field(None, description="12-digit IC number, may include dashes")
    full_name: Optional[str] = Field(None, description="Full name as on IC, all caps")
    date_of_birth: Optional[str] = Field(None, description="YYMMDD or DD/MM/YYYY")
    gender: Optional[Literal["male", "female", "unknown"]] = None
    nationality: Optional[str] = None    # "WARGANEGARA" / "PEMASTAUTIN TETAP" / etc.
    religion: Optional[str] = None       # Present on back of some ICs
    race: Optional[str] = None           # Malay/Chinese/Indian/etc.
    address: Optional[str] = None        # Full address from back of card
    state_of_birth: Optional[str] = None # Derived from IC SS digits

    # Derived
    age: Optional[int] = None
    birth_year: Optional[int] = None

    @field_validator("ic_number")
    @classmethod
    def validate_ic(cls, v):
        if v is None:
            return v
        digits = re.sub(r"\D", "", v)
        if len(digits) == 12:
            # Format with standard dashes: YYMMDD-SS-GGGG
            return f"{digits[:6]}-{digits[6:8]}-{digits[8:]}"
        return v


STATE_CODES = {
    "01": "Johor", "02": "Kedah", "03": "Kelantan", "04": "Melaka",
    "05": "Negeri Sembilan", "06": "Pahang", "07": "Pulau Pinang",
    "08": "Perak", "09": "Perlis", "10": "Selangor",
    "11": "Terengganu", "12": "Sabah", "13": "Sarawak",
    "14": "Wilayah Persekutuan KL", "15": "Wilayah Persekutuan Labuan",
    "16": "Wilayah Persekutuan Putrajaya",
    "71": "Foreign-born (Series 1)", "72": "Foreign-born (Series 2)",
    "82": "Foreign-born (Series 3)", "83": "Foreign-born (Series 4)",
}


# ── 2. Singapore NRIC ─────────────────────────────────────────────────────────

class SGNRICResult(DocumentBase):
    """Singapore National Registration Identity Card."""
    document_type: Literal["sg_nric"] = "sg_nric"

    nric_number: Optional[str] = None   # S/T/F/G + 7 digits + letter
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    race: Optional[str] = None
    country_of_birth: Optional[str] = None
    address: Optional[str] = None


# ── 3. SSM Business Registration Certificate ──────────────────────────────────

class SSMRegistrationResult(DocumentBase):
    """
    SSM (Suruhanjaya Syarikat Malaysia) Business Registration Certificate.
    Covers: Sole Proprietorship, Partnership, and Sdn Bhd certificates.
    """
    document_type: Literal["ssm_registration"] = "ssm_registration"

    # Core fields
    registration_number: Optional[str] = Field(None,
        description="Old format: XXXXXX-X or new 12-digit: 202XXXXXXXXXXX")
    business_name: Optional[str] = None    # Nama Perniagaan / Company Name
    business_type: Optional[str] = None    # Sole Proprietorship / Sdn Bhd / LLP
    registration_date: Optional[str] = None
    commencement_date: Optional[str] = None

    # Address
    registered_address: Optional[str] = None
    correspondence_address: Optional[str] = None

    # Owner / Director
    owner_name: Optional[str] = None       # For sole proprietors
    owner_ic: Optional[str] = None
    directors: Optional[list[str]] = None  # For companies

    # Business activity
    business_activity: Optional[str] = None
    msic_code: Optional[str] = None        # 5-digit Malaysia Standard Industrial Classification

    # Status
    status: Optional[str] = None           # "Active" / "Wound Up" / "Struck Off"
    expiry_date: Optional[str] = None      # Annual renewal date

    # Capital (for companies)
    paid_up_capital: Optional[str] = None
    authorised_capital: Optional[str] = None


# ── 4. Malaysian Tax Invoice ──────────────────────────────────────────────────

class InvoiceLineItem(BaseModel):
    """Single line item on an invoice."""
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    tax_code: Optional[str] = None   # SST / GST / Exempt


class MalaysianInvoiceResult(DocumentBase):
    """
    Malaysian tax invoice or receipt.
    Supports SST (Sales and Service Tax) format.
    """
    document_type: Literal["invoice"] = "invoice"

    # Header
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None

    # Seller
    seller_name: Optional[str] = None
    seller_ssm: Optional[str] = None      # SSM registration number
    seller_tax_id: Optional[str] = None   # SST/GST registration
    seller_address: Optional[str] = None
    seller_phone: Optional[str] = None
    seller_email: Optional[str] = None

    # Buyer
    buyer_name: Optional[str] = None
    buyer_ic: Optional[str] = None
    buyer_address: Optional[str] = None

    # Line items
    line_items: Optional[list[InvoiceLineItem]] = None

    # Totals
    subtotal: Optional[float] = None
    sst_rate: Optional[str] = None       # e.g. "6%" or "8%"
    sst_amount: Optional[float] = None
    discount: Optional[float] = None
    total_amount: Optional[float] = None
    currency: str = "MYR"

    # Payment
    payment_method: Optional[str] = None
    bank_details: Optional[str] = None


# ── 5. LHDN EA Form ───────────────────────────────────────────────────────────

class LHDNEAFormResult(DocumentBase):
    """
    LHDN EA Form (Borang EA) — Annual employee tax statement from employer.
    Provided by employers to employees for income tax filing.
    Assessment year is typically the year prior to filing.
    """
    document_type: Literal["ea_form"] = "ea_form"

    assessment_year: Optional[str] = None     # e.g. "2025" (for YA 2025)
    employer_name: Optional[str] = None
    employer_e_number: Optional[str] = None   # LHDN employer reference
    employer_address: Optional[str] = None

    # Employee details
    employee_name: Optional[str] = None
    employee_ic: Optional[str] = None
    employee_tax_file: Optional[str] = None   # SG/OG number

    # Income (all in MYR)
    gross_salary: Optional[float] = None
    bonus: Optional[float] = None
    allowances: Optional[float] = None
    benefits_in_kind: Optional[float] = None
    total_gross_income: Optional[float] = None

    # Deductions
    epf_employee: Optional[float] = None      # 11% contribution
    socso_employee: Optional[float] = None
    eis_employee: Optional[float] = None
    pcb_deducted: Optional[float] = None      # Monthly Tax Deduction

    # Net income
    net_income: Optional[float] = None


# ── 6. Malaysian Payslip ──────────────────────────────────────────────────────

class MalaysianPayslipResult(DocumentBase):
    """Malaysian monthly salary slip (gaji / payslip)."""
    document_type: Literal["payslip"] = "payslip"

    # Header
    employer_name: Optional[str] = None
    employee_name: Optional[str] = None
    employee_id: Optional[str] = None
    employee_ic: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    pay_period: Optional[str] = None    # e.g. "October 2024"

    # Earnings
    basic_salary: Optional[float] = None
    overtime: Optional[float] = None
    allowances: Optional[float] = None
    bonus: Optional[float] = None
    gross_pay: Optional[float] = None

    # Deductions
    epf_employee: Optional[float] = None
    socso_employee: Optional[float] = None
    eis_employee: Optional[float] = None
    income_tax_pcb: Optional[float] = None
    other_deductions: Optional[float] = None
    total_deductions: Optional[float] = None

    # Net
    net_pay: Optional[float] = None
    bank_account: Optional[str] = None   # Masked, e.g. ****1234


# ── 7. Malaysian Bank Statement ───────────────────────────────────────────────

class BankTransaction(BaseModel):
    """Single transaction from a bank statement."""
    date: Optional[str] = None
    description: Optional[str] = None
    reference: Optional[str] = None
    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None


class BankStatementResult(DocumentBase):
    """Malaysian bank account statement."""
    document_type: Literal["bank_statement"] = "bank_statement"

    bank_name: Optional[str] = None
    account_holder: Optional[str] = None
    account_number: Optional[str] = None   # May be masked
    account_type: Optional[str] = None     # Savings / Current / Islamic

    statement_period_from: Optional[str] = None
    statement_period_to: Optional[str] = None

    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    total_credits: Optional[float] = None
    total_debits: Optional[float] = None

    transactions: Optional[list[BankTransaction]] = None
    transaction_count: Optional[int] = None


# ── 8. Utility Bill ───────────────────────────────────────────────────────────

class UtilityBillResult(DocumentBase):
    """Malaysian utility bill — TNB electricity, Unifi/TM broadband, Maxis, etc."""
    document_type: Literal["utility_bill"] = "utility_bill"

    provider: Optional[str] = None     # "TNB" / "Unifi" / "Maxis" / "Syabas"
    account_number: Optional[str] = None
    account_holder: Optional[str] = None
    service_address: Optional[str] = None

    bill_date: Optional[str] = None
    due_date: Optional[str] = None
    billing_period_from: Optional[str] = None
    billing_period_to: Optional[str] = None

    amount_due: Optional[float] = None
    previous_balance: Optional[float] = None
    current_charges: Optional[float] = None
    total_amount: Optional[float] = None

    # Electricity specific
    units_consumed: Optional[float] = None   # kWh
    tariff_category: Optional[str] = None    # Domestic / Commercial


# ── 9. EPF / KWSP Statement ───────────────────────────────────────────────────

class EPFStatementResult(DocumentBase):
    """EPF (KWSP) annual statement."""
    document_type: Literal["epf_statement"] = "epf_statement"

    member_name: Optional[str] = None
    member_ic: Optional[str] = None
    membership_number: Optional[str] = None
    statement_year: Optional[str] = None

    # Account 1 (retirement)
    account1_opening: Optional[float] = None
    account1_contributions: Optional[float] = None
    account1_dividends: Optional[float] = None
    account1_withdrawals: Optional[float] = None
    account1_closing: Optional[float] = None

    # Account 2 (pre-retirement)
    account2_opening: Optional[float] = None
    account2_contributions: Optional[float] = None
    account2_dividends: Optional[float] = None
    account2_withdrawals: Optional[float] = None
    account2_closing: Optional[float] = None

    total_savings: Optional[float] = None
    dividend_rate: Optional[str] = None   # e.g. "5.50%"


# ── Document type registry ────────────────────────────────────────────────────

DOCUMENT_SCHEMAS = {
    "mykad": MyKadResult,
    "sg_nric": SGNRICResult,
    "ssm_registration": SSMRegistrationResult,
    "invoice": MalaysianInvoiceResult,
    "ea_form": LHDNEAFormResult,
    "payslip": MalaysianPayslipResult,
    "bank_statement": BankStatementResult,
    "utility_bill": UtilityBillResult,
    "epf_statement": EPFStatementResult,
}

DOCUMENT_DESCRIPTIONS = {
    "mykad": "Malaysian IC / MyKad (national identity card)",
    "sg_nric": "Singapore NRIC or FIN card",
    "ssm_registration": "SSM business registration certificate",
    "invoice": "Malaysian tax invoice or receipt",
    "ea_form": "LHDN EA Form (annual employee tax statement)",
    "payslip": "Malaysian monthly salary payslip",
    "bank_statement": "Malaysian bank account statement",
    "utility_bill": "Utility bill (TNB, Unifi, Maxis, etc.)",
    "epf_statement": "EPF/KWSP annual savings statement",
}
