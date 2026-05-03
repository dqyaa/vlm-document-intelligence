"""
Document Intelligence Demo
===========================
Gradio interface for Malaysian document intelligence.
Deploy to Hugging Face Spaces or run locally.

Run:
    python demo/app.py
    # Open: http://localhost:7860
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

# Supported document types
DOC_TYPES = {
    "🔍 Auto-detect": None,
    "🪪 MyKad (Malaysian IC)": "mykad",
    "🪪 Singapore NRIC": "sg_nric",
    "📋 SSM Business Registration": "ssm_registration",
    "🧾 Invoice / Receipt": "invoice",
    "📑 LHDN EA Form": "ea_form",
    "💰 Payslip (Slip Gaji)": "payslip",
    "🏦 Bank Statement": "bank_statement",
    "💡 Utility Bill": "utility_bill",
    "💼 EPF/KWSP Statement": "epf_statement",
}

# Demo images (sample paths — use your own for real demo)
DEMO_DESCRIPTIONS = [
    "Upload a MyKad image to extract IC number, name, DOB...",
    "Upload an SSM certificate to extract registration details...",
    "Upload an invoice to extract amounts, line items, SST...",
    "Upload a payslip to extract salary breakdown...",
]

# Load extractor
_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        from extractors.vlm_extractor import VLMExtractor
        _extractor = VLMExtractor()
    return _extractor


def extract_document(image, doc_type_label: str) -> tuple[str, str, str]:
    """
    Main extraction function.
    Returns: (json_output, raw_text, status_message)
    """
    if image is None:
        return "{}", "", "⚠️ Please upload a document image."

    # Get doc_type from label
    doc_type = DOC_TYPES.get(doc_type_label)

    try:
        extractor = get_extractor()
        t0 = time.time()

        # Save PIL image to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            if hasattr(image, "save"):
                image.save(tmp.name, "JPEG")
            else:
                import shutil
                shutil.copy(image, tmp.name)
            tmp_path = tmp.name

        result = extractor.extract(
            image_path=tmp_path,
            doc_type=doc_type,
        )

        import os
        os.unlink(tmp_path)

        # Format output
        json_out = json.dumps(result.extracted_data, indent=2, ensure_ascii=False)

        status = (
            f"✅ **{result.document_type.replace('_', ' ').title()}** detected "
            f"({result.confidence:.0%} confidence) | "
            f"Model: {result.model_used} | "
            f"⚡ {result.latency_ms:.0f}ms"
        )

        if result.errors:
            status += f"\n⚠️ Errors: {', '.join(result.errors)}"
        if result.warnings:
            status += f"\nℹ️ {', '.join(result.warnings)}"

        return json_out, result.raw_text or "", status

    except Exception as e:
        return "{}", "", f"❌ Error: {str(e)}"


# ── Build Gradio UI ────────────────────────────────────────────────────────────

with gr.Blocks(
    title="🇲🇾 Malaysian Document Intelligence",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
        .status-box { padding: 12px; border-radius: 8px; }
        footer { display: none !important; }
    """
) as demo:

    gr.HTML("""
        <div style="text-align:center; padding:20px 0 10px;">
            <h1>🇲🇾 Malaysian Document Intelligence</h1>
            <p>Extract structured data from Malaysian documents using
            <strong>Qwen2.5-VL</strong> + <strong>EasyOCR</strong>.</p>
            <p>
                <strong>Supports:</strong>
                MyKad · SG NRIC · SSM Certificate · Invoice · EA Form ·
                Payslip · Bank Statement · Utility Bill · EPF Statement
            </p>
            <p style="font-size:0.85em; color:#666;">
                ⚠️ <em>Do not upload real personal documents in public demos.
                Use sample/dummy documents only.</em>
            </p>
            <p>
                Built by <a href="https://linkedin.com/in/aliyaalias">Aliya Alias</a> |
                <a href="https://github.com/aliyaalias19/vlm-document-intelligence">GitHub</a>
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📄 Upload Document Image",
                type="pil",
                height=350,
            )
            doc_type_input = gr.Dropdown(
                choices=list(DOC_TYPES.keys()),
                value="🔍 Auto-detect",
                label="Document Type",
            )
            extract_btn = gr.Button("🔍 Extract Data", variant="primary", size="lg")

            gr.HTML("""
                <div style="padding:12px; background:#f0f9ff; border-radius:8px; margin-top:12px;">
                    <h4 style="margin:0 0 8px;">Supported Documents</h4>
                    <ul style="margin:0; padding-left:20px; font-size:0.85em;">
                        <li>🪪 MyKad (Malaysian IC)</li>
                        <li>🪪 Singapore NRIC</li>
                        <li>📋 SSM Business Certificate</li>
                        <li>🧾 Tax Invoice / Receipt</li>
                        <li>📑 LHDN EA Form</li>
                        <li>💰 Payslip / Slip Gaji</li>
                        <li>🏦 Bank Statement</li>
                        <li>💡 Utility Bill (TNB, Unifi)</li>
                        <li>💼 EPF/KWSP Statement</li>
                    </ul>
                </div>
            """)

        with gr.Column(scale=2):
            status_output = gr.Markdown("Upload a document and click Extract.")

            with gr.Tabs():
                with gr.Tab("📊 Extracted Data (JSON)"):
                    json_output = gr.Code(
                        label="Structured Output",
                        language="json",
                        lines=20,
                    )
                with gr.Tab("📝 Raw OCR Text"):
                    raw_text_output = gr.Textbox(
                        label="OCR-extracted text",
                        lines=15,
                        interactive=False,
                    )

    # Wire up extraction
    extract_btn.click(
        fn=extract_document,
        inputs=[image_input, doc_type_input],
        outputs=[json_output, raw_text_output, status_output],
    )

    image_input.change(
        fn=lambda img, dt: ("", "", "Image uploaded. Click Extract to process.") if img else ("", "", ""),
        inputs=[image_input, doc_type_input],
        outputs=[json_output, raw_text_output, status_output],
    )

    gr.HTML("""
        <div style="margin-top:20px; padding:16px; background:#fff3cd; border-radius:8px;">
            <h3>⚙️ How It Works</h3>
            <ol style="margin:0; padding-left:20px;">
                <li><strong>Upload</strong> any Malaysian document image (JPG, PNG)</li>
                <li><strong>Auto-detect</strong> document type using keyword patterns</li>
                <li><strong>Extract</strong> text using Qwen2.5-VL (GPU) or EasyOCR (CPU)</li>
                <li><strong>Parse</strong> structured fields with type-specific prompts</li>
                <li><strong>Validate</strong> output against Pydantic schemas</li>
            </ol>
        </div>
        <div style="margin-top:12px; padding:16px; background:#f8f9fa; border-radius:8px;">
            <h3>🔧 Technical Details</h3>
            <ul style="margin:0; padding-left:20px;">
                <li><strong>Primary model:</strong> Qwen2.5-VL-7B-Instruct (free, Apache 2.0)</li>
                <li><strong>Fallback:</strong> EasyOCR (Malay + English) + regex extraction</li>
                <li><strong>Languages:</strong> Bahasa Malaysia, English, mixed</li>
                <li><strong>Validation:</strong> Pydantic v2 schemas per document type</li>
                <li><strong>Malaysian-specific:</strong> MyKad IC parsing, SSM format, MYR amounts</li>
            </ul>
        </div>
    """)

if __name__ == "__main__":
    print("🇲🇾 Starting Malaysian Document Intelligence Demo...")
    print("   Open: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
