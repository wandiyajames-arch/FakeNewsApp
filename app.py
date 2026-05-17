import gradio as gr
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from pypdf import PdfReader

print("--- Initializing Digital Content Verification System ---")

# 1. Load YOUR Custom BERT Transformer Model
print("Loading Proprietary BERT Pipeline...")
verifier_pipeline = pipeline(
    "text-classification", 
    model="./my_custom_welfake_bert",
    tokenizer="./my_custom_welfake_bert"
)

# 2. URL Scraping Engine
def scrape_article(url):
    """Fetches a URL and extracts all the paragraph text."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text()])
        
        if not article_text:
            return "Error: Could not extract text from this URL."
        return article_text
        
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

# 3. PDF Extraction Engine
def extract_pdf_text(file_obj):
    """Extracts raw text from an uploaded PDF document."""
    try:
        reader = PdfReader(file_obj.name)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
                
        if not text.strip():
            return "Error: Could not read text. This PDF might be scanned images rather than text."
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# 4. Ultra-Premium Text Analytics Dashboard
def generate_analytics(text):
    """Calculates basic structural analytics and builds a premium visual dashboard."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    # Average adult reading speed is ~200 words per minute
    reading_time = max(1, round(word_count / 200)) 
    
    analytics_html = f"""
    <div style="background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); padding: 35px; border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.4); text-align: center; color: white; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.7);">
        <h2 style="margin-top: 0; font-size: 26px; font-weight: 800; color: #93c5fd; letter-spacing: 1px; margin-bottom: 35px; text-transform: uppercase;">Content Insights</h2>
        
        <div style="display: flex; justify-content: space-around; align-items: center; font-size: 16px; color: #cbd5e1;">
            
            <div style="flex: 1;">
                <span style="font-size: 48px; font-weight: 900; color: #60a5fa; text-shadow: 0 0 15px rgba(96, 165, 250, 0.6);">{word_count:,}</span><br>
                <span style="text-transform: uppercase; letter-spacing: 2px; font-weight: bold; font-size: 12px; color: #94a3b8;">Words</span>
            </div>
            
            <div style="flex: 1; border-left: 1px solid rgba(255,255,255,0.1); border-right: 1px solid rgba(255,255,255,0.1);">
                <span style="font-size: 48px; font-weight: 900; color: #34d399; text-shadow: 0 0 15px rgba(52, 211, 153, 0.6);">{char_count:,}</span><br>
                <span style="text-transform: uppercase; letter-spacing: 2px; font-weight: bold; font-size: 12px; color: #94a3b8;">Characters</span>
            </div>
            
            <div style="flex: 1;">
                <span style="font-size: 48px; font-weight: 900; color: #f472b6; text-shadow: 0 0 15px rgba(244, 114, 182, 0.6);">~{reading_time}</span><br>
                <span style="text-transform: uppercase; letter-spacing: 2px; font-weight: bold; font-size: 12px; color: #94a3b8;">Min Read</span>
            </div>
            
        </div>
    </div>
    """
    return analytics_html

# 5. The Core Verification Logic
def verify_content(input_type, text_input, file_input):
    """Routes the input, runs analytics, runs BERT, and formats the output."""
    
    if input_type == "Paste Article Text":
        if not text_input: return "Please paste an article.", "", ""
        text_to_analyze = text_input
        
    elif input_type == "News URL":
        if not text_input: return "Please paste a URL.", "", ""
        text_to_analyze = scrape_article(text_input)
        
    elif input_type == "Upload PDF Document":
        if file_input is None: return "Please upload a PDF file.", "", ""
        text_to_analyze = extract_pdf_text(file_input)

    if text_to_analyze.startswith("Error"):
        return text_to_analyze, "", ""

    analytics_data = generate_analytics(text_to_analyze)

    truncated_text = " ".join(text_to_analyze.split()[:400])

    results = verifier_pipeline(truncated_text)[0]
    label = results['label']
    score = results['score']

    label_upper = label.upper()
    if "FAKE" in label_upper or label_upper == "LABEL_1": 
        verdict = "FAKE NEWS"
    else:
        verdict = "REAL NEWS"    
    
    confidence = f"{score * 100:.2f}%"
    
    return verdict, confidence, analytics_data

# 6. Ultra-Premium Cyberpunk Dark Mode CSS with Moving Gradient Background
custom_css = """
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

body, .gradio-container { 
    background: linear-gradient(-45deg, #0f172a, #1e1b4b, #111827, #0f172a) !important; 
    background-size: 400% 400% !important;
    animation: gradientBG 15s ease infinite !important;
    color: #f8fafc !important; 
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Glassmorphism containers for components */
.panel, .box, textarea, input, .wrap, .tabs { 
    background: rgba(30, 41, 59, 0.4) !important; 
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(59, 130, 246, 0.25) !important; 
    border-radius: 12px !important;
    color: white !important; 
    font-size: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Glowing focuses on text fields */
textarea:focus, input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important;
}

/* Headings and labels styling */
h1 { 
    font-size: 36px !important; 
    font-weight: 800 !important; 
    letter-spacing: -1px !important;
    background: linear-gradient(to right, #60a5fa, #a78bfa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

h2, h3, label, .markdown-text { color: #e2e8f0 !important; }

/* Premium Call-to-Action Action Button */
button.primary { 
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important; 
    border: none !important;
    color: white !important;
    font-size: 18px !important; 
    font-weight: 700 !important; 
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 14px !important; 
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    filter: brightness(1.1) !important;
}

button.primary:active {
    transform: translateY(1px) !important;
}
"""

# 7. Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as app:
    
    with gr.Row():
        gr.Markdown("""
        # Digital Content Verification Engine
        **Advanced NLP Node Architecture.** Analyze contextual authenticity matrices via fine-tuned Transformer states.
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_type = gr.Radio(
                choices=["Paste Article Text", "News URL", "Upload PDF Document"], 
                value="Paste Article Text", 
                label="Data Input Stream"
            )
            
            text_input = gr.Textbox(
                lines=6, 
                placeholder="Submit text payload or target network URL node for analysis...", 
                label="Source Payload"
            )
            
            file_input = gr.File(
                label="Binary Document Upload (PDF Format Only)",
                file_types=[".pdf"]
            )
            
            analyze_btn = gr.Button("Execute Analysis", variant="primary")
            
        with gr.Column(scale=1):
            output_verdict = gr.Textbox(label="Analysis Verdict Evaluation", lines=2)
            output_confidence = gr.Textbox(label="Model Token-Confidence Index")
            output_analytics = gr.HTML(label="Structural Visual Analytics Dashboard")
            
    analyze_btn.click(
        fn=verify_content, 
        inputs=[input_type, text_input, file_input], 
        outputs=[output_verdict, output_confidence, output_analytics]
    )

if __name__ == "__main__":
    app.launch()