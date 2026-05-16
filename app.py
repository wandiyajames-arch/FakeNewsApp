import gradio as gr
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from pypdf import PdfReader

print("--- Initializing Premium Digital Content Verification System ---")

# 1. Load the BERT Transformer Model
print("Loading BERT Pipeline...")
verifier_pipeline = pipeline(
    "text-classification", 
    model="hamzab/roberta-fake-news-classification" 
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

# 4. HUGE Text Analytics Generator
def generate_analytics(text):
    """Calculates basic structural analytics and builds a massive visual dashboard."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    # Average adult reading speed is ~200 words per minute
    reading_time = max(1, round(word_count / 200)) 
    
    # Completely redesigned HTML for massive visibility and contrast
    analytics_html = f"""
    <div style="background-color: #1e293b; padding: 30px; border-radius: 15px; border: 3px solid #3b82f6; text-align: center; color: white; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);">
        <h2 style="margin-top: 0; font-size: 32px; font-weight: bold; color: #93c5fd; margin-bottom: 30px;">📊 Content Analytics</h2>
        
        <div style="display: flex; justify-content: space-around; align-items: center; font-size: 18px; color: #cbd5e1;">
            
            <div style="flex: 1;">
                <span style="font-size: 54px; font-weight: 900; color: #60a5fa;">{word_count:,}</span><br>
                <span style="text-transform: uppercase; letter-spacing: 2px; font-weight: bold;">Words</span>
            </div>
            
            <div style="flex: 1; border-left: 2px solid #475569; border-right: 2px solid #475569;">
                <span style="font-size: 54px; font-weight: 900; color: #34d399;">{char_count:,}</span><br>
                <span style="text-transform: uppercase; letter-spacing: 2px; font-weight: bold;">Characters</span>
            </div>
            
            <div style="flex: 1;">
                <span style="font-size: 54px; font-weight: 900; color: #f472b6;">~{reading_time}</span><br>
                <span style="text-transform: uppercase; letter-spacing: 2px; font-weight: bold;">Min Read</span>
            </div>
            
        </div>
    </div>
    """
    return analytics_html

# 5. The Core Verification Logic
def verify_content(input_type, text_input, file_input):
    """Routes the input, runs analytics, runs BERT, and formats the output."""
    
    # Route 1: Direct Text
    if input_type == "Paste Article Text":
        if not text_input: return "Please paste an article.", "", ""
        text_to_analyze = text_input
        
    # Route 2: URL Scraper
    elif input_type == "News URL":
        if not text_input: return "Please paste a URL.", "", ""
        text_to_analyze = scrape_article(text_input)
        
    # Route 3: PDF Document
    elif input_type == "Upload PDF Document":
        if file_input is None: return "Please upload a PDF file.", "", ""
        text_to_analyze = extract_pdf_text(file_input)

    # Catch any scraping or PDF reading errors
    if text_to_analyze.startswith("Error"):
        return text_to_analyze, "", ""

    # Generate Analytics before truncating the text
    analytics_data = generate_analytics(text_to_analyze)

    # BERT truncation
    truncated_text = " ".join(text_to_analyze.split()[:400])

    # Run the Transformer
    results = verifier_pipeline(truncated_text)[0]
    label = results['label']
    score = results['score']

    # Format the Verdict
    label_upper = label.upper()
    if "FAKE" in label_upper or label_upper == "LABEL_0": 
        verdict = "🚨 HIGH RISK: LIKELY FABRICATED OR FAKE NEWS"
    else:
        verdict = "✅ VERIFIED: LIKELY REAL NEWS"    
    
    confidence = f"{score * 100:.2f}%"
    
    return verdict, confidence, analytics_data

# 6. Custom CSS to force a beautiful Dark Mode
custom_css = """
body, .gradio-container { background-color: #0f172a !important; color: #f8fafc !important; }
.panel, .box, textarea, input, .wrap { background-color: #1e293b !important; border: 1px solid #3b82f6 !important; color: white !important; font-size: 16px !important;}
h1, h2, h3, label, .markdown-text { color: #f8fafc !important; }
button.primary { font-size: 20px !important; font-weight: bold !important; padding: 15px !important; }
"""

# 7. Build the Gradio Interface
# We use Base theme, but the custom CSS above will convert it into a premium Navy Dark Mode
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as app:
    
    with gr.Row():
        gr.Markdown("""
        # 🛡️ Premium Digital Content Verification
        **Powered by BERT Architecture.** Instantly verify the authenticity of raw text, live URLs, and PDF documents.
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input Section
            input_type = gr.Radio(
                choices=["Paste Article Text", "News URL", "Upload PDF Document"], 
                value="Paste Article Text", 
                label="Select Input Method"
            )
            
            text_input = gr.Textbox(
                lines=5, 
                placeholder="Paste the news text or a live link here...", 
                label="Text or URL Input"
            )
            
            file_input = gr.File(
                label="PDF Upload (Use this if 'Upload PDF Document' is selected)",
                file_types=[".pdf"]
            )
            
            analyze_btn = gr.Button("Analyze Content", variant="primary")
            
        with gr.Column(scale=1):
            # Output Section
            output_verdict = gr.Textbox(label="System Verdict", lines=2)
            output_confidence = gr.Textbox(label="AI Confidence Score")
            
            # Massive Analytics Section
            output_analytics = gr.HTML(label="Article Analytics")
            
    # Connect everything together
    analyze_btn.click(
        fn=verify_content, 
        inputs=[input_type, text_input, file_input], 
        outputs=[output_verdict, output_confidence, output_analytics]
    )

if __name__ == "__main__":
    app.launch()