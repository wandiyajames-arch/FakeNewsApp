import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import os
from pypdf import PdfReader

# ---------------------------------------------------------
# 0. Auto-Generate Sample Files for the Upload Tab
# ---------------------------------------------------------
def generate_samples():
    with open("sample_real_news.txt", "w", encoding="utf-8") as f:
        f.write("The United Nations held a summit in New York today to discuss global climate change initiatives. Leaders from over 50 countries signed a new agreement aiming to reduce carbon emissions by 20% over the next decade.")

    with open("sample_fake_news.txt", "w", encoding="utf-8") as f:
        f.write("BREAKING: The moon has declared independence from Earth and is now an autonomous space collective, demanding immediate entry into the interstellar council. A team of astronauts has been detained as leverage.")

generate_samples()

# ---------------------------------------------------------
# 1. Setup & Load Model
# ---------------------------------------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("Loading Deep Learning Model and Tokenizer...")
model = tf.keras.models.load_model('welfake_lstm_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 300 

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------------------------------------------------
# 2. Logic Functions
# ---------------------------------------------------------
def analyze_news(article_text):
    if not article_text or len(article_text.strip().split()) < 10:
        return "⚠️ Insufficient Input", "The LSTM requires at least 10 words to analyze sequential patterns."
        
    cleaned = preprocess_text(article_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)[0][0]
    
    if prediction > 0.5:
        return "🚨 FAKE NEWS", f"Confidence: {prediction*100:.2f}%\nAnalysis: The LSTM identified manipulative linguistic patterns characteristic of misinformation."
    else:
        return "✅ REAL NEWS", f"Confidence: {(1-prediction)*100:.2f}%\nAnalysis: The text aligns with the structural and narrative style of factual journalism."

def process_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return analyze_news(text)
    except:
        return "⚠️ Error", "Failed to extract text from URL. The site may be protected."

def process_file(file):
    try:
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(file.name)
            text = " ".join([page.extract_text() for page in reader.pages])
        else:
            with open(file.name, 'r', encoding='utf-8') as f:
                text = f.read()
        return analyze_news(text)
    except:
        return "⚠️ Error", "Failed to process the uploaded file."

# ---------------------------------------------------------
# 3. Attractive UI Construction
# ---------------------------------------------------------
custom_css = """
#header { text-align: center; padding: 25px; background: linear-gradient(90deg, #0c4a6e, #0284c7); color: white; border-radius: 12px; margin-bottom: 25px; }
.verdict_box textarea { font-size: 26px !important; text-align: center !important; font-weight: bold !important; }
.result_real textarea { color: #16a34a !important; border: 3px solid #16a34a !important; background: #f0fdf4 !important; }
.result_fake textarea { color: #dc2626 !important; border: 3px solid #dc2626 !important; background: #fef2f2 !important; }
footer {display: none !important;}
"""

with gr.Blocks(title="AIMS Digital Verifier") as interface:
    # --- Header ---
    gr.HTML("""
        <div id='header'>
            <h1>📰 Digital Content Verification System</h1>
            <p>Misinformation Detection powered by Deep Learning (LSTM)</p>
        </div>
    """)

    with gr.Row():
        # --- LEFT COLUMN ---
        with gr.Column(scale=2):
            with gr.Accordion("📖 Technical Methodology", open=False):
                gr.Markdown("""
                This system utilizes a **Long Short-Term Memory (LSTM)** neural network. 
                Unlike classical models, LSTMs retain 'memory' of previous words in a sequence, 
                allowing the system to detect subtle manipulative rhetoric and sensationalist narrative flows.
                """)

            with gr.Tabs():
                with gr.TabItem("📝 Text Analysis"):
                    text_input = gr.Textbox(label="Article Content", placeholder="Paste content here...", lines=8)
                    # Added Text Examples back
                    gr.Examples(
                        examples=[
                            ["The United Nations held a summit in New York today to discuss global climate change initiatives. Leaders from over 50 countries signed a new agreement."],
                            ["BREAKING: Scientists have discovered that eating chocolate every day actually makes you immortal, but only if you eat it while standing on one leg."]
                        ],
                        inputs=text_input,
                        label="💡 Sample Texts"
                    )
                    text_btn = gr.Button("Analyze Content", variant="primary")

                with gr.TabItem("🔗 URL Scraper"):
                    url_input = gr.Textbox(label="Article Link", placeholder="https://www.bbc.com/news/...")
                    gr.Examples(
                        examples=[["https://news.un.org/en/story/2024/02/1146602"]],
                        inputs=url_input,
                        label="💡 Sample URL"
                    )
                    url_btn = gr.Button("Scrape & Analyze", variant="primary")

                with gr.TabItem("📄 File Upload"):
                    file_input = gr.File(label="Upload News (.txt or .pdf)", file_types=[".txt", ".pdf"])
                    # Added File Examples back
                    gr.Examples(
                        examples=[["sample_real_news.txt"], ["sample_fake_news.txt"]],
                        inputs=file_input,
                        label="💡 Sample Files"
                    )
                    file_btn = gr.Button("Analyze Document", variant="primary")

        # --- RIGHT COLUMN ---
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Analysis Result")
            verdict = gr.Textbox(label="Classification", interactive=False, elem_classes=["verdict_box"])
            details = gr.Textbox(label="Confidence & Logic Analysis", interactive=False, lines=5)
            
            gr.Markdown("---")
            gr.Markdown("""
            **Academic Context:**
            - **Institution:** AIMS Senegal
            - **Model:** LSTM Neural Network
            - **Target Precision:** 97.12%
            - **Dataset:** WELFake (72,134 samples)
            """)

    # Styling Logic
    def style_verdict(v):
        if "REAL" in v: return gr.update(elem_classes=["verdict_box", "result_real"])
        if "FAKE" in v: return gr.update(elem_classes=["verdict_box", "result_fake"])
        return gr.update(elem_classes=["verdict_box"])

    # Handlers
    text_btn.click(analyze_news, inputs=text_input, outputs=[verdict, details]).then(style_verdict, verdict, verdict)
    url_btn.click(process_url, inputs=url_input, outputs=[verdict, details]).then(style_verdict, verdict, verdict)
    file_btn.click(process_file, inputs=file_input, outputs=[verdict, details]).then(style_verdict, verdict, verdict)

# Launch
if __name__ == "__main__":
    interface.launch(
        theme=gr.themes.Soft(primary_hue="sky", secondary_hue="slate"),
        css=custom_css
    )