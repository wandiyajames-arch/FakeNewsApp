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

# ---------------------------------------------------------
# 0. Auto-Generate Sample Files for the Upload Tab
# This ensures the files exist on Hugging Face automatically!
# ---------------------------------------------------------
with open("sample_real_news.txt", "w", encoding="utf-8") as f:
    f.write("The United Nations held a summit in New York today to discuss global climate change initiatives. Leaders from over 50 countries signed a new agreement aiming to reduce carbon emissions by 20% over the next decade.")

with open("sample_fake_news.txt", "w", encoding="utf-8") as f:
    f.write("BREAKING: The moon has declared independence from Earth and is now an autonomous space collective, demanding immediate entry into the interstellar council. A team of astronauts has been detained as leverage.")

# ---------------------------------------------------------
# 1. Setup NLTK
# ---------------------------------------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("Loading Deep Learning Model and Tokenizer...")

# ---------------------------------------------------------
# 2. Load the Model and Tokenizer
# ---------------------------------------------------------
model = tf.keras.models.load_model('welfake_lstm_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 300 

# ---------------------------------------------------------
# 3. Text Cleaning
# ---------------------------------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------------------------------------------------
# 4. Core Analysis Logic
# ---------------------------------------------------------
def analyze_news(article_text):
    if len(article_text.strip().split()) < 10:
        return (
            "⚠️ Error: Provide More Input",
            "The system requires at least a few sentences (minimum 10 words) to accurately analyze linguistic patterns."
        )
        
    cleaned = preprocess_text(article_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    
    prediction = model.predict(padded, verbose=0)[0][0]
    
    if prediction > 0.5:
        confidence = prediction * 100
        classification = f"🚨 FAKE NEWS"
        analysis = f"The Deep Learning model is {confidence:.1f}% confident that this text contains fabricated or manipulative linguistic patterns."
    else:
        confidence = (1 - prediction) * 100
        classification = f"✅ REAL NEWS"
        analysis = f"The Deep Learning model is {confidence:.1f}% confident that this text aligns with the standard, factual style of professional journalism."
        
    return classification, analysis

# ---------------------------------------------------------
# 5. Helper Functions for New Inputs
# ---------------------------------------------------------
def process_url(url):
    if not url.strip().startswith("http"):
        return "⚠️ Error: Invalid URL", "Please provide a complete web address starting with http:// or https://"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        
        if not article_text.strip():
            return "⚠️ Error: Extraction Failed", "Could not find readable article text on this webpage."
            
        return analyze_news(article_text)
    except Exception as e:
        return "⚠️ Error: Connection Failed", "Could not reach the website. The site may be protected against automated scrapers."

def process_file(file_path):
    if file_path is None:
        return "⚠️ Error: No File", "Please upload a document."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            article_text = f.read()
        return analyze_news(article_text)
    except Exception as e:
        return "⚠️ Error: Unreadable File", "Could not read the file. Please ensure it is a standard .txt document."

# ---------------------------------------------------------
# 6. Build the Tabbed UI with Examples
# ---------------------------------------------------------
custom_css = """
.result_textbox textarea { font-size: 20px !important; font-weight: bold !important; }
.result_real textarea { border: 2px solid #22c55e !important; background-color: #f0fdf4 !important; color: #166534 !important; }
.result_fake textarea { border: 2px solid #ef4444 !important; background-color: #fef2f2 !important; color: #991b1b !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky"), css=custom_css) as interface:
    
    gr.Markdown(
        """
        # 📰 Digital Content Verification System
        **MSc Project:** This Deep Learning framework utilizes Long Short-Term Memory (LSTM) networks to analyze the veracity of news articles.
        """
    )
    
    # --- INPUT TABS ---
    with gr.Tabs():
        
        # TAB 1: Paste Text
        with gr.TabItem("📝 Paste Text"):
            text_input = gr.Textbox(
                lines=6, label="Paste the news article here:", 
                placeholder="Minimum 10 words required..."
            )
            gr.Examples(
                examples=[
                    ["The United Nations held a summit in New York today to discuss global climate change initiatives. Leaders from over 50 countries signed a new agreement aiming to reduce carbon emissions by 20% over the next decade."],
                    ["BREAKING: Anonymous whistleblowers inside the World Economic Forum have leaked documents proving that all tap water in major cities will be replaced with a synthetic mind-control serum by the year 2028."]
                ],
                inputs=text_input,
                label="💡 Click an example to auto-fill:"
            )
            text_btn = gr.Button("🔍 Analyze Text", variant="primary")
            
        # TAB 2: URL Link
      # TAB 2: URL Link
        with gr.TabItem("🔗 Submit URL"):
            url_input = gr.Textbox(
                lines=1, label="Paste a link to a news article:", 
                placeholder="https://www.bbc.com/news/..."
            )
            # --- NEW SPECIFIC EXAMPLES ADDED HERE ---
            gr.Examples(
                examples=[
                    ["https://news.un.org/en/story/2024/02/1146602"], # Real, stable UN News article
                    ["https://www.theonion.com/study-finds-every-style-of-parenting-produces-terrible-1819575895"] # Fake/Satire article
                ],
                inputs=url_input,
                label="💡 Click an example URL to auto-fill:"
            )
            gr.Markdown("*Note: The system will automatically scrape the paragraph text from the webpage.*")
            url_btn = gr.Button("🔍 Scrape and Analyze Link", variant="primary")
            
        # TAB 3: File Upload
        with gr.TabItem("📄 Upload Document"):
            file_input = gr.File(
                label="Upload a plain text document (.txt)", 
                file_types=[".txt"]
            )
            gr.Examples(
                examples=[
                    ["sample_real_news.txt"],
                    ["sample_fake_news.txt"]
                ],
                inputs=file_input,
                label="💡 Click an example file to auto-upload:"
            )
            file_btn = gr.Button("🔍 Analyze Document", variant="primary")

    # --- RESULTS SECTION ---
    gr.Markdown("---")
    with gr.Group():
        gr.Markdown("### 📊 Classification & Linguistic Analysis")
        with gr.Row():
            verdict_output = gr.Textbox(label="System Classification", interactive=False)
            explanation_output = gr.Textbox(label="Linguistic Pattern Analysis", interactive=False)

    # Function to update styling
    def apply_result_styling(verdict):
        if "REAL" in verdict:
            return gr.update(elem_classes=["result_textbox", "result_real"])
        elif "FAKE" in verdict:
            return gr.update(elem_classes=["result_textbox", "result_fake"])
        return gr.update(elem_classes=[])

    # Connect buttons
    text_btn.click(fn=analyze_news, inputs=text_input, outputs=[verdict_output, explanation_output]) \
        .then(fn=apply_result_styling, inputs=verdict_output, outputs=verdict_output)
        
    url_btn.click(fn=process_url, inputs=url_input, outputs=[verdict_output, explanation_output]) \
        .then(fn=apply_result_styling, inputs=verdict_output, outputs=verdict_output)
        
    file_btn.click(fn=process_file, inputs=file_input, outputs=[verdict_output, explanation_output]) \
        .then(fn=apply_result_styling, inputs=verdict_output, outputs=verdict_output)

if __name__ == "__main__":
    interface.launch()