import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

# 1. Setup NLTK (Mandatory for cleaning text)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

print("Loading Deep Learning Model and Tokenizer...")

# 2. Load the Model and Tokenizer
model = tf.keras.models.load_model('welfake_lstm_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 300 

# 3. Text Cleaning
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# 4. Modified Analysis Logic
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
        analysis = f"The Deep Learning model is {confidence:.1f}% confident that this article contains fabricated or manipulative linguistic patterns."
    else:
        confidence = (1 - prediction) * 100
        classification = f"✅ REAL NEWS"
        analysis = f"The Deep Learning model is {confidence:.1f}% confident that this article aligns with the standard, factual style of professional journalism."
        
    return classification, analysis

# Custom CSS for the glowing result effect
custom_css = """
.result_textbox textarea {
    font-size: 20px !important;
    font-weight: bold !important;
}
.result_real textarea {
    border: 2px solid #22c55e !important;
    background-color: #f0fdf4 !important;
    color: #166534 !important;
}
.result_fake textarea {
    border: 2px solid #ef4444 !important;
    background-color: #fef2f2 !important;
    color: #991b1b !important;
}
"""

# 5. Build the UI
# NOTE: theme and css moved to .launch() for Gradio 6.0 compatibility
with gr.Blocks() as interface:
    
    gr.Markdown(
        """
        # 📰 Digital Content Verification System
        **MSc Project:** This Deep Learning framework utilizes Long Short-Term Memory (LSTM) networks to analyze the veracity of news articles.
        """
    )
    
    with gr.Group():
        gr.Markdown("### 1. Input News Article Text")
        news_input = gr.Textbox(
            lines=8, 
            label="Paste the full news article here for classification:", 
            placeholder="Minimum 10 words required..."
        )
        
    analyze_btn = gr.Button("🔍 Analyze Article Veracity", variant="primary")
    
    gr.Markdown("---")
    with gr.Group():
        gr.Markdown("### 2. Analysis Results")
        with gr.Row():
            verdict_output = gr.Textbox(
                label="System Classification", 
                placeholder="Awaiting input...", 
                interactive=False
            )
            explanation_output = gr.Textbox(
                label="Linguistic Pattern Analysis", 
                placeholder="Awaiting input...", 
                interactive=False
            )
        
    gr.Markdown("---")
    with gr.Group():
        gr.Markdown("### 💡 Try an Example Article:")
        gr.Examples(
            examples=[
                ["The United Nations held a summit in New York today to discuss global climate change initiatives. Leaders from over 50 countries signed a new agreement aiming to reduce carbon emissions by 20% over the next decade."],
                ["BREAKING: The moon has declared independence from Earth and is now an autonomous space collective, demanding immediate entry into the interstellar council."]
            ],
            inputs=news_input
        )

    # Function to update styling
    def apply_result_styling(verdict):
        if "REAL" in verdict:
            return gr.update(elem_classes=["result_textbox", "result_real"])
        elif "FAKE" in verdict:
            return gr.update(elem_classes=["result_textbox", "result_fake"])
        return gr.update(elem_classes=[])

    analyze_btn.click(
        fn=analyze_news, 
        inputs=news_input, 
        outputs=[verdict_output, explanation_output]
    ).then(
        fn=apply_result_styling,
        inputs=verdict_output,
        outputs=verdict_output
    )

# 6. Launch with Style
if __name__ == "__main__":
    # Theme and CSS are passed here in newer Gradio versions
    interface.launch(theme=gr.themes.Soft(primary_hue="sky"), css=custom_css)