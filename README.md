# 📰 Digital Content Verification System

### **Deep Learning for Automated Fact-Checking**

This project is a **Deep Learning-based web application** designed to verify the veracity of news articles. Developed as part of my MSc program at **AIMS Senegal**, the system utilizes Long Short-Term Memory (LSTM) networks to identify linguistic patterns associated with misinformation and fabricated content.

---

## 🚀 Live Demo

Access the live application here: **[Hugging Face Space: Digital Content Verification](https://huggingface.co/spaces/wandiya39/Digital-Content-Verification)**

---

## 🛠️ Features

The system provides a multidisciplinary approach to content verification through three primary input methods:

* **📝 Direct Text Analysis:** Paste raw text for immediate classification based on linguistic markers.
* **🔗 Live Web Scraping:** Input a URL to automatically extract article content using `BeautifulSoup4` and `Requests`.
* **📄 Document Parsing:** Upload `.txt` files directly for bulk analysis of local documents.
* **📊 Confidence Metrics:** Provides a percentage-based confidence score for every classification.

---

## 🧬 Technical Architecture

### **The Model**

* **Architecture:** Recurrent Neural Network (RNN) using **LSTM** layers for sequence processing and context retention.
* **Preprocessing:** Tokenization and padding of sequences (300 tokens) with **NLTK** stop-word removal.
* **Backend:** TensorFlow/Keras for model inference.

### **The Interface**

* **Gradio:** Used to build a responsive, tabbed UI that works across desktop and mobile devices.
* **Custom CSS:** Implemented color-coded feedback (Green for Factual, Red for Potential Misinformation) to enhance user experience.

---

## 🛡️ DevOps & CI/CD

This project demonstrates modern software engineering best practices:

* **Automated Deployment:** Integrated a GitHub Actions workflow (`sync.yml`) that automatically syncs code changes from this repository to Hugging Face Spaces.
* **Environment Hygiene:** Utilizes a strictly managed `.gitignore` and `requirements.txt` to ensure reproducible environments without bloated binary storage.

---

## 📂 Project Structure

```text
├── .github/workflows/
│   └── sync.yml            # CI/CD pipeline configuration
├── app.py                  # Main Gradio application logic
├── welfake_lstm_model.h5   # Pre-trained Deep Learning model
├── tokenizer.pkl           # Saved tokenizer for text preprocessing
├── requirements.txt        # Python dependencies
└── .gitignore              # Repository safety rules

```

---

## 👨‍💻 Author

**James Wandiya**

*MSc Student | Computer Science Professional | AIMS Senegal*

> This project is part of a larger research focus on **AI in Cybersecurity and NLP**. I am dedicated to building tools that foster trust in digital ecosystems.

---

### **How to use this:**

1. Go to your GitHub repository.
2. Click the **Edit** pencil icon on your `README.md`.
3. Delete the old `# FakeNewsApp` lines.
4. Paste this entire block of text in.
5. Click **Commit changes**.

Your repository will now look like a professional portfolio piece! Does this capture the "vibe" you were looking for?
