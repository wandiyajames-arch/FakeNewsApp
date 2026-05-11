<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=200&section=header&text=Digital%20Content%20Verification&fontSize=38&fontColor=e94560&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20for%20Automated%20Fact-Checking&descAlignY=58&descColor=a8b2d8" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/wandiya39/Digital-Content-Verification)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)

<br/>

> **Leveraging LSTM-based deep learning to detect misinformation and protect the integrity of digital information.**

<br/>

[🚀 Live Demo](https://huggingface.co/spaces/wandiya39/Digital-Content-Verification) &nbsp;·&nbsp;
[📖 Documentation](#-technical-architecture) &nbsp;·&nbsp;
[👨‍💻 Author](#-author) &nbsp;·&nbsp;
[🐛 Report Bug](https://github.com/wandiya39/Digital-Content-Verification/issues)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Technical Architecture](#-technical-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [DevOps & CI/CD](#️-devops--cicd)
- [Results & Performance](#-results--performance)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🔍 Overview

The **Digital Content Verification System** is a research-grade web application developed as part of an MSc thesis at **AIMS Senegal**. In an era of rapidly spreading misinformation, this tool provides an automated, AI-driven pipeline for assessing the credibility of news content.

The system employs a **Long Short-Term Memory (LSTM)** recurrent neural network — a sequence-aware deep learning architecture — to detect subtle linguistic patterns that distinguish factual reporting from fabricated content. The model was trained on the **WELFake dataset**, a benchmark corpus curated for fake news detection research.

```
Input (Text / URL / .txt File)
        │
        ▼
  Preprocessing Layer
  (Tokenization → Stop-word Removal → Padding to 300 tokens)
        │
        ▼
   LSTM Network
  (Sequence Encoding → Context Retention → Dense Output)
        │
        ▼
  Classification Result
  ✅ FACTUAL  or  ⚠️ POTENTIAL MISINFORMATION
  + Confidence Score (%)
```

---

## 🚀 Live Demo

The application is deployed and publicly accessible on Hugging Face Spaces:

<div align="center">

### ➡️ [**huggingface.co/spaces/wandiya39/Digital-Content-Verification**](https://huggingface.co/spaces/wandiya39/Digital-Content-Verification)

*No setup required. Runs entirely in your browser.*

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 **Direct Text Analysis** | Paste any raw text for immediate classification using linguistic marker detection |
| 🔗 **Live Web Scraping** | Submit a URL; the system extracts article content via `BeautifulSoup4` and `Requests` |
| 📄 **Document Parsing** | Upload `.txt` files for batch analysis of local documents |
| 📊 **Confidence Scoring** | Every prediction is accompanied by a percentage-based confidence metric |
| 🎨 **Color-Coded Feedback** | Visual cues: **Green** for Factual content, **Red** for Potential Misinformation |
| 📱 **Responsive UI** | Gradio-powered tabbed interface that works across desktop and mobile |

---

## 🧬 Technical Architecture

### Model Design

The core of the system is a **Recurrent Neural Network** built with LSTM cells, chosen for their ability to capture long-range dependencies in text — a critical capability for detecting nuanced misinformation patterns.

```
Embedding Layer  →  LSTM Layer(s)  →  Dropout  →  Dense (Sigmoid)
```

| Component | Detail |
|---|---|
| **Architecture** | LSTM-based RNN |
| **Framework** | TensorFlow / Keras |
| **Sequence Length** | 300 tokens (padded/truncated) |
| **Preprocessing** | NLTK tokenization + stop-word removal |
| **Training Dataset** | WELFake (a merged fake news benchmark corpus) |
| **Output** | Binary classification + confidence probability |

### Application Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow 2.x, Keras |
| **NLP Preprocessing** | NLTK |
| **Web Scraping** | BeautifulSoup4, Requests |
| **UI Framework** | Gradio |
| **Deployment** | Hugging Face Spaces |
| **CI/CD** | GitHub Actions |

---

## 📂 Project Structure

```
digital-content-verification/
│
├── .github/
│   └── workflows/
│       └── sync.yml              # CI/CD: auto-sync to Hugging Face Spaces
│
├── app.py                        # Main Gradio application & inference logic
├── welfake_lstm_model.h5         # Pre-trained LSTM model weights
├── tokenizer.pkl                 # Serialized tokenizer for text preprocessing
│
├── requirements.txt              # Pinned Python dependencies
└── .gitignore                    # Repository hygiene rules
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/wandiya39/Digital-Content-Verification.git
cd Digital-Content-Verification

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python app.py
```

The Gradio interface will be accessible at `http://localhost:7860` in your browser.

### Usage

1. **Text Tab** — Paste any news excerpt directly into the text box and click *Analyze*.
2. **URL Tab** — Enter the full URL of a news article; the system will scrape and classify it.
3. **File Tab** — Upload a `.txt` file containing article content for offline analysis.

---

## 🛡️ DevOps & CI/CD

```
GitHub Repository (main branch)
        │
        │  git push
        ▼
GitHub Actions (sync.yml)
        │
        │  Automated sync on every commit
        ▼
Hugging Face Spaces
        │
        │  Live rebuild
        ▼
Public Application (zero-downtime)
```

- **Continuous Deployment** — `sync.yml` automatically mirrors every commit to Hugging Face Spaces.
- **Reproducible Environments** — Strictly managed `requirements.txt` ensures consistent dependency resolution.
- **Clean Repository** — `.gitignore` prevents accidental commits of large binary files.

---

## 📈 Results & Performance

> *Detailed evaluation metrics will be added upon completion of the full research paper.*

The model was trained and evaluated on the **WELFake dataset**, which aggregates four prominent fake news corpora to reduce sampling bias. Preliminary results demonstrate strong performance on held-out test data, with the LSTM architecture capturing contextual and sequential patterns that simpler bag-of-words models miss.

---

## 🗺️ Roadmap

- [ ] Add transformer-based model (BERT/DistilBERT) for improved accuracy
- [ ] Expand language support beyond English
- [ ] Integrate source credibility scoring alongside content analysis
- [ ] Publish full evaluation metrics and confusion matrix
- [ ] Add explainability layer (attention visualization)

---

## 👨‍💻 Author

<div align="center">

### James Wandiya

**MSc Student — Big Data & Data Science**
**African Institute for Mathematical Sciences (AIMS), Senegal**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/james-wandiya-87b90912a)
[![Hugging Face](https://img.shields.io/badge/🤗-wandiya39-FFD21E?style=for-the-badge)](https://huggingface.co/wandiya39)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/wandiya39)

</div>

> This project is part of a broader research focus on **AI applications in Cybersecurity and Natural Language Processing**. My work aims to build tools that foster trust, transparency, and accountability in digital ecosystems.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=100&section=footer" width="100%"/>

*Built with ❤️ at AIMS Senegal · © 2025 James Wandiya*

</div>
