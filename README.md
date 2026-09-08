# 📰 Fake News Detection System using AI & NLP

An AI-powered web application that detects potentially fake or misleading news using **Machine Learning, Natural Language Processing (NLP), real-time web search, trusted news sources, and Groq AI**.

## 🚀 Project Overview

The **Fake News Detection System** is designed to analyze news articles and determine whether the information is likely to be **REAL or FAKE**.

The system combines a trained Machine Learning model with real-time web-based verification and AI-powered fact-checking to provide a more comprehensive result.

### Key Technologies

* 🐍 Python
* 🌐 Flask
* 🤖 Machine Learning
* 🧠 Natural Language Processing (NLP)
* 📊 TF-IDF Vectorization
* ⚡ Passive Aggressive Classifier
* 🔎 Real-time Web Search
* 🤖 Groq AI / Llama
* 📰 Trusted News Source Verification
* 🎨 HTML, CSS & JavaScript

---

## ✨ Features

### 🤖 Machine Learning Detection

The system uses a **Passive Aggressive Classifier** trained on a large news dataset to classify news content.

### 🧠 NLP-Based Text Analysis

News text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)** before being passed to the Machine Learning model.

### 🔎 Real-Time Web Search

The system searches the web for relevant information related to the submitted news article.

### 📰 Trusted Source Detection

Search results from established news organizations such as Reuters, BBC, Associated Press and other trusted sources can be considered during verification.

### 🤖 AI Fact-Checking

Groq-powered Llama AI analyzes the available information and generates a structured fact-checking result.

### 🌐 Article URL Analysis

Users can provide a news article URL. The application attempts to extract the article content and analyze it.

### 📊 ML Evaluation

The project includes visualizations for:

* Class distribution
* Text length distribution
* Preprocessing comparison
* Train/test split
* Confusion matrix
* Precision, Recall and F1-score
* Cross-validation
* Model comparison
* Advanced validation
* Summary dashboard

---

## 🏗️ System Architecture

```text
                User
                  │
                  ▼
          News / Article URL
                  │
                  ▼
          Flask Web Application
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   Article Scraping     Text Processing
        │                   │
        └─────────┬─────────┘
                  ▼
            TF-IDF Vectorizer
                  │
                  ▼
       Passive Aggressive Classifier
                  │
                  ▼
          ML Prediction
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   Web Search          Trusted Sources
        │                   │
        └─────────┬─────────┘
                  ▼
             Groq / Llama AI
                  │
                  ▼
          Final Fact-Check
                  │
                  ▼
          REAL / FAKE Result
```

---

## 🧪 Machine Learning Methodology

### 1. Data Collection

The model was trained using a large news dataset containing real and fake news articles.

### 2. Text Preprocessing

The news text is processed before training and prediction.

Typical NLP preprocessing includes:

* Lowercase conversion
* Removal of unnecessary characters
* Text cleaning
* Tokenization
* Feature extraction

### 3. TF-IDF Vectorization

TF-IDF converts textual information into numerical feature vectors.

It gives higher importance to words that are useful for distinguishing documents while reducing the importance of very common words.

### 4. Passive Aggressive Classifier

The project uses a **Passive Aggressive Classifier (PAC)** for binary text classification.

The classifier is particularly suitable for large-scale text classification because it can learn efficiently from incoming training examples.

### 5. Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Cross-validation

---

## 📁 Project Structure

```text
fake-news-detection/
│
├── app.py
├── index.html
├── train.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
│
├── 1_class_distribution.png
├── 2_text_length_distribution.png
├── 3_avg_text_stats.png
├── 4_preprocessing_before_after.png
├── 5_train_test_split.png
├── 6_confusion_matrix.png
├── 7_precision_recall_f1.png
├── 8_summary_dashboard.png
├── 9_cross_validation.png
├── 10_train_test_cv_comparison.png
└── 11_advanced_validation.png
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/imranchoudhury461-sys/fake-news-detection.git
```

### 2. Open the project directory

```bash
cd fake-news-detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

The Groq API key should **not** be placed directly inside the source code.

Set it as an environment variable.

Windows:

```bash
set GROQ_API_KEY=your_groq_api_key
```

Linux / macOS:

```bash
export GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application

```bash
python app.py
```

The application will then be available locally through the Flask development server.

---

## 🔐 Security

API keys and other sensitive credentials should never be committed to GitHub.

The application reads the Groq API key through an environment variable:

```python
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

For deployment, configure `GROQ_API_KEY` through the hosting platform's environment-variable or secret settings.

---

## 📊 Project Visualizations

The repository contains visualizations demonstrating the model development and evaluation process.

These include:

* Dataset class distribution
* Text statistics
* Preprocessing results
* Dataset splitting
* Confusion matrix
* Precision/Recall/F1 evaluation
* Cross-validation
* Model comparison
* Advanced validation

---

## 🎯 Purpose of the Project

The purpose of this project is to demonstrate how **Machine Learning, Natural Language Processing, real-time information retrieval, and Generative AI** can be combined to create a practical news verification system.

The project is developed as an **MCA final-year project** and demonstrates the complete workflow from data preprocessing and model training to deployment and real-time prediction.

---

## ⚠️ Disclaimer

This system is an AI-assisted research and educational tool.

A prediction of **REAL** or **FAKE** should not be considered an absolute determination of truth. News verification can require human judgment and multiple reliable sources.

---

## 👨‍💻 Author

**Imran Choudhury**

MCA Project — Fake News Detection System using AI & NLP

---

## 📜 License

This project is intended for educational and research purposes.
