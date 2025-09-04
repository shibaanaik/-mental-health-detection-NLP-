# Mental health detection using NLP

A Natural Language Processing (NLP) project that detects mental health conditions from social media posts.
The model classifies text into five categories:

*🟦 Stress

*🟨 Depression

*🟩 Bipolar disorder

*🟥 Personality disorder

*🟪 Anxiety

This project uses TF-IDF + Random Forest to achieve high accuracy and is deployed as a Streamlit web app.

# 🚀 Features

## Preprocessing pipeline:

->Clean text (remove links, emails, special characters)
->Tokenization
->Stopword removal
->Stemming & Lemmatization
->Trained Random Forest Classifier
->82% accuracy on test data
->Interactive Streamlit web app for predictions
->Shows prediction confidence scores

# Installation & Setup
## 1. Clone repository
git clone https://github.com/yourusername/Mental-Health-Detection.git
cd Mental-Health-Detection

## Create environment (recommended with conda)
conda create -n mentalhealth python=3.10 -y
conda activate mentalhealth

## Install dependencies
pip install -r requirements.txt

## Run Streamlit app
streamlit run app.py

# Model Performance

Algorithm: Random Forest Classifier

Accuracy: ~82%

Macro F1-score: 0.82

# Future Improvements

Use deep learning models (BERT, RoBERTa, DistilBERT) for higher accuracy

Handle class imbalance better (SMOTE or weighted loss)

Add explainability (SHAP/LIME)

Deploy on cloud platforms (Streamlit Cloud / Hugging Face Spaces)
