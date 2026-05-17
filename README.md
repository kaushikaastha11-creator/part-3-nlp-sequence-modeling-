# Part 3: NLP and Sequence Modeling Mini Project

## Overview
Customer Support Sentiment Analysis — classifying customer messages as **Negative**, **Neutral**, or **Positive** using both traditional ML and sequence-aware deep learning approaches.

## Dataset
`customer_support_text_classification.csv` — 1,500 records, 6 columns:

| Column | Description |
|--------|-------------|
| `ticket_id` | Unique support ticket identifier |
| `channel` | Communication channel (chat, phone, email, social) |
| `customer_message` | Raw customer text |
| `sentiment_label` | Target: negative / neutral / positive |
| `word_count` | Pre-computed word count |
| `urgent_flag` | Binary urgency indicator |

## Task Summary

| Task | Description | Result |
|------|-------------|--------|
| 1 | Dataset Understanding | 1,500 records, 3 balanced classes, avg 12.7 words |
| 2 | Text Preprocessing | Lowercase → Strip noise → Tokenize → Remove stopwords |
| 3 | Text Vectorization | TF-IDF (unigram+bigram) + Bag-of-Words |
| 4 | Baseline Models | LR+TF-IDF: 100% acc / NB+BoW: 100% acc (synthetic data) |
| 5 | Sequence Architecture | Bidirectional LSTM design with full layer explanation |
| 6 | Transformer Reflection | RNN limitations, LSTM gates, attention, transformers in GenAI |

## Repository Structure
```
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_evaluation.png
    ├── model_evaluation.csv
    └── sample_predictions.txt
```

## Setup & Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Key Findings

### Models
Both baseline models achieved **100% accuracy and F1** on the test set. This is expected for a synthetic dataset where each class has strongly distinctive vocabulary patterns — in real-world data, performance would typically be 75–90% and the LSTM/Transformer approach would provide a meaningful advantage.

### Why Text Must Be Vectorised
Machine learning models operate on matrices of numbers. Text vectorisation maps each document to a point in a high-dimensional feature space, enabling mathematical distance computations and gradient-based optimisation.

### Sequence Modelling Advantage
TF-IDF treats each document as a **bag** — order is lost. The sentence "not happy" has the same BoW representation as "happy not". A Bidirectional LSTM reads the sequence left-to-right and right-to-left, preserving order and capturing negations, intensifiers, and long-range context.

### Transformer Advantage
Transformers replace sequential processing with **parallel self-attention**, where every token attends to every other token simultaneously. This eliminates the vanishing gradient problem, enables massive parallelisation on GPUs, and scales to billions of parameters — the foundation of all modern large language models.

## Requirements
See `requirements.txt`
