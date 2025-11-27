# Mapply-ai-assessment

**Estimated time:** 4 hours  
**Language:** Python recommended (but any language is fine)

## Overview
This repository contains the starter code and instructions for the AI take-home assessment on Product Category Classification via Vector Embeddings & Similarity Search. Complete the tasks outlined in the provided assessment document. 

Build a prototype system that classifies new product descriptions into predefined Amazon categories by leveraging vector embeddings and nearest‑neighbor search.
Key Concepts Tested:
Generating vector embeddings for text
Building & querying a vector similarity index
K‑NN classification & evaluation

## Dataset
We will use the [bprateek/amazon_product_description](https://huggingface.co/datasets/bprateek/amazon_product_description) dataset. Use the `train` split for indexing and a 10% held-out test set.

Each record contains:
product_title (string)
category (string)
about_product (string)
product_specification (string)
category (string)

## Project structure
```
├── data/
│   └── download_and_prepare.py   # script to fetch & split HF data
├── src/
│   ├── embed.py                  # generates & saves embeddings
│   ├── index.py                  # builds & queries FAISS index
│   ├── classify.py               # K-NN classification logic
│   └── evaluate.py               # computes metrics & confusion matrix
├── notebooks/                    # optional EDA
├── outputs/
│   ├── embeddings/               # .npy or .json embeddings + labels
│   └── results/                  # evaluation CSVs & plots
├── requirements.txt              # or environment.yml
└── README.md
```

## Setup
1. Create a Python 3.8+ environment:
  ```
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate    # Windows
  ```
2. Install dependencies:
  ```
   pip install -r requirements.txt
  ```
3. (Optional) Verify the dataset loads:
  ```
   from datasets import load_dataset
   ds = load_dataset("bprateek/amazon_product_description", split="train")
   print(ds[0])
  ```
Notes:
You may use any programming language; place your implementation code under src/.
### Save all generated embeddings in the embeddings/ directory as persistent files (NumPy arrays, JSON, or your preferred format) so your similarity search routines can load and query them.


# Tasks & Deliverables:

1. **Data Preparation**  
   - Download the `train` split from [bprateek/amazon_product_description].  
   - **Split** into **index set (80%)** and **test set (20%)**, stratified by `category`.

2. **Embedding Generation**  
   - Use a pretrained model (e.g. `sentence-transformers/all-MiniLM-L6-v2`).  
   - Save embeddings to `outputs/embeddings/` (NumPy, JSON, etc.).  
   - Note your model choice in the README.

3. **Indexing**  
   - Build a FAISS index (or your DB of choice).  
   - Try at least two index types (e.g. `IndexFlatL2` vs. `IndexIVFFlat` vs. `IndexHNSWFlat`).  
   - Document indexing time & memory.

4. **Similarity-Based Classification**  
   - Implement K-NN (default **K=5**, justify choice).  
   - Predict categories for the test set.

5. **Evaluation**  
   - Compute accuracy, precision, recall, F1 per category.  
   - Produce a confusion matrix.
   - Write a short analysis (≤300 words).

6. **Submission**  
   - GitHub repo with: code, `requirements.txt`, `README.md`.  
   - One-page PDF summarizing approach, results, and next steps.

## Notes

- **Language**: you may use any language; ensure your README has the exact build/run steps.  
- **Embeddings**: persist them on disk (`.npy`, `.json`, or in a vector DB) so indexing can be repeated without re-embedding.

## Be sure to save your repo as YOUR_LAST_NAME-mapply-ai-assessment

Good luck!  
