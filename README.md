#  Advanced Medical Record Document Clustering and Analysis System

> **A scalable, AI-powered platform for organizing, analyzing, and interacting with unstructured electronic health records (EHRs)**

---

## Project Overview

This project tackles one of the most pressing problems in healthcare: **navigating unstructured, lengthy medical records.**

Using advanced natural language processing, machine learning, and generative AI, this system can:

* **Cluster** related pages from large PDF documents—even when scattered
* **Maintain metadata continuity** (Date of Service, Provider, Facility, etc.)
* **Interact** with medical documents through a conversational AI interface

It helps doctors, clinicians, and medical staff **quickly understand patient histories**, extract critical insights, and make better decisions.

---

##  Objectives

* **Page Clustering:** Group semantically similar pages within massive EHRs
* **Data Continuity:** Propagate metadata like headers, provider, and date across related pages
* **Information Extraction:** Pull key data like diagnoses, medications, and procedures
* **Interactive Analysis:** Provide a GUI for structured visualization
* **Document Chat:** Allow natural language querying using conversational AI

---

## Technical Architecture

### 1. Document Ingestion & Processing

* Built a modular document parser using a base `DocumentProcessor` class
* Supported formats:

  * PDF (`PyMuPDF`, `PyPDF2`)
  * DOC/DOCX (`python-docx`)
  * XLS/XLSX (`pandas`, `openpyxl`)
  * CSV (`pandas`)

### 2. Medical NLP & Entity Recognition

* Used **regex** for extracting:

  * Diagnoses
  * Medications
  * Procedures
* Applied **medical sentiment analysis**: *Concerning*, *Neutral*, *Optimistic*
* Custom logic to detect:

  * Names
  * Dosages
  * Medical abbreviations

### 3. Page Clustering (Unsupervised ML)

* **Sentence Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2)
* **Dimensionality Reduction:** UMAP
* **Clustering Algorithm:** HDBSCAN (handles arbitrary shaped clusters and noise)
* Clusters fragmented records even if non-contiguous

### 4. Interactive Streamlit UI

* **Clustering Dashboard:** Visualize page clusters
* **Document Explorer:** Page-wise entity extraction
* **Report Generator:** Download structured summaries
* **Chat Interface:** Ask questions about the document in plain English

### 5. AI Document Chat

* Integrated **Google Gemini 1.5 Flash model** for contextual QA
* Chunked document text using `RecursiveCharacterTextSplitter`
* Created FAISS-based **vector store** for retrieval-augmented generation (RAG)
* Maintains **chat history and context** for conversational flow

---

##  Key Innovations

| Feature                       | Description                                                     |
| ----------------------------- | --------------------------------------------------------------- |
|  Multi-format support       | Reads PDF, DOCX, XLSX, CSV with fallback parsing                |
|  Embedding-based clustering | Uses Sentence Transformers + UMAP + HDBSCAN                     |
|  Medical-specific NLP       | Regex, sentiment, and entity detection tailored for healthcare  |
|  Chat with EHR              | Gemini-powered document QA with memory and vector retrieval     |
|  Context preservation       | Maintains Date of Service, Provider, Facility through all pages |

---

##  System Pipeline

```plaintext
[Input: PDF/DOCX/XLSX/CSV]
        │
        ▼
[Text Extraction + Page Separation]
        │
        ▼
[Sentence Embedding → UMAP → HDBSCAN]
        │
        ▼
[Clustering + Metadata Propagation]
        │
        ├─► [Medical Entity Extraction]
        ├─► [Sentiment Analysis]
        ├─► [Interactive Streamlit Dashboard]
        └─► [Chatbot with Gemini + FAISS Retrieval]
```



## Real-World Applications

*  **Hospitals & Clinics:** Faster diagnosis & patient management
*  **Insurance Firms:** Automated claim review and verification
*  **Research Labs:** Large-scale medical text analysis
*  **EHR Providers:** Backend tools for smarter document storage & retrieval

---

##  Tech Stack

| Component        | Library / Tool                          |
| ---------------- | --------------------------------------- |
| Backend          | Python                                  |
| NLP              | HuggingFace Transformers, Regex, NLTK   |
| Clustering       | Sentence Transformers, UMAP, HDBSCAN    |
| Vector DB        | FAISS                                   |
| Chat Integration | Google Generative AI (Gemini 1.5 Flash) |
| Frontend UI      | Streamlit                               |
| Document Parsing | PyMuPDF, python-docx, pandas            |

---

##  How to Run

1. **Clone the repo**

```bash
git clone https://github.com/Sudipta2307/AI-powered-Document-Segmentation-and-Classification-System.git
cd medical-doc-analyzer
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run Streamlit app**

```bash
streamlit run app.py
```




