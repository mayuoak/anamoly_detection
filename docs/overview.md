#  LLM & ML-Powered Text Processing Pipeline

##  Overview
This project implements a **pipeline for text processing** using **Large Language Models (LLMs)** and **Machine Learning (ML)** techniques. The pipeline consists of multiple agents that perform:
- **Data augmentation**
- **Entity recognition (ER)**
- **Sentiment analysis**
- **Theme extraction**
- **Vectorization (TF-IDF & OpenAI embeddings)**
- **Anomaly detection (Isolation Forest)**

## ️ Architecture
The pipeline is modular and built using **LangChain, OpenAI API, and Scikit-learn**.

## **1 LLM Design (LangChain & OpenAI)**
The pipeline leverages **LangChain's `ChatOpenAI` API** (GPT-based models) to handle NLP tasks.

### **BaseAgent (Parent Class for All Agents)**
- Uses `ChatOpenAI` for LLM-based processing.
- Implements `get_response()` to query OpenAI's API.
- Tracks API calls and logs metadata.

### **LLM-Powered Agents**
Each subclass of `BaseAgent` specializes in a specific NLP task:

| **Agent**         | **Task** |
|------------------|----------|
| `ParaphraseAgent` | Generates paraphrased variations of text. |
| `NERAgent`       | Extracts named entities from text. |
| `sentimentAgent` | Identifies sentiment (positive, negative, neutral). |
| `themeAgent`     | Detects themes in text. |
| `Vectorizer`      | Vectorizes text data into embeddings.|

- These agents **batch process text** to minimize latency and reduce API costs.

---
## **2 ML Design (Scikit-Learn & OpenAI Embeddings)**
In addition to LLMs, the pipeline includes **traditional ML models**.

### **Text Embeddings (Feature Engineering)**
The `vectorizer` class transforms text into numerical embeddings using:
1. **TF-IDF (`TfidfVectorizer`)** → Extracts term frequency-based representations.
2. **OpenAI Embeddings (`text-embedding-3-small`)** → Generates deep semantic embeddings. **(Used this for now)**

### **Anomaly Detection (`IsolationForest`)**
The `anomalyAgent` applies **Isolation Forest** to:
- Identify outliers in text embeddings.
- Assign an `anomaly_score` to each text.
- Uses embeddings + sentiment features concatenated for more feature coverage

---

## **3 Data Processing Pipeline (`PipelineManager`)**
The `PipelineManager` orchestrates the full **NLP pipeline**:

- 1 **Data Augmentation** (`augment_data()`) → Generates synthetic text variations.
- 2 **Feature Extraction**
    - Extracts **Entities**
    - Extracts **Sentiment**
    - Extracts **Themes**
- 3 **Vectorization** - Converts text into embeddings.
- 4 **Anomaly Detection** - Detects unusual texts.

 **Final Output:** Processed data is stored in:
- `output/augmented_data.csv`
- `output/features_data.csv`
- `output/vectorized_data.csv`
- `output/anomaly_data.csv`
- Metadata tracked in `metadata.json`

---

##  Key Features
- **LLM Integration:** OpenAI’s GPT model for various NLP tasks.
- **ML Techniques:** Uses TF-IDF and OpenAI embeddings for vectorization.
- **Anomaly Detection:** Identifies anomalies in text data.
- **Batch Processing:** Efficient API usage to optimize costs. `output/metadata.json` file also stores total api calls.
- **Modular & Scalable:** Each agent is **independent and reusable**.

---

##  Installation & Usage
### **1 Install Dependencies**
```bash
conda create --name anamoly_detection python -y
conda activate anamoly_detection
pip install -r requirements.txt
```
```python
import nltk
nltk.download('stopwords')
```

### **2 Run the Pipeline**
```bash
python run.py
```
---
### **3 Run experimentation**
```bash
python experiments.py
```
#### This script will run multiple experiments with different contamination values and store the results in `output/metadata.json` file
---

### Future work
- Experimenting with multiple vectorizers output concatenated. Need experimentation.
- Experimenting with multiple ML algorithms with voting between 2 or more ML algorithms (Local Outlier Factor (LOF) with isolationForest is what comes to mind)
- Trying out multiple LLMs in addition to OpenAI
    - Gemini might be better for summarization and Entity Recognition as they are trained on more longer contexts and more multimodal and multilingual data
- Adding more features like
    - **TF-IDF Score**
    - **Part-of-Speech (POS) Tag Ratios**