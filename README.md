### Detailed overview and instructions in `docs/overview.md` file# 

**Anomaly Detection using Isolation Forest**

This project implements **Isolation Forest (IF)** for detecting anomalies in data. Isolation Forest is an unsupervised learning algorithm that isolates anomalies instead of profiling normal instances, making it efficient for high-dimensional datasets.

## **Why Isolation Forest?**
- **Efficient:** Works well with large datasets.
- **Scalable:** Handles high-dimensional data effectively.

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

### evaluate with hypothetical ground truth
```bash
python evaluate.py
```

#### This script will run multiple experiments with different contamination values and store the results in `output/metadata.json` file

## **When to Use Isolation Forest?**
✅ Ideal for **unsupervised anomaly detection