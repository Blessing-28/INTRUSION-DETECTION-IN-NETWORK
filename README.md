# Network Intrusion Detection System (NIDS)

## Table of Contents

- [Project Overview](#project-overview)  
- [Data Sources](#data-sources)  
- [Tools](#tools)  
- [Data Cleaning/Preparation](#data-cleaningpreparation)  
- [Exploratory Data Analysis](#exploratory-data-analysis)  
- [Model Development](#model-development)  
- [Results/Findings](#resultsfindings)  
- [Recommendations](#recommendations)  
- [Limitations](#limitations)  

### Project Overview
This project implements a **Network Intrusion Detection System (NIDS)** using supervised machine learning techniques. The system leverages an **ensemble learning approach (Stacking Classifier)** to combine multiple base models (Linear Regression, K‑Nearest Neighbors (KNN), and Gaussian Naive Bayes, with a Logistic Regression meta‑model). The goal is to accurately classify network traffic as either **normal** or **malicious**, providing robust detection of intrusions while minimizing false positives and false negatives.

### Data Sources
- **Kaggle Network Traffic Dataset** – Real‑time network traffic data containing features such as IP addresses, port numbers, protocols, packet sizes, flags, and derived statistical features.  
- **Preprocessed Feature Vectors** – Extracted attributes structured for supervised learning, labeled as *normal* or *attack*.  


### Tools
- **Python** – Core programming language  
- **Scikit‑learn** – Machine learning models (LinearRegression, KNeighborsClassifier, GaussianNB, LogisticRegression, StackingClassifier)  
- **Pandas & NumPy** – Data preprocessing and transformation  
- **Matplotlib & Seaborn** – Visualization (confusion matrix, ROC curve, scatter plots, class distribution)  
- **Jupyter Notebook** – Experimentation and reporting  

### Data Cleaning/Preparation
Steps performed:
1. Conversion of categorical features (`protocol_type`, `service`, `flag`, `class`) into numerical codes.  
2. Handling missing values and noisy data.  
3. Feature scaling using **StandardScaler**.  
4. Splitting dataset into **70% training** and **30% testing**.  

```python
train_data['protocol_type'] = train_data['protocol_type'].astype('category').cat.codes
train_data['service'] = train_data['service'].astype('category').cat.codes
train_data['flag'] = train_data['flag'].astype('category').cat.codes
train_data['class'] = train_data['class'].astype('category').cat.codes
```

---

### Exploratory Data Analysis
EDA focused on:
- **Class distribution** – Balanced dataset (≈11,800 normal vs. ≈13,400 anomalies).  
- **Feature relationships** – Scatterplots of `duration` vs. `num_compromised` showed most connections had zero compromises.  
- **Correlation analysis** – Identified discriminative features for intrusion detection.  


### Model Development
The ensemble model uses **Stacking Classifier** with:
- **Base Models**: Linear Regression, KNN, Gaussian Naive Bayes  
- **Meta‑Model**: Logistic Regression  
- **Cross‑Validation**: cv=3 to prevent overfitting  
- **Parallelization**: n_jobs=-1 for faster training  

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

estimators = [
    ('lr', LinearRegression()),
    ('knn', KNeighborsClassifier()),
    ('gnb', GaussianNB())
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)
```

### Results/Findings
- **Accuracy**: ~99.15%  
- **Precision/Recall/F1**: 0.99 for both normal and anomalous traffic  
- **Confusion Matrix**:  
  - True Negatives: 2340  
  - True Positives: 2656  
  - False Positives: 25  
  - False Negatives: 18  
- **ROC Curve**: AUC = 1.00 (perfect separability between normal and malicious traffic)  

### Recommendations
- Tune model complexity to balance fit and generalization.  
- Implement adversarial robustness testing to defend against evasion attacks.  
- Optimize for real‑time performance (e.g., parallelization, model quantization).  
- Enable continuous monitoring and automated retraining to adapt to evolving threats.  
- Improve interpretability with feature importance analysis for actionable insights.  

### Limitations
- Dataset may not fully represent modern traffic patterns.  
- Rare attack types (U2R, R2L) remain underrepresented.  
- Real‑world deployment performance depends on hardware and network scale.  
