# Codility Playground

A playground repository containing solutions to Codility challenges and machine learning examples. This repository demonstrates various algorithmic problem-solving techniques and ML implementations using Python.

## Problems and Solutions

### 1. Binary Gap (`binary_gap.py`)
**Problem**: Find the longest sequence of consecutive zeros surrounded by ones in the binary representation of a positive integer N.

**Solution**: Two different approaches are implemented:
- **Solution 1**: Iterates through the binary string character by character, tracking gaps between '1's using state flags
- **Solution 2**: Uses string searching with `find()` method to locate consecutive '1's and calculate gaps between them

**Key Concepts**: Binary representation, string manipulation, algorithmic optimization
**Time Complexity**: O(log N) - where N is the input number

**Example**: 
- N = 137 (binary: 10001001) → Binary gap = 3
- N = 529 (binary: 1000010001) → Binary gap = 4

---

### 2. Auto MPG Regression (`auto_mpg.py`)
**Problem**: Predict car fuel efficiency (miles per gallon) using various car attributes like cylinders, displacement, horsepower, weight, acceleration, model year, and origin.

**Solution**: Comprehensive machine learning pipeline comparing multiple regression algorithms:
- **Linear Regression** (with/without polynomial feature engineering)
- **Random Forest Regressor**
- **LightGBM Regressor**
- **XGBoost Regressor**
- **Support Vector Regression (SVR)**
- **Multi-layer Perceptron (MLP)**

**Features**:
- Data preprocessing with missing value handling
- 5-fold cross-validation for robust evaluation
- Feature engineering with polynomial features and scaling
- Comprehensive visualization with scatter plots and performance comparisons
- Evaluation metrics: Mean Absolute Error (MAE) and R² Score

**Key Concepts**: Regression analysis, ensemble methods, feature engineering, cross-validation, model comparison

---

### 3. Credit Card Fraud Detection (`credit_card_fraud.py`)
**Problem**: Detect fraudulent credit card transactions using unsupervised anomaly detection techniques on highly imbalanced data.

**Solution**: Anomaly detection system using:
- **Isolation Forest**: Tree-based anomaly detection algorithm
- **Local Outlier Factor (LOF)**: Density-based anomaly detection

**Features**:
- Comprehensive data inspection and quality analysis
- Training on legitimate transactions only (unsupervised approach)
- Optimal threshold finding to match realistic fraud rates (~0.17%)
- Multiple evaluation metrics: ROC AUC, confusion matrices, precision/recall
- Real-time fraud detection simulation
- Visualization of anomaly scores and decision boundaries

**Key Concepts**: Anomaly detection, class imbalance, unsupervised learning, fraud detection, threshold optimization

---

### 4. Handwritten Digits Clustering (`handwritten_digits.py`)
**Problem**: Group handwritten digit images (0-9) into clusters without using the true labels, then evaluate how well the clusters correspond to actual digits.

**Solution**: Comprehensive clustering analysis using:
- **K-Means Clustering** with multiple configurations
- **Agglomerative Clustering** with different linkage methods
- **DBSCAN** for density-based clustering

**Features**:
- PCA dimensionality reduction for visualization (64D → 2D)
- Multiple clustering algorithms comparison
- Comprehensive evaluation metrics:
  - External: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
  - Internal: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index
- Cluster-to-digit mapping analysis
- Parameter optimization (k-values, linkage methods, DBSCAN parameters)
- Side-by-side visualizations comparing clustering results

**Key Concepts**: Unsupervised learning, clustering algorithms, dimensionality reduction, cluster evaluation

---

### 5. SMS Spam Classification (`sms_spam.py`)
**Problem**: Classify SMS messages as spam or legitimate (ham) using text classification techniques.

**Solution**: Text classification pipeline using:
- **TF-IDF Vectorization** with n-grams (unigrams and bigrams)
- **Logistic Regression** classifier
- **Scikit-learn Pipeline** for streamlined processing

**Features**:
- Automatic dataset download and preprocessing
- Stratified train-test split (80/20)
- Text preprocessing with stop words removal and lowercasing
- Model evaluation with accuracy, F1-score, and classification report
- Feature importance analysis showing most influential n-grams
- Model persistence with joblib
- Real-time prediction examples

**Key Concepts**: Natural language processing, text classification, feature extraction, pipeline design

---

## Setup and Usage

### Prerequisites
- Python 3.7+
- Required packages: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `lightgbm`, `xgboost`, `joblib`, `requests`

### Installation
```bash
# Clone the repository
git clone https://github.com/el97179/codility_playground.git
cd codility_playground

# Install required packages
pip install scikit-learn pandas numpy matplotlib lightgbm xgboost joblib requests

# Or create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # If requirements.txt exists
```

### Running the Examples
```bash
# Run individual examples
python binary_gap.py
python auto_mpg.py
python credit_card_fraud.py  # Requires creditcard.csv dataset
python handwritten_digits.py
python sms_spam.py
```

## Dataset Requirements

- **Credit Card Fraud**: Requires `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **SMS Spam**: Automatically downloads the SMS Spam Collection dataset
- **Auto MPG & Handwritten Digits**: Use built-in scikit-learn datasets

## Key Learning Outcomes

- **Algorithmic Problem Solving**: Efficient string and binary manipulation techniques
- **Machine Learning Pipeline Design**: End-to-end ML workflows with proper evaluation
- **Anomaly Detection**: Techniques for handling imbalanced datasets and fraud detection
- **Clustering Analysis**: Unsupervised learning and cluster evaluation methods
- **Text Classification**: NLP preprocessing and feature engineering for text data
- **Model Comparison**: Systematic evaluation and comparison of multiple algorithms
