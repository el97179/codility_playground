import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import requests
import zipfile
from io import StringIO

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)
from io import BytesIO
with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
    with zip_ref.open(zip_ref.namelist()[0]) as f:
        content = f.read().decode('utf-8', errors='ignore')
        df = pd.read_csv(StringIO(content), sep='\t', names=['label', 'text'])

# Data inspection
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Average message length: {df['text'].str.len().mean():.1f} characters")

# Data preprocessing
df['text'] = df['text'].str.lower()
df = df.dropna().drop_duplicates()
print(f"After preprocessing: {df.shape[0]} samples")

# Prepare data
X = df['text']
y = df['label'].map({'spam': 1, 'ham': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Test examples
examples = [
    "Win a free iPhone now!",
    "Hey, how are you doing?"
]
for text in examples:
    prediction = pipeline.predict([text])[0]
    confidence = pipeline.predict_proba([text])[0].max()
    pred_label = "SPAM" if prediction else "HAM"
    print(f"'{text}' -> {pred_label} (confidence: {confidence:.3f})")