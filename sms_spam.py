import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import zipfile
import requests
import os
from io import StringIO

def load_data(url):
    """Load SMS spam dataset from URL or local file."""
    try:
        if url.startswith('http'):
            print("Downloading dataset...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save the zip file temporarily
            zip_path = '/tmp/smsspamcollection.zip'
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract and read the SMS data file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get the first file from the ZIP (should be 'SMSSpamCollection')
                file_list = zip_ref.namelist()
                print(f"Files in ZIP: {file_list}")
                
                sms_file = file_list[0]  # Use the first file
                print(f"Using first file: {sms_file}")
                
                with zip_ref.open(sms_file) as f:
                    # Read the file content with proper encoding
                    content = f.read().decode('utf-8', errors='ignore')
                    df = pd.read_csv(StringIO(content), sep='\t', names=['label', 'text'], header=None, encoding='utf-8')
            
            # Clean up temporary file
            os.remove(zip_path)
        else:
            # Load from local file
            df = pd.read_csv(url, sep='\t', names=['label', 'text'], header=None)
        
        print(f"Dataset loaded successfully with {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have internet connection or provide a local file path.")
        return None

def preprocess_data(df):
    """Preprocess the data and create train/test split."""
    # Convert labels to binary (spam=1, ham=0)
    df['label_binary'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Stratified train/test split (80/20)
    X = df['text']
    y = df['label_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train):
    """Build and train the SMS spam classification pipeline."""
    # Create pipeline with TfidfVectorizer and LogisticRegression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', lowercase=True)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed!")
    
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the trained model."""
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    return accuracy, f1

def show_influential_features(pipeline, n_features=10):
    """Show most influential n-grams for spam/ham classification."""
    # Get feature names and coefficients
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    
    # Get indices of most positive and negative coefficients
    top_spam_indices = np.argsort(coefficients)[-n_features:][::-1]
    top_ham_indices = np.argsort(coefficients)[:n_features]
    
    print(f"\nTop {n_features} features indicating SPAM:")
    for idx in top_spam_indices:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    print(f"\nTop {n_features} features indicating HAM:")
    for idx in top_ham_indices:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

def save_model(pipeline, filepath='sms_spam_classifier.joblib'):
    """Save the trained pipeline for reuse."""
    joblib.dump(pipeline, filepath)
    print(f"Model saved to {filepath}")

def main():
    """Main function to run the SMS spam classifier."""

    dataset_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    print("SMS Spam Classifier")
    print("=" * 50)
    
    # Load data
    df = load_data(dataset_path)
    if df is None:
        print("Please download the SMS Spam Collection dataset and update the dataset_path variable.")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Build and train model
    pipeline = build_and_train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, f1 = evaluate_model(pipeline, X_test, y_test)
    
    # Show influential features (stretch goal)
    show_influential_features(pipeline)
    
    # Save model (stretch goal)
    save_model(pipeline)
    
    # Example predictions
    print(f"\nExample Predictions:")
    example_texts = [
        "Congratulations! You've won a free iPhone! Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT! Your account will be suspended unless you verify immediately",
        "Thanks for the birthday wishes yesterday!"
    ]
    
    for text in example_texts:
        prediction = pipeline.predict([text])[0]
        probability = pipeline.predict_proba([text])[0][prediction]
        label = "SPAM" if prediction == 1 else "HAM"
        print(f"  Text: '{text[:50]}...'")
        print(f"  Prediction: {label} (confidence: {probability:.3f})")
        print()

if __name__ == "__main__":
    main()