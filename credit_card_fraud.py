import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

def load_creditcard_data(filepath='creditcard.csv'):
    """Load credit card fraud dataset from CSV file"""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {filepath}")
        
        # Data inspection
        print("\n" + "="*60)
        print("DATASET INSPECTION")
        print("="*60)
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column information
        print(f"\nColumns ({len(df.columns)}):")
        print(df.columns.tolist())
        
        # Data types
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print(f"\nNo missing values found ✓")
        
        # Target variable analysis
        if 'Class' in df.columns:
            print(f"\nTarget variable (Class) distribution:")
            class_counts = df['Class'].value_counts().sort_index()
            print(class_counts)
            fraud_rate = df['Class'].mean()
            print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
            print(f"Class imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1 (Normal:Fraud)")
        
        # Feature statistics
        print(f"\nFeature statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Numeric columns: {len(numeric_cols)}")
        
        # Show summary statistics for key columns
        if 'Amount' in df.columns:
            print(f"\nAmount column statistics:")
            print(f"  Min: ${df['Amount'].min():.2f}")
            print(f"  Max: ${df['Amount'].max():.2f}")
            print(f"  Mean: ${df['Amount'].mean():.2f}")
            print(f"  Median: ${df['Amount'].median():.2f}")
            print(f"  Std: ${df['Amount'].std():.2f}")
        
        if 'Time' in df.columns:
            print(f"\nTime column statistics:")
            print(f"  Min: {df['Time'].min():.0f} seconds")
            print(f"  Max: {df['Time'].max():.0f} seconds")
            print(f"  Duration: {(df['Time'].max() - df['Time'].min())/3600:.1f} hours")
        
        # V columns (PCA transformed features)
        v_columns = [col for col in df.columns if col.startswith('V')]
        if v_columns:
            print(f"\nPCA-transformed features (V1-V28): {len(v_columns)}")
            v_stats = df[v_columns].describe()
            print(f"  Mean range: [{v_stats.loc['mean'].min():.3f}, {v_stats.loc['mean'].max():.3f}]")
            print(f"  Std range: [{v_stats.loc['std'].min():.3f}, {v_stats.loc['std'].max():.3f}]")
        
        # Duplicate check
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️  Found {duplicates} duplicate rows")
        else:
            print(f"\nNo duplicate rows found ✓")
        
        # Data quality summary
        print(f"\n" + "="*40)
        print("DATA QUALITY SUMMARY")
        print("="*40)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        print(f"Data completeness: {completeness:.2f}%")
        print(f"Ready for modeling: {'✓' if completeness > 99 and duplicates < df.shape[0]*0.01 else '⚠️'}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the file exists in the current directory.")
        print("You can download the dataset from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        raise
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        raise

class CreditCardFraudDetector:
    def __init__(self, contamination=0.001, random_state=0):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True
        )
        self.threshold = None
        
    def load_and_preprocess_data(self, df):
        """Load and preprocess the data"""
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud rate: {df['Class'].mean():.4f}")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_models(self, X_train, y_train):
        """Train the anomaly detection models"""
        # Get only legitimate transactions for training
        legitimate_mask = y_train == 0
        X_legitimate = X_train[legitimate_mask]
        
        print(f"Training on {len(X_legitimate)} legitimate transactions")
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_legitimate)
        
        # Train Local Outlier Factor
        self.lof.fit(X_legitimate)
        
        print("Models trained successfully!")
    
    def find_optimal_threshold(self, X_val, y_val, target_fraud_rate=0.0017):
        """Find threshold that gives approximately the target fraud rate"""
        # Get anomaly scores
        if_scores = self.isolation_forest.decision_function(X_val)
        
        # Try different thresholds
        thresholds = np.percentile(if_scores, np.linspace(0.1, 5, 100))
        best_threshold = None
        best_diff = float('inf')
        
        for threshold in thresholds:
            predicted_fraud_rate = np.mean(if_scores < threshold)
            diff = abs(predicted_fraud_rate - target_fraud_rate)
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        
        self.threshold = best_threshold
        predicted_rate = np.mean(if_scores < best_threshold)
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"Predicted fraud rate: {predicted_rate:.4f}")
        
        return best_threshold
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models and return metrics"""
        results = {}
        
        # Isolation Forest evaluation
        if_scores = self.isolation_forest.decision_function(X_test)
        if_predictions = (if_scores < self.threshold).astype(int)
        
        if_auc = roc_auc_score(y_test, -if_scores)  # Negative because lower scores = more anomalous
        if_cm = confusion_matrix(y_test, if_predictions)
        
        results['isolation_forest'] = {
            'auc': if_auc,
            'confusion_matrix': if_cm,
            'predictions': if_predictions,
            'scores': if_scores
        }
        
        # Local Outlier Factor evaluation
        lof_scores = self.lof.decision_function(X_test)
        lof_threshold = np.percentile(lof_scores, (1 - 0.0017) * 100)
        lof_predictions = (lof_scores < lof_threshold).astype(int)
        
        lof_auc = roc_auc_score(y_test, -lof_scores)
        lof_cm = confusion_matrix(y_test, lof_predictions)
        
        results['local_outlier_factor'] = {
            'auc': lof_auc,
            'confusion_matrix': lof_cm,
            'predictions': lof_predictions,
            'scores': lof_scores
        }
        
        return results
    
    def print_evaluation_results(self, results, y_test):
        """Print detailed evaluation results"""
        for model_name, metrics in results.items():
            print(f"\n{'='*50}")
            print(f"{model_name.upper()} RESULTS")
            print(f"{'='*50}")
            
            print(f"ROC AUC Score: {metrics['auc']:.4f}")
            
            print(f"\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"True Negatives: {cm[0,0]}")
            print(f"False Positives: {cm[0,1]}")
            print(f"False Negatives: {cm[1,0]}")
            print(f"True Positives: {cm[1,1]}")
            
            # Calculate additional metrics
            precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
            recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nPrecision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
    
    def plot_results(self, results, y_test):
        """Plot confusion matrices and score distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot confusion matrices
        for i, (model_name, metrics) in enumerate(results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[0, i], cbar=False)
            axes[0, i].set_title(f'{model_name.replace("_", " ").title()} - Confusion Matrix')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('Actual')
        
        # Plot score distributions
        for i, (model_name, metrics) in enumerate(results.items()):
            scores = metrics['scores']
            fraud_scores = scores[y_test == 1]
            normal_scores = scores[y_test == 0]
            
            axes[1, i].hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
            axes[1, i].hist(fraud_scores, bins=50, alpha=0.7, label='Fraud', density=True)
            axes[1, i].set_title(f'{model_name.replace("_", " ").title()} - Score Distribution')
            axes[1, i].set_xlabel('Anomaly Score')
            axes[1, i].set_ylabel('Density')
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("Credit Card Fraud Detection using Isolation Forest")
    print("="*60)
    
    # Load credit card data from CSV
    print("Loading credit card fraud dataset...")
    df = load_creditcard_data('creditcard.csv')
    
    # Initialize detector
    detector = CreditCardFraudDetector(contamination=0.001, random_state=0)
    
    # Load and preprocess data
    X, y = detector.load_and_preprocess_data(df)
    
    # Split data into train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    detector.train_models(X_train, y_train)
    
    # Find optimal threshold
    detector.find_optimal_threshold(X_val, y_val, target_fraud_rate=0.0017)
    
    # Evaluate models
    results = detector.evaluate_models(X_test, y_test)
    
    # Print results
    detector.print_evaluation_results(results, y_test)
    
    # Plot results
    detector.plot_results(results, y_test)
    
    # Demonstrate real-time fraud detection
    print(f"\n{'='*60}")
    print("REAL-TIME FRAUD DETECTION SIMULATION")
    print(f"{'='*60}")
    
    # Use some actual transactions from the test set for simulation
    sample_indices = np.random.choice(len(X_test), size=10, replace=False)
    sample_transactions = X_test[sample_indices]
    sample_labels = y_test.iloc[sample_indices] if hasattr(y_test, 'iloc') else y_test[sample_indices]
    
    if_scores = detector.isolation_forest.decision_function(sample_transactions)
    if_predictions = (if_scores < detector.threshold).astype(int)
    
    print("New Transaction Analysis:")
    for i, (score, pred, actual) in enumerate(zip(if_scores, if_predictions, sample_labels)):
        status = "🚨 FRAUD ALERT" if pred == 1 else "✅ Normal"
        actual_status = "FRAUD" if actual == 1 else "Normal"
        correct = "✓" if pred == actual else "✗"
        print(f"Transaction {i+1}: Score = {score:.4f} | {status} | Actual: {actual_status} {correct}")

if __name__ == "__main__":
    main()