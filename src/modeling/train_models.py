"""Train and evaluate machine learning models."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import joblib
from pathlib import Path
from typing import Dict, Tuple
import json

class ModelTrainer:
    """Train and evaluate ML models for readmission prediction."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42):
        """Initialize model trainer."""
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create stratified train/test split."""
        print("\n" + "="*80)
        print("CREATING TRAIN/TEST SPLIT")
        print("="*80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"\nTrain Set:")
        print(f"  - Samples: {len(X_train)}")
        print(f"  - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
        print(f"  - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
        
        print(f"\nTest Set:")
        print(f"  - Samples: {len(X_test)}")
        print(f"  - Class 0: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
        print(f"  - Class 1: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train logistic regression model."""
        print("\n" + "-"*80)
        print("TRAINING: Logistic Regression")
        print("-"*80)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        
        print("✓ Model trained successfully")
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train random forest model."""
        print("\n" + "-"*80)
        print("TRAINING: Random Forest")
        print("-"*80)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        
        print("✓ Model trained successfully")
    
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train gradient boosting model."""
        print("\n" + "-"*80)
        print("TRAINING: Gradient Boosting")
        print("-"*80)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            subsample=0.8
        )
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        
        print("✓ Model trained successfully")
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all models."""
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate all trained models."""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': model
            }
            
            # Print metrics
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC AUC:   {roc_auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n  Confusion Matrix:")
            print(f"    True Negative:  {cm[0][0]}")
            print(f"    False Positive: {cm[0][1]}")
            print(f"    False Negative: {cm[1][0]}")
            print(f"    True Positive:  {cm[1][1]}")
    
    def select_best_model(self, metric: str = 'f1') -> str:
        """Select best model based on metric."""
        print("\n" + "="*80)
        print("MODEL SELECTION")
        print("="*80)
        
        best_score = -1
        best_name = None
        
        print(f"\nSelecting best model based on {metric.upper()}:\n")
        
        for model_name, results in self.results.items():
            score = results[metric]
            print(f"  {model_name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n✓ Best Model: {best_name} ({metric}={best_score:.4f})")
        
        return best_name
    
    def save_best_model(self, output_dir: str = 'output'):
        """Save best model to disk."""
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model() first.")
        
        model_path = output_path / f"best_model_{self.best_model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(self.best_model, model_path)
        
        print(f"✓ Model saved to: {model_path}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name],
            'feature_count': len(self.X.columns)
        }
        
        metadata_path = output_path / f"model_metadata_{self.best_model_name.lower().replace(' ', '_')}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all model results."""
        summary_data = []
        
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1'],
                'ROC AUC': results['roc_auc']
            })
        
        return pd.DataFrame(summary_data)