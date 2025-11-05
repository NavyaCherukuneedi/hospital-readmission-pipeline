"""Generate predictions using trained model."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseConnection
from modeling.feature_engineering import FeatureEngineer


class PredictionGenerator:
    """Generate predictions using trained model."""
    
    def __init__(self, model_path: str):
        """Load trained model."""
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and probabilities."""
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        return y_pred, y_pred_proba
    
    def create_predictions_dataframe(
        self, 
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> pd.DataFrame:
        """Create comprehensive predictions dataframe."""
        
        predictions_df = pd.DataFrame({
            'actual_readmitted': y_true.values,
            'predicted_readmitted': y_pred,
            'prediction_probability': y_pred_proba,
            'prediction_confidence': np.maximum(y_pred_proba, 1 - y_pred_proba),
            'correct_prediction': (y_pred == y_true.values).astype(int)
        })
        
        # Add risk category
        predictions_df['risk_category'] = pd.cut(
            y_pred_proba,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return predictions_df


def run_prediction_pipeline():
    """Load model, generate predictions, and save results."""
    
    print("="*80)
    print("PREDICTION GENERATION PIPELINE")
    print("="*80)
    
    # 1. Load transformed data
    print("\n1. Loading transformed data...")
    db = DatabaseConnection()
    if not db.test_connection():
        raise ConnectionError("Failed to connect to database")
    
    df = pd.read_sql("SELECT * FROM hospital_readmissions_transformed", db.engine)
    print(f"   ✓ Loaded {len(df)} rows")
    
    # 2. Feature engineering
    print("\n2. Feature engineering...")
    fe = FeatureEngineer(df)
    X, y = fe.prepare_features()
    print(f"   ✓ Prepared {X.shape[1]} features")
    
    # 3. Train/test split
    print("\n3. Creating train/test split...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Scale features
    print("\n4. Scaling features...")
    X_train_scaled, X_test_scaled = fe.scale_numeric_features(X_train, X_test)
    print(f"   ✓ Features scaled")
    
    # 5. Load model
    print("\n5. Loading trained model...")
    model_path = Path(__file__).parent.parent.parent / 'output' / 'best_model_random_forest.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    predictor = PredictionGenerator(str(model_path))
    
    # 6. Generate predictions
    print("\n6. Generating predictions on test set...")
    y_pred, y_pred_proba = predictor.predict(X_test_scaled)
    print(f"   ✓ Generated {len(y_pred)} predictions")
    
    # 7. Create predictions dataframe
    print("\n7. Creating predictions dataframe...")
    predictions_df = predictor.create_predictions_dataframe(
        X_test_scaled, y_test, y_pred, y_pred_proba
    )
    print(f"   ✓ Created predictions dataframe with shape {predictions_df.shape}")
    
    # 8. Save predictions to database
    print("\n8. Saving predictions to database...")
    predictions_df.to_sql(
        'model_predictions',
        db.engine,
        if_exists='replace',
        index=False
    )
    print(f"   ✓ Saved to table: model_predictions")
    
    # 9. Print prediction statistics
    print("\n9. Prediction Statistics:")
    print(f"   Predicted Readmitted: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.2f}%)")
    print(f"   Predicted Not Readmitted: {(1-y_pred).sum()} ({(1-y_pred).sum()/len(y_pred)*100:.2f}%)")
    print(f"   Average Probability: {y_pred_proba.mean():.4f}")
    print(f"   Correct Predictions: {predictions_df['correct_prediction'].sum()} ({predictions_df['correct_prediction'].mean()*100:.2f}%)")
    
    # 10. Risk category distribution
    print("\n10. Risk Category Distribution:")
    print(predictions_df['risk_category'].value_counts().sort_index())
    
    db.close()
    
    print("\n" + "="*80)
    print("PREDICTION PIPELINE COMPLETE!")
    print("="*80)
    
    return predictions_df


if __name__ == "__main__":
    predictions_df = run_prediction_pipeline()