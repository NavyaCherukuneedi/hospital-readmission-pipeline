"""Machine learning pipeline."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseConnection
from feature_engineering import FeatureEngineer
from train_models import ModelTrainer

def run_modeling_pipeline():
    """Load data, engineer features, train models, and evaluate."""
    
    print("="*80)
    print("MACHINE LEARNING PIPELINE")
    print("="*80)
    
    # 1. Load transformed data
    print("\n1. Loading transformed data...")
    db = DatabaseConnection()
    if not db.test_connection():
        raise ConnectionError("Failed to connect to database")
    
    df = pd.read_sql("SELECT * FROM hospital_readmissions_transformed", db.engine)
    print(f"   ✓ Loaded {len(df)} rows with {df.shape[1]} columns")
    
    # 2. Feature engineering
    print("\n2. Feature engineering...")
    fe = FeatureEngineer(df)
    X, y = fe.prepare_features()
    feature_summary = fe.get_summary(X)
    print(f"   ✓ Prepared {feature_summary['total_features']} features")
    
    # 3. Train/test split
    print("\n3. Creating train/test split...")
    trainer = ModelTrainer(X, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = trainer.train_test_split()
    
    # 4. Feature scaling
    print("\n4. Scaling numeric features...")
    X_train_scaled, X_test_scaled = fe.scale_numeric_features(X_train, X_test)
    print("   ✓ Features scaled")
    
    # 5. Train models
    print("\n5. Training models...")
    trainer.train_all_models(X_train_scaled, y_train)
    
    # 6. Evaluate models
    print("\n6. Evaluating models...")
    trainer.evaluate_models(X_test_scaled, y_test)
    
    # 7. Select best model
    print("\n7. Selecting best model...")
    best_model_name = trainer.select_best_model(metric='f1')
    
    # 8. Save best model
    print("\n8. Saving best model...")
    trainer.save_best_model(output_dir='output')
    
    # 9. Summary
    print("\n9. Results Summary:")
    print(trainer.get_results_summary().to_string(index=False))
    
    db.close()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    run_modeling_pipeline()