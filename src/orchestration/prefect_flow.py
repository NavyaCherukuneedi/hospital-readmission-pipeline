"""Prefect workflow orchestration for the data pipeline."""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from prefect import flow, task, get_run_logger

@task(name="Load Raw Data")
def load_raw_data():
    """Load raw data from database."""
    logger = get_run_logger()
    logger.info("Loading raw hospital readmission data...")
    
    from database import DatabaseConnection
    db = DatabaseConnection()
    
    df = pd.read_sql("SELECT * FROM hospital_readmissions", db.engine)
    db.close()
    
    logger.info(f"✓ Loaded {len(df)} rows from raw data table")
    return df

@task(name="Transform Data")
def transform_data(df):
    """Transform raw data."""
    logger = get_run_logger()
    logger.info("Transforming data...")
    
    from transformation.transform_data import DataTransformer
    
    transformer = DataTransformer(df)
    transformed_df = transformer.transform_all()
    
    logger.info(f"✓ Transformed data: {transformed_df.shape}")
    return transformed_df

@task(name="Engineer Features")
def engineer_features(df):
    """Engineer features for modeling."""
    logger = get_run_logger()
    logger.info("Engineering features...")
    
    from modeling.feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer(df)
    X, y = fe.prepare_features()
    
    logger.info(f"✓ Engineered {X.shape[1]} features")
    return X, y

@task(name="Train Models")
def train_and_evaluate_models(X, y):
    """Train and evaluate machine learning models."""
    logger = get_run_logger()
    logger.info("Training models...")
    
    from sklearn.model_selection import train_test_split
    from modeling.train_models import ModelTrainer
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    from modeling.feature_engineering import FeatureEngineer
    fe = FeatureEngineer(X)
    X_train_scaled, X_test_scaled = fe.scale_numeric_features(X_train, X_test)
    
    trainer = ModelTrainer(X, y)
    trainer.train_all_models(X_train_scaled, y_train)
    trainer.evaluate_models(X_test_scaled, y_test)
    best_model_name = trainer.select_best_model(metric='f1')
    trainer.save_best_model(output_dir='output')
    
    logger.info(f"✓ Best model: {best_model_name}")
    return trainer.get_results_summary()

@task(name="Generate Predictions")
def generate_predictions(X, y):
    """Generate predictions using trained model."""
    logger = get_run_logger()
    logger.info("Generating predictions...")
    
    import joblib
    from sklearn.model_selection import train_test_split
    from modeling.feature_engineering import FeatureEngineer
    from modeling.make_predictions import PredictionGenerator
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    fe = FeatureEngineer(X)
    X_train_scaled, X_test_scaled = fe.scale_numeric_features(X_train, X_test)
    
    model_path = Path(__file__).parent.parent.parent / 'output' / 'best_model_random_forest.pkl'
    predictor = PredictionGenerator(str(model_path))
    
    y_pred, y_pred_proba = predictor.predict(X_test_scaled)
    predictions_df = predictor.create_predictions_dataframe(
        X_test_scaled, y_test, y_pred, y_pred_proba
    )
    
    logger.info(f"✓ Generated {len(predictions_df)} predictions")
    return predictions_df

@task(name="Save Results")
def save_results_to_database(transformed_df, predictions_df):
    """Save transformed data and predictions to database."""
    logger = get_run_logger()
    logger.info("Saving results to database...")
    
    from database import DatabaseConnection
    db = DatabaseConnection()
    
    # Save transformed data
    transformed_df.to_sql(
        'hospital_readmissions_transformed',
        db.engine,
        if_exists='replace',
        index=False,
        chunksize=1000
    )
    
    # Save predictions
    predictions_df.to_sql(
        'model_predictions',
        db.engine,
        if_exists='replace',
        index=False
    )
    
    db.close()
    logger.info("✓ Results saved to database")

@task(name="Generate Report")
def generate_execution_report(model_results, predictions_df):
    """Generate execution report."""
    logger = get_run_logger()
    logger.info("Generating execution report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'SUCCESS',
        'model_performance': model_results.to_dict(),
        'predictions_count': len(predictions_df),
        'high_risk_patients': len(predictions_df[predictions_df['risk_category'] == 'Very High']),
        'average_probability': float(predictions_df['prediction_probability'].mean())
    }
    
    logger.info("✓ Execution report generated")
    return report

@flow(name="Hospital Readmission Pipeline", description="End-to-end ML pipeline for hospital readmission prediction")
def hospital_readmission_pipeline():
    """Main orchestration flow."""
    logger = get_run_logger()
    
    logger.info("="*80)
    logger.info("HOSPITAL READMISSION PREDICTION PIPELINE - PREFECT FLOW")
    logger.info("="*80)
    
    try:
        # Execute pipeline steps
        raw_data = load_raw_data()
        transformed_data = transform_data(raw_data)
        X, y = engineer_features(transformed_data)
        model_results = train_and_evaluate_models(X, y)
        predictions_df = generate_predictions(X, y)
        save_results_to_database(transformed_data, predictions_df)
        report = generate_execution_report(model_results, predictions_df)
        
        logger.info("="*80)
        logger.info("PIPELINE EXECUTION SUCCESSFUL!")
        logger.info("="*80)
        logger.info(f"Timestamp: {report['timestamp']}")
        logger.info(f"Predictions Generated: {report['predictions_count']}")
        logger.info(f"High-Risk Patients: {report['high_risk_patients']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    hospital_readmission_pipeline()