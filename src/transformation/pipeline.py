"""Data transformation pipeline."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseConnection
from transform_data import DataTransformer

def run_transformation_pipeline():
    """Load raw data, transform, and save to database."""
    
    print("="*80)
    print("DATA TRANSFORMATION PIPELINE")
    print("="*80)
    
    # Connect to database
    print("\n1. Connecting to database...")
    db = DatabaseConnection()
    if not db.test_connection():
        raise ConnectionError("Failed to connect to database")
    print("   ✓ Database connection successful")
    
    # Load raw data
    print("\n2. Loading raw data...")
    df = pd.read_sql("SELECT * FROM hospital_readmissions", db.engine)
    print(f"   ✓ Loaded {len(df)} rows with {df.shape[1]} columns")
    
    # Transform data
    print("\n3. Transforming data...")
    transformer = DataTransformer(df)
    transformed_df = transformer.transform_all()
    
    
    summary = transformer.get_summary()
    print("\n4. Transformation Summary:")
    print(f"   Original shape: {summary['original_shape']}")
    print(f"   Transformed shape: {summary['transformed_shape']}")
    print(f"   Columns added: {summary['columns_added']}")
    print(f"   Memory usage: {summary['memory_usage_mb']:.2f} MB")
    
    # Save transformed data
    print("\n5. Saving transformed data to database...")
    try:
        transformed_df.to_sql(
            'hospital_readmissions_transformed',
            db.engine,
            if_exists='replace',
            index=False,
            chunksize=1000
        )
        print(f"   ✓ Successfully saved to 'hospital_readmissions_transformed' table")
    except Exception as e:
        print(f"   ✗ Error saving data: {e}")
        raise
    
    # Verify
    print("\n6. Verifying transformed data...")
    result = db.query(
        "SELECT COUNT(*) FROM hospital_readmissions_transformed"
    )
    row_count = result[0][0]
    print(f"   ✓ Verified: {row_count} rows in transformed table")
    
    # Display sample
    print("\n7. Sample of transformed data:")
    sample = pd.read_sql(
        "SELECT * FROM hospital_readmissions_transformed LIMIT 5",
        db.engine
    )
    print(sample.to_string())
    
    db.close()
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    run_transformation_pipeline()