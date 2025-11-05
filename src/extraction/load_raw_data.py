"""Extract and load raw hospital readmission data."""

import os
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseConnection

def load_raw_data():
    """Load raw CSV data into PostgreSQL."""
    
    # Find CSV files in data directory
    data_dir = Path(__file__).parent.parent.parent / 'data'
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    # Connect to database
    db = DatabaseConnection()
    
    if not db.test_connection():
        raise ConnectionError("Failed to connect to database")
    
    print("✓ Successfully connected to database")
    
    # Load each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        
        # Read CSV
        df = pd.read_csv(csv_file)
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Generate table name from filename
        table_name = csv_file.stem.lower()
        print(f"  Table name: {table_name}")
        
        # Insert into database
        try:
            df.to_sql(table_name, db.engine, if_exists='replace', index=False)
            print(f"  ✓ Successfully loaded {len(df)} rows into table '{table_name}'")
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            raise
    
    db.close()
    print("\n✓ Data loading complete!")

if __name__ == "__main__":
    load_raw_data()