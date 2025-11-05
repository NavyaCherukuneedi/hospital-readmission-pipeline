
import pandas as pd
import numpy as np
from typing import Tuple

class DataTransformer:
    """Transform raw hospital readmission data."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize transformer with dataframe."""
        self.df = df.copy()
        self.original_shape = df.shape
    
    def transform_all(self) -> pd.DataFrame:
        """Run all transformations in sequence."""
        print("Starting data transformation...")
        
        # Apply all transformations
        self.df = self.convert_age_to_numeric()
        self.df = self.standardize_test_results()
        self.df = self.handle_missing_specialty()
        self.df = self.convert_yes_no_to_binary()
        self.df = self.encode_categorical_features()
        self.df = self.create_derived_features()
        
        print(f"Transformation complete!")
        print(f"Original shape: {self.original_shape}")
        print(f"Transformed shape: {self.df.shape}")
        
        return self.df
    
    def convert_age_to_numeric(self) -> pd.DataFrame:
        """Convert age from string ranges to numeric midpoint."""
        print("\n1. Converting age to numeric...")
        
        age_mapping = {
            '[40-50)': 45,
            '[50-60)': 55,
            '[60-70)': 65,
            '[70-80)': 75,
            '[80-90)': 85,
            '[90-100)': 95
        }
        
        self.df['age_numeric'] = self.df['age'].map(age_mapping)
        
        # Verify conversion
        null_ages = self.df['age_numeric'].isnull().sum()
        if null_ages > 0:
            print(f"  Warning: {null_ages} unmapped age values")
        else:
            print(f"  ✓ Successfully converted all {len(self.df)} age values")
        
        return self.df
    
    def standardize_test_results(self) -> pd.DataFrame:
        """Standardize glucose and A1C test results."""
        print("\n2. Standardizing test results...")
        
        # Glucose test: no=0, normal=1, high=2
        glucose_mapping = {'no': 0, 'normal': 1, 'high': 2}
        self.df['glucose_test_encoded'] = self.df['glucose_test'].map(glucose_mapping)
        
        # A1C test: no=0, normal=1, high=2
        a1c_mapping = {'no': 0, 'normal': 1, 'high': 2}
        self.df['A1Ctest_encoded'] = self.df['A1Ctest'].map(a1c_mapping)
        
        print(f"  ✓ Glucose test encoded: {self.df['glucose_test_encoded'].nunique()} unique values")
        print(f"  ✓ A1C test encoded: {self.df['A1Ctest_encoded'].nunique()} unique values")
        
        return self.df
    
    def handle_missing_specialty(self) -> pd.DataFrame:
        """Handle missing medical specialty values."""
        print("\n3. Handling missing medical specialty...")
        
        missing_count = (self.df['medical_specialty'] == 'Missing').sum()
        print(f"  Found {missing_count} missing specialty values ({missing_count/len(self.df)*100:.2f}%)")
        
        #  a binary flag for missing and replace with 'Unknown'
        self.df['specialty_is_missing'] = (self.df['medical_specialty'] == 'Missing').astype(int)
        
        # Replace 'Missing' with 'Unknown' for encoding
        self.df['medical_specialty'] = self.df['medical_specialty'].replace('Missing', 'Unknown')
        
        print(f"  ✓ Created specialty_is_missing flag")
        print(f"  ✓ Unique specialties: {self.df['medical_specialty'].nunique()}")
        
        return self.df
    
    def convert_yes_no_to_binary(self) -> pd.DataFrame:
        """Convert yes/no columns to binary (0/1)."""
        print("\n4. Converting yes/no columns to binary...")
        
        yes_no_columns = ['change', 'diabetes_med', 'readmitted']
        
        for col in yes_no_columns:
            self.df[f'{col}_binary'] = (self.df[col] == 'yes').astype(int)
            print(f"  ✓ {col}: {self.df[f'{col}_binary'].unique()}")
        
        return self.df
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        print("\n5. Encoding categorical features...")
        
        categorical_cols = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']
        
        for col in categorical_cols:
            # Get dummies and drop first to avoid multicollinearity
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
            self.df = pd.concat([self.df, dummies], axis=1)
            print(f"  ✓ {col}: Created {len(dummies.columns)} binary features")
        
        return self.df
    
    def create_derived_features(self) -> pd.DataFrame:
        """Create new derived features."""
        print("\n6. Creating derived features...")
        
        # Total previous visits
        self.df['total_previous_visits'] = (
            self.df['n_outpatient'] + 
            self.df['n_inpatient'] + 
            self.df['n_emergency']
        )
        
        # Has previous inpatient history
        self.df['has_inpatient_history'] = (self.df['n_inpatient'] > 0).astype(int)
        
        # High medication count (above median)
        median_meds = self.df['n_medications'].median()
        self.df['high_medication_count'] = (self.df['n_medications'] > median_meds).astype(int)
        
        # Age group (categorical)
        self.df['age_group'] = pd.cut(
            self.df['age_numeric'], 
            bins=[0, 50, 60, 70, 80, 90, 100],
            labels=['40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
        )
        
        print(f"  ✓ total_previous_visits created")
        print(f"  ✓ has_inpatient_history created")
        print(f"  ✓ high_medication_count created")
        print(f"  ✓ age_group created")
        
        return self.df
    
    def get_summary(self) -> dict:
        """Get transformation summary."""
        return {
            'original_shape': self.original_shape,
            'transformed_shape': self.df.shape,
            'columns_added': self.df.shape[1] - self.original_shape[1],
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }