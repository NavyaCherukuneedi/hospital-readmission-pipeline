"""Data validation rules and checks."""

import pandas as pd
import numpy as np

class DataValidator:
    """Validate hospital readmission data quality."""
    
    def __init__(self, df):
        """Initialize validator with dataframe."""
        self.df = df
        self.errors = []
        self.warnings = []
    
    def validate_all(self):
        """Run all validation checks."""
        self.check_required_columns()
        self.check_data_types()
        self.check_numeric_ranges()
        self.check_categorical_values()
        self.check_target_variable()
        return self.get_report()
    
    def check_required_columns(self):
        """Verify all required columns exist."""
        required = ['readmitted', 'time_in_hospital', 'age', 'medical_specialty']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
    
    def check_data_types(self):
        """Check if data types are appropriate."""
        # This will be filled based on your findings
        pass
    
    def check_numeric_ranges(self):
        """Verify numeric columns are within expected ranges."""
        numeric_checks = {
            'time_in_hospital': (0, 365),
            'n_lab_procedures': (0, 1000),
            'n_procedures': (0, 1000),
            'n_medications': (0, 1000),
        }
        
        for col, (min_val, max_val) in numeric_checks.items():
            if col in self.df.columns:
                out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
                if len(out_of_range) > 0:
                    self.warnings.append(
                        f"{col}: {len(out_of_range)} values out of range [{min_val}, {max_val}]"
                    )
    
    def check_categorical_values(self):
        """Verify categorical columns have expected values."""
        expected_values = {
            'readmitted': ['yes', 'no'],
            'glucose_test': ['yes', 'no'],
            'A1Ctest': ['yes', 'no'],
            'change': ['yes', 'no'],
            'diabetes_med': ['yes', 'no'],
        }
        
        for col, expected in expected_values.items():
            if col in self.df.columns:
                unexpected = set(self.df[col].unique()) - set(expected)
                if unexpected:
                    self.warnings.append(
                        f"{col}: unexpected values {unexpected}"
                    )
    
    def check_target_variable(self):
        """Verify target variable has no nulls."""
        if 'readmitted' in self.df.columns:
            nulls = self.df['readmitted'].isnull().sum()
            if nulls > 0:
                self.errors.append(f"Target variable has {nulls} null values")
    
    def get_report(self):
        """Return validation report."""
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings
        }