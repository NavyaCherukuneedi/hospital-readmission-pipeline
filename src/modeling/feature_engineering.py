"""Feature engineering and preparation for modeling."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple

class FeatureEngineer:
    """Prepare features for machine learning."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize feature engineer with dataframe."""
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.numeric_features = None
        self.categorical_features = None
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling."""
        print("Preparing features for modeling...")
        
        # Separate target from features
        target = self.df['readmitted_binary'].copy()
        X = self.df.drop(columns=['readmitted_binary', 'readmitted', 'age']).copy()
        
        print(f"\nFeature Matrix Shape: {X.shape}")
        print(f"Target Shape: {target.shape}")
        print(f"Target Distribution:")
        print(f"  - Class 0 (Not Readmitted): {(target == 0).sum()} ({(target == 0).sum()/len(target)*100:.2f}%)")
        print(f"  - Class 1 (Readmitted): {(target == 1).sum()} ({(target == 1).sum()/len(target)*100:.2f}%)")
        
        # Identify numeric and categorical columns before encoding
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"\nNumeric Features: {len(self.numeric_features)}")
        print(f"Categorical Features: {len(self.categorical_features)}")
        
        # Encode remaining categorical features (one-hot encoding)
        if len(self.categorical_features) > 0:
            print(f"\nOne-hot encoding {len(self.categorical_features)} categorical features...")
            X = pd.get_dummies(X, columns=self.categorical_features, drop_first=True)
            print(f"  Encoded. New shape: {X.shape}")
        
        # Remove low-variance features
        X = self._remove_low_variance_features(X)
        
        return X, target
    
    def _remove_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with very low variance."""
        print("\nRemoving low-variance features...")
        
        initial_cols = len(X.columns)
        
        # Calculate variance for all numeric features (after one-hot encoding, all are numeric)
        variances = X.var()
        
        # Remove features with variance below threshold
        threshold = variances.quantile(0.05)  # Remove bottom 5%
        high_variance_cols = variances[variances > threshold].index.tolist()
        
        # Keep only high variance features
        X = X[high_variance_cols]
        
        removed = initial_cols - len(X.columns)
        print(f"  Removed {removed} low-variance features")
        print(f"  Kept {len(X.columns)} features")
        
        return X
    
    def scale_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numeric features using training data statistics."""
        print("\nScaling numeric features...")
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            
            print(f"  Scaled {len(numeric_cols)} numeric features")
        
        return X_train_scaled, X_test_scaled
    
    def get_summary(self, X: pd.DataFrame) -> dict:
        """Get summary of features."""
        return {
            'total_features': X.shape[1],
            'numeric_features': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(X.select_dtypes(include=['object', 'category']).columns),
            'feature_names': X.columns.tolist()
        }
    
    def get_feature_summary(self, X: pd.DataFrame) -> dict:
        """Get summary of features."""
        return {
            'total_features': X.shape[1],
            'numeric_features': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(X.select_dtypes(include=['object', 'category']).columns),
            'feature_names': X.columns.tolist()
        }