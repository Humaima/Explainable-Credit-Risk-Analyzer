# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Optional, List

class CreditDataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.outlier_thresholds = {}  # Store outlier thresholds for consistency
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""
        df = df.copy()
        
        # Payment behavior features
        df['total_late_payments'] = df['late_30_59_days'] + df['late_60_89_days'] + df['late_90_days']
        df['has_ever_late'] = (df['total_late_payments'] > 0).astype(int)
        df['severe_late_ratio'] = df['late_90_days'] / (df['total_late_payments'] + 1)
        
        # Credit utilization features
        df['credit_lines_per_age'] = df['open_credit_lines'] / (df['age'] + 1)
        df['estate_ratio'] = df['real_estate_loans'] / (df['open_credit_lines'] + 1)
        
        # Income-related features
        df['income_per_dependent'] = df['monthly_income'] / (df['dependents'] + 1)
        
        # For transform, we need to use the median from training for high_income_flag
        if 'monthly_income_median' in self.__dict__:
            df['high_income_flag'] = (df['monthly_income'] > self.monthly_income_median).astype(int)
        else:
            df['high_income_flag'] = (df['monthly_income'] > df['monthly_income'].median()).astype(int)
            self.monthly_income_median = df['monthly_income'].median()
        
        # Debt features
        df['debt_per_income'] = df['debt_ratio'] * df['monthly_income'] / (df['monthly_income'] + 1)
        df['high_utilization_flag'] = (df['revolving_utilization'] > 0.8).astype(int)
        
        # Interaction features
        df['age_income_interaction'] = df['age'] * df['high_income_flag']
        df['late_utilization_interaction'] = df['total_late_payments'] * df['revolving_utilization']
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], fit: bool = False) -> pd.DataFrame:
        """Cap outliers at 99th percentile"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                if fit:
                    # During fit, calculate and store the threshold
                    upper_limit = df[col].quantile(0.99)
                    self.outlier_thresholds[col] = upper_limit
                else:
                    # During transform, use stored threshold
                    upper_limit = self.outlier_thresholds.get(col, df[col].quantile(0.99))
                df[col] = df[col].clip(upper=upper_limit)
        return df
    
    def fit(self, df: pd.DataFrame) -> 'CreditDataPreprocessor':
        """Fit the preprocessor on training data"""
        df = df.copy()
        
        # Define feature groups
        numeric_features = ['revolving_utilization', 'age', 'late_30_59_days', 'debt_ratio',
                           'monthly_income', 'open_credit_lines', 'late_90_days',
                           'real_estate_loans', 'late_60_89_days', 'dependents']
        
        # Handle outliers (fit mode)
        outlier_cols = ['revolving_utilization', 'debt_ratio', 'monthly_income']
        df = self.handle_outliers(df, outlier_cols, fit=True)
        
        # Create new features
        df = self.create_features(df)
        
        # Get all feature columns (excluding target)
        feature_cols = [col for col in df.columns if col != 'target']
        self.feature_names = feature_cols
        
        # Fit imputer and scaler
        self.imputer.fit(df[feature_cols])
        self.scaler.fit(df[feature_cols])
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if self.feature_names is None:
            raise ValueError("Preprocessor not fitted. Call fit() or fit_transform() first.")
        
        df = df.copy()
        
        # Handle outliers (transform mode - use stored thresholds)
        outlier_cols = ['revolving_utilization', 'debt_ratio', 'monthly_income']
        df = self.handle_outliers(df, outlier_cols, fit=False)
        
        # Create new features (using stored statistics)
        df = self.create_features(df)
        
        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing expected feature: {col}")
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Impute missing values
        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=self.feature_names
        )
        
        # Scale features
        X_scaled = self.scaler.transform(df_imputed)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data"""
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: str) -> None:
        """Save preprocessor to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'outlier_thresholds': self.outlier_thresholds,
            'monthly_income_median': getattr(self, 'monthly_income_median', None)
        }, path)
    
    def load(self, path: str) -> None:
        """Load preprocessor from disk"""
        data = joblib.load(path)
        self.imputer = data['imputer']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.outlier_thresholds = data.get('outlier_thresholds', {})
        self.monthly_income_median = data.get('monthly_income_median', None)