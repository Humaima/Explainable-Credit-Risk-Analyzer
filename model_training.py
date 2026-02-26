# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                            recall_score, f1_score, confusion_matrix,
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Optional, Tuple, List, Union, Any

class CreditRiskModel:
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = self._create_model()
        self.feature_importance: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        
    def _create_model(self) -> Union[LogisticRegression, RandomForestClassifier, XGBClassifier]:
        """Create model based on type"""
        if self.model_type == 'logistic':
            return LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:  # xgboost
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=5,  # Adjust for imbalance
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> None:
        """Train the model"""
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # XGBoost supports eval_set, others don't
        if self.model_type == 'xgboost' and X_val is not None and y_val is not None:
            # For XGBoost, keep as DataFrames to preserve feature names
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature importance based on model type
        if self.model_type == 'logistic':
            if hasattr(self.model, 'coef_'):
                self.feature_importance = np.abs(self.model.coef_[0])
        else:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                 dataset_name: str = "Test") -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        print(f"\n{dataset_name} Set Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name} Set')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'outputs/confusion_matrix_{dataset_name.lower()}.png')
        plt.close()  # Close the figure to free memory
        plt.show()
        
        return metrics, cm
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
        """Perform cross-validation"""
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        print(f"\nCross-Validation ROC-AUC Scores: {cv_scores}")
        print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def plot_roc_curves(self, models_dict: Dict[str, 'CreditRiskModel'], 
                        X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Compare ROC curves of multiple models"""
        plt.figure(figsize=(10, 8))
        
        for name, model_obj in models_dict.items():
            y_proba = model_obj.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        plt.show()
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_importance = data['feature_importance']
        self.feature_names = data.get('feature_names', None)