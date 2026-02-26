# src/fairness_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional, Union

class FairnessAnalyzer:
    def __init__(self, sensitive_features: Dict):
        """
        sensitive_features: dict with feature name as key and privileged group as value
        e.g., {'age': {'threshold': 30}}
        """
        self.sensitive_features = sensitive_features
        self.results = {}
        
    def create_groups(self, df: pd.DataFrame, sensitive_feature: str, 
                     threshold: Optional[float] = None, 
                     categories: Optional[Dict] = None):
        """Create privileged and unprivileged groups"""
        if threshold is not None:
            # For numerical features
            privileged = df[sensitive_feature] > threshold
            unprivileged = df[sensitive_feature] <= threshold
        elif categories:
            # For categorical features
            privileged = df[sensitive_feature].isin(categories.get('privileged', []))
            unprivileged = df[sensitive_feature].isin(categories.get('unprivileged', []))
        else:
            # Default: use median as threshold
            threshold = df[sensitive_feature].median()
            privileged = df[sensitive_feature] > threshold
            unprivileged = df[sensitive_feature] <= threshold
        
        return privileged, unprivileged
    
    def calculate_demographic_parity(self, y_pred: np.ndarray, group_mask: pd.Series) -> float:
        """Calculate demographic parity (positive rate difference)"""
        if group_mask.sum() == 0:
            return 0.0
        group_pos_rate = y_pred[group_mask].mean()
        return float(group_pos_rate)
    
    def calculate_equal_opportunity(self, y_true: pd.Series, y_pred: np.ndarray, 
                                   group_mask: pd.Series) -> float:
        """Calculate equal opportunity (TPR difference)"""
        positive_actual = (y_true == 1)
        if positive_actual.sum() == 0:
            return 0.0
        group_tpr = y_pred[group_mask & positive_actual].sum() / max(positive_actual.sum(), 1)
        return float(group_tpr)
    
    def calculate_predictive_parity(self, y_true: pd.Series, y_pred: np.ndarray, 
                                   group_mask: pd.Series) -> float:
        """Calculate predictive parity (PPV difference)"""
        positive_pred = (y_pred == 1)
        if positive_pred.sum() == 0:
            return 0.0
        group_ppv = (y_pred[group_mask] & y_true[group_mask]).sum() / max(y_pred[group_mask].sum(), 1)
        return float(group_ppv)
    
    def analyze_fairness(self, df: pd.DataFrame, y_true: pd.Series, 
                        y_pred: np.ndarray, y_proba: np.ndarray) -> None:
        """Comprehensive fairness analysis"""
        for feature, config in self.sensitive_features.items():
            print(f"\n{'='*50}")
            print(f"Fairness Analysis for: {feature}")
            print('='*50)
            
            # Create groups
            if 'threshold' in config:
                privileged, unprivileged = self.create_groups(
                    df, feature, threshold=config['threshold']
                )
            elif 'categories' in config:
                privileged, unprivileged = self.create_groups(
                    df, feature, categories=config['categories']
                )
            else:
                privileged, unprivileged = self.create_groups(
                    df, feature, threshold=df[feature].median()
                )
            
            # Calculate metrics
            metrics = {}
            
            # Demographic Parity (acceptance rate)
            pos_rate_priv = self.calculate_demographic_parity(y_pred, privileged)
            pos_rate_unpriv = self.calculate_demographic_parity(y_pred, unprivileged)
            metrics['demographic_parity_ratio'] = pos_rate_priv / max(pos_rate_unpriv, 0.001)
            metrics['demographic_parity_diff'] = pos_rate_priv - pos_rate_unpriv
            
            # Equal Opportunity (True Positive Rate)
            tpr_priv = self.calculate_equal_opportunity(y_true, y_pred, privileged)
            tpr_unpriv = self.calculate_equal_opportunity(y_true, y_pred, unprivileged)
            metrics['equal_opportunity_ratio'] = tpr_priv / max(tpr_unpriv, 0.001)
            metrics['equal_opportunity_diff'] = tpr_priv - tpr_unpriv
            
            # Predictive Parity (Precision)
            ppv_priv = self.calculate_predictive_parity(y_true, y_pred, privileged)
            ppv_unpriv = self.calculate_predictive_parity(y_true, y_pred, unprivileged)
            metrics['predictive_parity_ratio'] = ppv_priv / max(ppv_unpriv, 0.001)
            metrics['predictive_parity_diff'] = ppv_priv - ppv_unpriv
            
            self.results[feature] = metrics
            
            # Print results
            print(f"\nPrivileged group positive rate: {pos_rate_priv:.3f}")
            print(f"Unprivileged group positive rate: {pos_rate_unpriv:.3f}")
            print(f"Demographic Parity Ratio: {metrics['demographic_parity_ratio']:.3f}")
            print(f"Demographic Parity Difference: {metrics['demographic_parity_diff']:.3f}")
            
            print(f"\nPrivileged group TPR: {tpr_priv:.3f}")
            print(f"Unprivileged group TPR: {tpr_unpriv:.3f}")
            print(f"Equal Opportunity Ratio: {metrics['equal_opportunity_ratio']:.3f}")
            
            print(f"\nPrivileged group PPV: {ppv_priv:.3f}")
            print(f"Unprivileged group PPV: {ppv_unpriv:.3f}")
            print(f"Predictive Parity Ratio: {metrics['predictive_parity_ratio']:.3f}")
    
    def plot_fairness_metrics(self) -> None:
        """Visualize fairness metrics"""
        if not self.results:
            print("No fairness results to plot")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        features = list(self.results.keys())
        metrics = ['demographic_parity_ratio', 'equal_opportunity_ratio', 'predictive_parity_ratio']
        titles = ['Demographic Parity', 'Equal Opportunity', 'Predictive Parity']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            values = [self.results[f][metric] for f in features]
            colors = ['green' if 0.8 <= v <= 1.25 else 'red' for v in values]
            
            axes[i].bar(features, values, color=colors)
            axes[i].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            axes[i].axhline(y=0.8, color='red', linestyle=':', alpha=0.5)
            axes[i].axhline(y=1.25, color='red', linestyle=':', alpha=0.5)
            axes[i].set_title(title)
            axes[i].set_ylabel('Ratio (Privileged/Unprivileged)')
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        # Create outputs directory if it doesn't exist
        os.makedirs('/outputs', exist_ok=True)
        plt.savefig('/outputs/fairness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
    def generate_fairness_report(self) -> List[str]:
        """Generate a fairness report"""
        report = []
        report.append("# Fairness Analysis Report\n")
        
        for feature, metrics in self.results.items():
            report.append(f"## {feature}\n")
            
            # Use ASCII characters instead of Unicode emojis
            if metrics['demographic_parity_ratio'] < 0.8 or metrics['demographic_parity_ratio'] > 1.25:
                report.append("[!] **Potential demographic parity violation**")
            else:
                report.append("[OK] Demographic parity satisfied")
                
            if metrics['equal_opportunity_ratio'] < 0.8 or metrics['equal_opportunity_ratio'] > 1.25:
                report.append("[!] **Potential equal opportunity violation**")
            else:
                report.append("[OK] Equal opportunity satisfied")
                
            if metrics['predictive_parity_ratio'] < 0.8 or metrics['predictive_parity_ratio'] > 1.25:
                report.append("[!] **Potential predictive parity violation**")
            else:
                report.append("[OK] Predictive parity satisfied")
            
            report.append("")
        
        # Write to file with UTF-8 encoding
        os.makedirs('/outputs', exist_ok=True)
        with open('/outputs/fairness_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Also print a summary to console
        print("\n" + "="*50)
        print("FAIRNESS ANALYSIS SUMMARY")
        print("="*50)
        for line in report[:10]:  # Print first 10 lines
            print(line)
        
        return report