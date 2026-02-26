import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

class ModelExplainer:
    def __init__(self, model: Union[RandomForestClassifier, XGBClassifier, LogisticRegression], 
                 X_train: pd.DataFrame, feature_names: list):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_explainer: Optional[Any] = None
        self.lime_explainer: Optional[Any] = None
        
    def explain_with_shap(self, X_sample: pd.DataFrame, plot_type: str = 'bar') -> Optional[np.ndarray]:
        """Generate SHAP explanations"""
        # Create SHAP explainer
        if self.shap_explainer is None:
            # Use TreeExplainer for tree-based models, KernelExplainer for others
            if hasattr(self.model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # Use a sample of background data for KernelExplainer
                background = self.X_train.sample(n=min(100, len(self.X_train)), random_state=42)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create plots
        plt.figure(figsize=(12, 5))
        
        if plot_type == 'bar':
            plt.subplot(1, 2, 1)
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                            show=False, plot_type='bar')
            plt.title('SHAP Feature Importance (Bar)')
            
            plt.subplot(1, 2, 2)
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                            show=False)
            plt.title('SHAP Summary Plot')
        
        elif plot_type == 'waterfall' and X_sample.shape[0] == 1:
            # Waterfall plot for single prediction
            shap.waterfall_plot(
                shap.Explanation(values=shap_values[0],
                                base_values=self.shap_explainer.expected_value if not isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) else self.shap_explainer.expected_value[1],
                                data=X_sample.iloc[0].values,
                                feature_names=self.feature_names)
            )
        
        plt.tight_layout()
        plt.savefig('outputs/shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values
    
    def explain_with_lime(self, X_sample: pd.DataFrame, instance_idx: int = 0) -> Any:
        """Generate LIME explanations"""
        if self.lime_explainer is None:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                class_names=['Good', 'Bad'],
                mode='classification',
                discretize_continuous=True
            )
        
        # Get single instance
        instance = X_sample.iloc[instance_idx:instance_idx+1]
        
        # Generate explanation
        exp = self.lime_explainer.explain_instance(
            data_row=instance.values[0],
            predict_fn=lambda x: self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names)),
            num_features=10
        )
        
        # Plot explanation
        fig = exp.as_pyplot_figure()
        plt.title(f'LIME Explanation - Instance {instance_idx}')
        plt.tight_layout()
        plt.savefig(f'outputs/lime_explanation_{instance_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return exp
    
    def compare_explanations(self, X_sample: pd.DataFrame, instance_idx: int = 0) -> None:
        """Compare SHAP and LIME explanations for same instance"""
        # Get SHAP values for single instance
        shap_values = self.explain_with_shap(
            X_sample.iloc[instance_idx:instance_idx+1], 
            plot_type='waterfall'
        )
        
        # Get LIME explanation
        lime_exp = self.explain_with_lime(X_sample, instance_idx)
        
        if shap_values is not None:
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # SHAP waterfall values
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'value': np.abs(shap_values[0])
            }).sort_values('value', ascending=False).head(10)
            
            axes[0].barh(range(len(shap_importance)), shap_importance['value'].values)
            axes[0].set_yticks(range(len(shap_importance)))
            axes[0].set_yticklabels(shap_importance['feature'].values)
            axes[0].set_xlabel('|SHAP value|')
            axes[0].set_title('SHAP Top 10 Features')
            axes[0].invert_yaxis()
            
            # LIME feature weights
            lime_weights = pd.DataFrame(lime_exp.as_list(), columns=['feature', 'weight'])
            lime_weights['abs_weight'] = lime_weights['weight'].abs()
            lime_weights = lime_weights.sort_values('abs_weight', ascending=False).head(10)
            
            colors = ['red' if w < 0 else 'green' for w in lime_weights['weight']]
            axes[1].barh(range(len(lime_weights)), lime_weights['weight'].values, color=colors)
            axes[1].set_yticks(range(len(lime_weights)))
            axes[1].set_yticklabels(lime_weights['feature'].values)
            axes[1].set_xlabel('LIME weight')
            axes[1].set_title('LIME Top 10 Features')
            axes[1].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig('outputs/shap_vs_lime_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()