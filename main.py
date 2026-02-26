# src/main_fixed.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import CreditDataPreprocessor
from src.model_training import CreditRiskModel
from src.model_explainability import ModelExplainer
from src.fairness_analysis import FairnessAnalyzer

warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("EXPLAINABLE CREDIT RISK MODEL - FIXED VERSION")
    print("="*60)
    
    # Get absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    outputs_dir = os.path.join(project_root, 'outputs')
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Models will be saved to: {models_dir}")
    print(f"Outputs will be saved to: {outputs_dir}")
    
    # 1. Load data
    print("\n1. Loading data...")
    data_path = os.path.join(data_dir, 'cs-training.csv')
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please download the dataset from Kaggle and place it in the data/ folder")
        return
    
    df = pd.read_csv(data_path, index_col=0)
    
    # Rename columns
    column_names = {
        'SeriousDlqin2yrs': 'target',
        'RevolvingUtilizationOfUnsecuredLines': 'revolving_utilization',
        'age': 'age',
        'NumberOfTime30-59DaysPastDueNotWorse': 'late_30_59_days',
        'DebtRatio': 'debt_ratio',
        'MonthlyIncome': 'monthly_income',
        'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines',
        'NumberOfTimes90DaysLate': 'late_90_days',
        'NumberRealEstateLoansOrLines': 'real_estate_loans',
        'NumberOfTime60-89DaysPastDueNotWorse': 'late_60_89_days',
        'NumberOfDependents': 'dependents'
    }
    df = df.rename(columns=column_names)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = CreditDataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    
    # Split features and target
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 3. Train models
    print("\n3. Training models...")
    models = {}
    
    for model_type in ['logistic', 'random_forest', 'xgboost']:
        print(f"\nTraining {model_type}...")
        try:
            model = CreditRiskModel(model_type=model_type)
            model.train(X_train, y_train, X_val, y_val)
            models[model_type] = model
            
            # Evaluate on validation set
            model.evaluate(X_val, y_val, dataset_name="Validation")
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not models:
        print("No models were successfully trained!")
        return
    
    # 4. Compare models
    print("\n4. Comparing models on test set...")
    best_model = None
    best_auc = 0
    best_model_name = ""
    
    for name, model in models.items():
        try:
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba)
            print(f"{name} Test AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    if best_model is None:
        print("No valid model found for evaluation!")
        return
    
    print(f"\nBest model: {best_model_name} with AUC = {best_auc:.4f}")
    
    # Plot ROC curves
    try:
        best_model.plot_roc_curves(models, X_test, y_test)
    except Exception as e:
        print(f"Error plotting ROC curves: {e}")
    
    # 5. Model explainability
    print("\n5. Generating model explanations...")
    try:
        explainer = ModelExplainer(
            best_model.model,
            X_train,
            X_train.columns.tolist()
        )
        
        # SHAP analysis
        print("\nRunning SHAP analysis...")
        X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
        shap_values = explainer.explain_with_shap(X_sample)
        
        # LIME analysis
        print("\nRunning LIME analysis...")
        lime_exp = explainer.explain_with_lime(X_test, instance_idx=0)
        
        # Compare explanations
        print("\nComparing SHAP and LIME...")
        explainer.compare_explanations(X_test, instance_idx=0)
    except Exception as e:
        print(f"Error in model explainability: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Fairness analysis
    print("\n6. Analyzing fairness...")
    try:
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)
        
        # Define sensitive features
        sensitive_features = {
            'age': {'threshold': 30},
            'dependents': {'threshold': 2},
            'monthly_income': {'threshold': X_test['monthly_income'].median()}
        }
        
        fairness_analyzer = FairnessAnalyzer(sensitive_features)
        fairness_analyzer.analyze_fairness(X_test, y_test, y_pred, y_proba)
        fairness_analyzer.plot_fairness_metrics()
        fairness_analyzer.generate_fairness_report()
    except Exception as e:
        print(f"Error in fairness analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Save best model and preprocessor
    print("\n7. Saving model and preprocessor...")
    try:
        model_path = os.path.join(models_dir, 'best_model.joblib')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        
        print(f"Saving model to: {model_path}")
        print(f"Saving preprocessor to: {preprocessor_path}")
        
        best_model.save(model_path)
        preprocessor.save(preprocessor_path)
        
        # Verify files were created
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"✅ Model saved successfully! Size: {file_size} bytes")
        else:
            print(f"❌ Model file not found at {model_path}")
            
        if os.path.exists(preprocessor_path):
            file_size = os.path.getsize(preprocessor_path)
            print(f"✅ Preprocessor saved successfully! Size: {file_size} bytes")
        else:
            print(f"❌ Preprocessor file not found at {preprocessor_path}")
            
    except Exception as e:
        print(f"Error saving models: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("Models saved in:", models_dir)
    print("="*60)

if __name__ == "__main__":
    main()