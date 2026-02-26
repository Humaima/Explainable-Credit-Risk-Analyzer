# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the preprocessor class
from src.data_preprocessing import CreditDataPreprocessor

# Page config
st.set_page_config(
    page_title="Credit Risk Explainability Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .good-risk {
        color: #4caf50;
        font-weight: bold;
    }
    .bad-risk {
        color: #f44336;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💰 Credit Risk Explainability Dashboard</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Understand why loan applications are approved or rejected</p>', 
            unsafe_allow_html=True)

# Load models and preprocessor
@st.cache_resource
def load_artifacts():
    try:
        model_path = 'models/best_model.joblib'
        preprocessor_path = 'models/preprocessor.joblib'
        
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}")
            return None, None
            
        if not os.path.exists(preprocessor_path):
            st.error(f"Preprocessor not found at {preprocessor_path}")
            return None, None
        
        # Load model
        model_data = joblib.load(model_path)
        model = model_data['model']
        
        # Load preprocessor
        preprocessor_data = joblib.load(preprocessor_path)
        
        # Reconstruct preprocessor
        preprocessor = CreditDataPreprocessor()
        
        if isinstance(preprocessor_data, dict):
            preprocessor.imputer = preprocessor_data['imputer']
            preprocessor.scaler = preprocessor_data['scaler']
            preprocessor.feature_names = preprocessor_data['feature_names']
            preprocessor.outlier_thresholds = preprocessor_data.get('outlier_thresholds', {})
            preprocessor.monthly_income_median = preprocessor_data.get('monthly_income_median', None)
        else:
            preprocessor = preprocessor_data
        
        # Test the preprocessor with a small sample
        test_df = pd.DataFrame({
            'revolving_utilization': [0.3],
            'age': [35],
            'late_30_59_days': [0],
            'debt_ratio': [0.4],
            'monthly_income': [5000],
            'open_credit_lines': [5],
            'late_90_days': [0],
            'real_estate_loans': [1],
            'late_60_89_days': [0],
            'dependents': [0]
        })
        
        # This should work without errors
        _ = preprocessor.transform(test_df)
        
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

model, preprocessor = load_artifacts()

if model is None or preprocessor is None:
    st.warning("""
    ⚠️ Models not found or failed to load! Please run the training pipeline first:
    
    ```
    cd src
    python main_fixed.py
    ```
    
    Make sure the training completes successfully and the models are saved in the 'models' folder.
    """)
    
    # Show files in current directory for debugging
    st.write("Files in current directory:", os.listdir('.'))
    if os.path.exists('models'):
        st.write("Files in models directory:", os.listdir('models'))
    
    st.stop()

# Sidebar
st.sidebar.header("📋 Applicant Information")

# Create input form
with st.sidebar.form("applicant_form"):
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    
    st.subheader("Financial Information")
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=500)
    revolving_utilization = st.slider("Revolving Credit Utilization", 0.0, 1.0, 0.3, 0.01)
    debt_ratio = st.slider("Debt Ratio", 0.0, 2.0, 0.4, 0.01)
    
    st.subheader("Credit History")
    open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=5)
    real_estate_loans = st.number_input("Real Estate Loans", min_value=0, max_value=10, value=1)
    
    st.subheader("Late Payments")
    late_30_59 = st.number_input("30-59 Days Late", min_value=0, max_value=20, value=0)
    late_60_89 = st.number_input("60-89 Days Late", min_value=0, max_value=20, value=0)
    late_90 = st.number_input("90+ Days Late", min_value=0, max_value=20, value=0)
    
    submitted = st.form_submit_button("🔍 Predict & Explain")

# Main content
if submitted:
    # Create dataframe from input
    input_data = pd.DataFrame({
        'revolving_utilization': [revolving_utilization],
        'age': [age],
        'late_30_59_days': [late_30_59],
        'debt_ratio': [debt_ratio],
        'monthly_income': [monthly_income],
        'open_credit_lines': [open_credit_lines],
        'late_90_days': [late_90],
        'real_estate_loans': [real_estate_loans],
        'late_60_89_days': [late_60_89],
        'dependents': [dependents]
    })
    
    try:
        # Show input data for debugging
        with st.expander("Debug: Input Data"):
            st.write("Base features:", input_data.columns.tolist())
            st.write(input_data)
        
        # Preprocess input
        input_processed = preprocessor.transform(input_data)
        
        with st.expander("Debug: Processed Data"):
            st.write("Engineered features:", input_processed.columns.tolist())
            st.write(input_processed)
            st.write(f"Number of features: {len(input_processed.columns)}")
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_processed)[0, 1]
            prediction = model.predict(input_processed)[0]
        else:
            prediction = model.predict(input_processed)[0]
            probability = 0.5
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Default Probability", f"{probability:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown('<p class="bad-risk">⚠️ HIGH RISK</p>', unsafe_allow_html=True)
                st.markdown("**Loan Status: Likely to Default**")
            else:
                st.markdown('<p class="good-risk">✅ LOW RISK</p>', unsafe_allow_html=True)
                st.markdown("**Loan Status: Likely to Repay**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            confidence = max(probability, 1-probability)
            st.metric("Confidence", f"{confidence:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Explainability section
        st.header("🔍 Decision Explanation")
        
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Raw Values", "What-If Analysis"])
        
        with tab1:
            st.subheader("Top Factors Influencing Decision")
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                importance = np.ones(len(input_processed.columns)) / len(input_processed.columns)
            
            feature_names = input_processed.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(10)
            
            # Create bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show SHAP analysis if possible
            try:
                if hasattr(model, 'feature_importances_'):
                    st.subheader("SHAP Analysis")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_processed)
                    
                    # Summary plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values[0] if isinstance(shap_values, list) else shap_values, 
                                     input_processed, show=False)
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.info(f"SHAP analysis not available: {str(e)}")
        
        with tab2:
            st.subheader("Processed Feature Values")
            # Display the actual values after preprocessing
            processed_df = input_processed.T.reset_index()
            processed_df.columns = ['Feature', 'Value']
            processed_df['Value'] = processed_df['Value'].round(3)
            st.dataframe(processed_df, use_container_width=True)
        
        with tab3:
            st.subheader("What-If Analysis")
            st.markdown("""
            Adjust the sliders below to see how changes in applicant information 
            would affect the default probability.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_revolving = st.slider(
                    "Test Revolving Utilization", 
                    0.0, 1.0, revolving_utilization, 0.05
                )
                test_age = st.slider("Test Age", 18, 100, age, 1)
                
            with col2:
                test_income = st.slider(
                    "Test Monthly Income", 
                    0, 20000, monthly_income, 500
                )
                test_late_payments = st.slider(
                    "Test Total Late Payments", 
                    0, 20, late_30_59 + late_60_89 + late_90, 1
                )
            
            # Create test instance
            test_data = input_data.copy()
            test_data['revolving_utilization'] = test_revolving
            test_data['age'] = test_age
            test_data['monthly_income'] = test_income
            # Distribute late payments evenly
            test_data['late_30_59_days'] = test_late_payments // 3
            test_data['late_60_89_days'] = test_late_payments // 3
            test_data['late_90_days'] = test_late_payments // 3
            
            try:
                # Preprocess and predict
                test_processed = preprocessor.transform(test_data)
                test_prob = model.predict_proba(test_processed)[0, 1]
                
                # Show comparison
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Probability", f"{probability:.1%}")
                col2.metric("New Probability", f"{test_prob:.1%}")
                delta = (test_prob - probability) * 100
                col3.metric("Change", f"{delta:+.1f}%", delta=delta)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=test_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': test_prob * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in what-if analysis: {str(e)}")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👈 Fill in the applicant information form and click 'Predict & Explain' to get started!")
    
    # Show sample explanation
    st.header("📊 About This Dashboard")
    st.markdown("""
    This dashboard provides:
    - **Prediction**: Whether the applicant is likely to default
    - **Feature Importance**: Which factors most influence the decision
    - **SHAP Analysis**: Detailed feature-by-feature explanation (when available)
    - **What-If Analysis**: How changes in applicant information affect the decision
    
    The model uses 21 features including engineered ones like:
    - Total late payments across all buckets
    - Credit lines per age
    - Income per dependent
    - Debt-to-income ratio
    - And more...
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, SHAP, and XGBoost</p>
    <p style='font-size: 0.8rem; color: gray;'>
        This is a demonstration project. Decisions should not be based solely on this model.
    </p>
</div>
""", unsafe_allow_html=True)