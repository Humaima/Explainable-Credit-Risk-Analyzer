import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Download dataset manually from Kaggle: 
# https://www.kaggle.com/c/GiveMeSomeCredit/data
# Place cs-training.csv and cs-test.csv in data/ folder

# Load datasets
df = pd.read_csv('data/cs-training.csv', index_col=0)
print("Dataset shape:", df.shape)
print("First few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Rename columns for clarity
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


# Target distribution
print("\nTarget distribution:")
print(df['target'].value_counts(normalize=True))


# Visualize target distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.countplot(data=df, x='target', ax=axes[0,0])
axes[0,0].set_title('Target Distribution')

# Age distribution
sns.histplot(data=df, x='age', hue='target', ax=axes[0,1], alpha=0.5)
axes[0,1].set_title('Age Distribution by Target')

# Monthly income (cap at 99th percentile for visualization)
income_capped = df['monthly_income'].clip(upper=df['monthly_income'].quantile(0.99))
sns.histplot(data=df, x=income_capped, hue='target', ax=axes[0,2], alpha=0.5)
axes[0,2].set_title('Monthly Income Distribution (capped)')

# Revolving utilization
util_capped = df['revolving_utilization'].clip(upper=1)
sns.boxplot(data=df, x='target', y=util_capped, ax=axes[1,0])
axes[1,0].set_title('Revolving Utilization by Target')

# Debt ratio
debt_capped = df['debt_ratio'].clip(upper=df['debt_ratio'].quantile(0.99))
sns.boxplot(data=df, x='target', y=debt_capped, ax=axes[1,1])
axes[1,1].set_title('Debt Ratio by Target')

# Late payments
late_features = ['late_30_59_days', 'late_60_89_days', 'late_90_days']
df_late = df[late_features + ['target']].melt(id_vars=['target'])
sns.boxplot(data=df_late, x='variable', y='value', hue='target', ax=axes[1,2])
axes[1,2].set_title('Late Payments by Target')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/exploratory_analysis.png', dpi=300, bbox_inches='tight')
plt.show()