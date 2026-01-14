import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
num_records = 10000

# 1. Generate Base Features
data = {
    'Customer_ID': [f'CUST-{i:05d}' for i in range(num_records)],
    'Tenure_Months': np.random.randint(1, 60, num_records),
    'Contract_Type': np.random.choice(['Monthly', 'One Year', 'Two Year'], num_records, p=[0.5, 0.3, 0.2]),
    'Monthly_Revenue': np.random.uniform(50, 500, num_records).round(2),
    'Support_Tickets': np.random.poisson(1.5, num_records),
    'Login_Frequency': np.random.randint(0, 31, num_records), # Logins per month
    'Feature_Adoption_Score': np.random.randint(1, 11, num_records) # Scale 1-10
}

df = pd.DataFrame(data)

# Calculate Lifetime Value (LTV)
df['LTV'] = (df['Monthly_Revenue'] * df['Tenure_Months']).round(2)

# 2. Create "Vibe-Based" Churn Logic (Mathematical Integrity)
# We want Churn to be higher if:
# - High Support Tickets
# - Low Login Frequency
# - Monthly Contract
# - Low Feature Adoption
# - Low Tenure (early churn)

# Calculate a "Churn Probability" score
churn_logic = (
    (df['Support_Tickets'] * 0.25) - 
    (df['Login_Frequency'] * 0.08) - 
    (df['Feature_Adoption_Score'] * 0.15) -
    (df['Tenure_Months'] * 0.01) +
    (df['Contract_Type'].map({'Monthly': 0.6, 'One Year': 0.2, 'Two Year': 0.05}))
)

# Normalize and convert to binary Label (1 = Churned, 0 = Retained)
prob = 1 / (1 + np.exp(-churn_logic)) # Sigmoid function
df['Churn_Label'] = np.where(np.random.rand(num_records) < prob, 1, 0)

# Add a "High Value Customer" flag (top 20% by Monthly Revenue)
revenue_threshold = df['Monthly_Revenue'].quantile(0.8)
df['Is_High_Value'] = np.where(df['Monthly_Revenue'] >= revenue_threshold, 1, 0)

# 3. Save to CSV
df.to_csv('churn_data.csv', index=False)
print(f"Dataset created: {num_records} records saved as 'churn_data.csv'")
print(f"Revenue range: ${df['Monthly_Revenue'].min()} - ${df['Monthly_Revenue'].max()}")
print(f"Churn rate: {df['Churn_Label'].mean():.2%}")