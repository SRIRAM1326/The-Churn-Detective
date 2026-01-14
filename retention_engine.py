import pandas as pd
import numpy as np

def map_retention_strategy(row):
    """Business logic to map churn risk to specific strategies."""
    if row['Risk_Level'] == 'Low':
        return "Customer Success - Standard Engagement"
    
    # High Risk High Value
    if row['Risk_Level'] == 'High' and row['Is_High_Value'] == 1:
        return "CRITICAL: Dedicated AM Outreach + Custom Discount"
    
    # High Support Segments
    if row['Segment'] == 'Frustrated/High-Support':
        return "Priority Support Follow-up - Technical Resolution"
    
    # Low engagement
    if row['Segment'] == 'Inactive/Low-Engagement':
        return "Product Reactivation Campaign - Feature Education"
    
    # High Risk
    if row['Risk_Level'] == 'High':
        return "Retention Specialist - Standard Loyalty Offer"
    
    # Medium Risk
    if row['Risk_Level'] == 'Medium':
        return "Automated Re-engagement Email Series"
    
    return "Standard Account Management"

# 1. Load Scored Data
print("Loading scored results...")
df = pd.read_csv('scored_customers.csv')

# 2. Apply Retention Strategy Mapping
print("Mapping retention strategies...")
df['Retention_Strategy'] = df.apply(map_retention_strategy, axis=1)

# 3. Calculate "Revenue at Risk"
# Revenue at Risk = Monthly Revenue * Churn Probability
df['Revenue_at_Risk'] = (df['Monthly_Revenue'] * (df['Churn_Probability'] / 100)).round(2)

# 4. Generate Priority Retention List
# Criteria: High Risk + High Value
priority_list = df[df['Risk_Level'] == 'High'].sort_values(by='Revenue_at_Risk', ascending=False)

# 5. Save Results
print("Saving actionable insights...")
df.to_csv('retention_insights.csv', index=False)
priority_list.head(100).to_csv('priority_retention_list.csv', index=False)

# --- Business Impact Visualization (Matplotlib) ---
import matplotlib.pyplot as plt
import seaborn as sns

print("\nGenerating Business Impact Chart...")
segment_revenue = df.groupby('Segment')['Revenue_at_Risk'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='Revenue_at_Risk', y='Segment', data=segment_revenue, palette='magma')
plt.title('Total Revenue at Risk by Customer Segment', fontsize=14)
plt.xlabel('Total Revenue at Risk ($)', fontsize=12)
plt.ylabel('Customer Segment', fontsize=12)
plt.tight_layout()
plt.savefig('revenue_by_segment.png')
print("Revenue impact chart saved: 'revenue_by_segment.png'")

print("\n--- Strategy Engine Complete ---")
print(f"Total Customers Scored: {len(df)}")
print(f"High Risk Customers: {len(df[df['Risk_Level'] == 'High'])}")
print(f"Total Revenue at Risk: ${df['Revenue_at_Risk'].sum():,.2f}")
print(f"Actionable strategies mapped for all risk levels.")
print("Saved: 'retention_insights.csv' and 'priority_retention_list.csv'")
