import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, roc_auc_score, classification_report
import pickle

# 1. Load Data
print("Loading data...")
df = pd.read_csv('churn_data.csv')

# 2. Preprocessing & Feature Engineering
print("Processing features...")
le = LabelEncoder()
df['Contract_Type_Encoded'] = le.fit_transform(df['Contract_Type'])

# Features for Prediction
features = ['Tenure_Months', 'Monthly_Revenue', 'Support_Tickets', 'Login_Frequency', 'Feature_Adoption_Score', 'Contract_Type_Encoded']
X = df[features]
y = df['Churn_Label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Churn Prediction (Random Forest)
print("Training Churn Prediction Model (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluation with custom threshold (Analyst Edge: Threshold = 0.3)
THRESHOLD = 0.3
print(f"Applying custom threshold: {THRESHOLD} to boost Recall...")
y_prob = rf_model.predict_proba(X_test)[:, 1]
y_pred_adj = (y_prob >= THRESHOLD).astype(int)

print("\n--- Model Performance (Optimized for Recall) ---")
print(f"Recall: {recall_score(y_test, y_pred_adj):.2%}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report (Custom Threshold):")
print(classification_report(y_test, y_pred_adj))

# --- Feature Importance Visualization (Analyst Edge) ---
import matplotlib.pyplot as plt
import seaborn as sns

print("\nExtracting Feature Importance...")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Save the plot
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('The Churn Detective: Top Drivers of Customer Churn', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved: 'feature_importance.png'")

# --- Confusion Matrix Visualization (Analyst Edge) ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_adj)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stay', 'Churn'])

plt.figure(figsize=(8,6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Optimized for Recall (Threshold=0.3)', fontsize=14)
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved: 'confusion_matrix.png'")

# 4. Behavioral Segmentation (K-Means)
print("\nPerforming Behavioral Segmentation (K-Means)...")
cluster_features = ['Login_Frequency', 'Feature_Adoption_Score', 'Support_Tickets']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Map Clusters to Business Personas (Vibe-based mapping)
cluster_map = {
    0: "Power Users",
    1: "Frustrated/High-Support",
    2: "Inactive/Low-Engagement",
    3: "Standard Users"
}
df['Segment'] = df['Cluster'].map(cluster_map)

# 5. Scoring the entire dataset
print("Scoring customers and saving results...")
all_probs = rf_model.predict_proba(X)[:, 1]
df['Churn_Probability'] = (all_probs * 100).round(2)
# Custom risk bins based on adjusted threshold (30% is the decision boundary)
df['Risk_Level'] = pd.cut(df['Churn_Probability'], bins=[0, 15, 30, 100], labels=['Low', 'Medium', 'High'], include_lowest=True)

# Save results
df.to_csv('scored_customers.csv', index=False)

# Save the model
with open('best_churn_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\nPipeline execution complete. Scored data saved to 'scored_customers.csv'.")
