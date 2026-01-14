# The Churn Detective: Predictive Customer Retention AI Agent

**The Churn Detective** is an end-to-end AI-powered agent designed to proactively identify high-risk, high-value customers and automatically recommend targeted retention strategies. This project transforms raw behavioral data into executive-ready insights.

---

## 📈 Project Overview

Customer churn is a silent revenue killer. This project implements a full "Decision System" that doesn't just predict *who* will leave, but also *why* they are leaving and *what* to do about it.

### Core Architecture
1. **Data Hub**: Generates a business-realistic dataset of 10,000+ customers with tenure, revenue (LTV), and engagement metrics.
2. **Modeling Pipeline**: 
   - **Churn Prediction**: A Random Forest model optimized for **85.2% Recall** to catch nearly all potential churners.
   - **Behavioral Segmentation**: K-Means clustering to categorize users (e.g., Power Users vs. Frustrated Users).
3. **Retention Engine**: Maps AI outputs to business strategies and calculates **Revenue at Risk**.
4. **Automated Reporting**: Generates automated PDF Summaries, PowerPoint Slides, and Prioritized "To-Do" lists for Success Teams.

---

## 🛠️ Tech Stack

- **Core**: Python (Pandas, NumPy)
- **AI/ML**: Scikit-Learn (Random Forest, K-Means)
- **Visualization**: Matplotlib, Seaborn
- **Automated Reporting**: `python-pptx`, `fpdf2`

---

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn python-pptx fpdf2
   ```

2. **Execute the Pipeline**:
   Run the master orchestrator to generate all data, models, and reports:
   ```bash
   python main.py
   ```

---

## 📊 Key Results: The "Analyst Edge"

| Metric | Performance | Business Impact |
| --- | --- | --- |
| **Model Recall** | **85.23%** | Misses only 15% of churners (vs 51% with default thresholds). |
| **ROC AUC** | **0.74** | Strong discriminative power between churn and stay segments. |
| **Automation** | **100%** | Zero-touch pipeline from raw data to executive slide. |

### Economic Decision Logic
We use a **0.3 Classification Threshold** to prioritize **Recall over Precision**. 
- **Cost of Missing a Churner:** $4,000 (LTV)
- **Cost of a False Alarm:** $10 (Retention discount)
- **Decision:** It is mathematically superior to capture 35% more churners even if we experience more false positives.

---

## 📂 Project Outputs

- `Churn_Detective_Executive_Report.pdf`: Detailed project summary.
- `Churn_Detective_Executive_Slide.pptx`: Presentation slide for leadership.
- `feature_importance.png`: Visual guide to churn drivers.
- `priority_retention_list.csv`: Sorted list of high-value customers for immediate outreach.

---

## 🕵️‍♂️ Model explainability: What drives Churn?

Based on the [feature_importance.png](feature_importance.png), the primary drivers identified were:
1. **Support Tickets**: High volume strongly correlates with immediate risk.
2. **Login Frequency**: Declining engagement is a primary early-warning signal.
3. **Monthly Revenue**: High-value customers have distinct behavior patterns captured by the model.
