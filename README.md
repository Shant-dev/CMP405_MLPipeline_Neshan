# CMP405_MLPipeline_Neshan


# Online Shoppers Revenue Prediction with AI Agent

A complete Machine Learning pipeline and AI agent that predicts whether a user visiting an e-commerce website is likely to make a purchase (`Revenue = True`).

---

## Project Overview

This repository contains:

- A complete ML pipeline (EDA → preprocessing → model training → evaluation)
- SMOTE balancing for class imbalance
- Hyperparameter tuning with RandomizedSearchCV
- An interactive **AI Agent (RevenueAgent)** to simulate predictions
- User input via Google Colab for hands-on testing

---

## Objective

To accurately classify user sessions from an online shopping dataset into two classes:

- `True`: The user made a purchase
- `False`: No purchase was made

---

##  Dataset Info

- **Source:** UCI / Kaggle
- **Rows:** 12,330 sessions
- **Target Column:** `Revenue` (binary)
- **Imbalanced Classes:** Yes (~15% True)
- **Features:**
  - Behavioral: `Administrative`, `Informational`, `ProductRelated`, `BounceRates`, etc.
  - Contextual: `Month`, `VisitorType`, `Weekend`

---

##  Machine Learning Models

Initial Models:
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ XGBoost

**Best Performer:** Random Forest

### Model Improvements:
- Applied **SMOTE** for class balancing
- Tuned Random Forest using `RandomizedSearchCV`

### Final Metrics (after SMOTE + tuning):
| Metric     | Score |
|------------|-------|
| Accuracy   | 88.3% |
| Precision  | 60.1% |
| Recall     | 73.3% |
| F1-score   | 66.0% |
| ROC AUC    | ~0.92 |

---

## RevenueAgent (AI Agent)

We created a custom class to interact with the model:

```python
class RevenueAgent:
    def __init__(...):
        # Load model, encoders

    def predict(self, user_input_dict):
        # Process user input, predict, return message
