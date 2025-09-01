# Section 4.1: Data Preprocessing for Fraud Detection

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# File paths
ieee_path = r"C:\Users\richa\Dataset\transaction_data.csv"
cardholder_path = r"C:\Users\richa\Dataset\creditcard_2023.csv"

# Load datasets
ieee_df = pd.read_csv(ieee_path)
card_df = pd.read_csv(cardholder_path)

# Inspect datasets
print("IEEE-CIS Dataset Shape:", ieee_df.shape)
print("European Cardholder Dataset Shape:", card_df.shape)
print("\nIEEE-CIS Missing Values:\n", ieee_df.isnull().sum().sort_values(ascending=False).head())
print("\nCardholder Missing Values:\n", card_df.isnull().sum().sort_values(ascending=False).head())

# Drop irrelevant columns (IEEE only, if they exist)
columns_to_drop = ['TransactionID', 'Unnamed: 0', 'isFlaggedFraud', 'nameOrig', 'nameDest']
ieee_df.drop(columns=[col for col in columns_to_drop if col in ieee_df.columns], axis=1, inplace=True)

# Drop rows with excessive missing values (optional thresholding)
ieee_df.dropna(thresh=int(ieee_df.shape[1] * 0.7), inplace=True)

# Fill remaining missing values
ieee_df.fillna(-999, inplace=True)

# Encode categorical variables (IEEE dataset)
for col in ieee_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    ieee_df[col] = le.fit_transform(ieee_df[col])

# Dataset targets
ieee_X = ieee_df.drop('isFraud', axis=1)
ieee_y = ieee_df['isFraud']

card_X = card_df.drop('Class', axis=1)
card_y = card_df['Class']

# Scale numerical features
scaler = StandardScaler()
ieee_X_scaled = scaler.fit_transform(ieee_X)
card_X_scaled = scaler.fit_transform(card_X)

# Apply SMOTE to IEEE-CIS only (Card dataset is already balanced)
sm = SMOTE(random_state=42)
ieee_X_resampled, ieee_y_resampled = sm.fit_resample(ieee_X_scaled, ieee_y)

# Final dataset shapes
print("\nIEEE-CIS Balanced Shape:", ieee_X_resampled.shape, ieee_y_resampled.shape)
print("Cardholder Dataset Shape:", card_X_scaled.shape, card_y.shape)

# Optional: Save preprocessed datasets
pd.DataFrame(ieee_X_resampled).to_csv(r"C:\Users\richa\Dataset\ieee_X_resampled.csv", index=False)
pd.DataFrame(ieee_y_resampled).to_csv(r"C:\Users\richa\Dataset\ieee_y_resampled.csv", index=False)
pd.DataFrame(card_X_scaled).to_csv(r"C:\Users\richa\Dataset\card_X_scaled.csv", index=False)
pd.DataFrame(card_y).to_csv(r"C:\Users\richa\Dataset\card_y.csv", index=False)



# Section 4.2 – Descriptive Statistics and Visualisations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed datasets
ieee_df = pd.read_csv(r"C:\Users\richa\Dataset\transaction_data.csv")
card_df = pd.read_csv(r"C:\Users\richa\Dataset\creditcard_2023.csv")

# ----------------------------- #
# 1. Descriptive Statistics
# ----------------------------- #

# Compute descriptive stats for numeric columns
ieee_stats = ieee_df.describe().T
card_stats = card_df.describe().T

# Save tables (optional)
ieee_stats.to_csv(r"C:\Users\richa\Dataset\ieee_descriptive_stats.csv")
card_stats.to_csv(r"C:\Users\richa\Dataset\card_descriptive_stats.csv")

# ----------------------------- #
# 2. Boxplot/Histogram: Transaction Amount
# ----------------------------- #

plt.figure(figsize=(8, 6))
sns.boxplot(x='isFraud', y='amount', data=ieee_df)
plt.title("Boxplot of Transaction Amount by Fraud")
plt.xlabel("Is Fraud")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\boxplot_transaction_amount.png")
plt.show()

# ----------------------------- #
# 3. Count Plot: Transaction Type vs Fraud
# ----------------------------- #

if 'type' in ieee_df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=ieee_df, x='type', hue='isFraud')
    plt.title("Transaction Type vs Fraud Count")
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(r"C:\Users\richa\Dataset\transaction_type_vs_fraud.png")
    plt.show()

# ----------------------------- #
# 4. Bar Chart: Device Usage vs Fraud
# ----------------------------- #

if 'device' in ieee_df.columns:
    plt.figure(figsize=(8, 6))
    device_fraud = ieee_df.groupby(['device', 'isFraud']).size().unstack(fill_value=0)
    device_fraud.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Device Usage by Fraud Classification")
    plt.xlabel("Device")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(r"C:\Users\richa\Dataset\device_vs_fraud.png")
    plt.show()



# Section 4.3 – Correlation and Feature Importance Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings("ignore")

# Load the original IEEE dataset
ieee_df = pd.read_csv(r"C:\Users\richa\Dataset\transaction_data.csv")

# Preprocessing
columns_to_drop = ['TransactionID', 'Unnamed: 0', 'nameOrig', 'nameDest', 'isFlaggedFraud']
ieee_df.drop(columns=[col for col in columns_to_drop if col in ieee_df.columns], inplace=True, errors='ignore')
ieee_df.fillna(-999, inplace=True)

# Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
for col in ieee_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    ieee_df[col] = le.fit_transform(ieee_df[col])

# Separate features and target
X = ieee_df.drop('isFraud', axis=1)
y = ieee_df['isFraud']

# ----------------------------- #
# 1. Correlation Heatmap
# ----------------------------- #

plt.figure(figsize=(12, 10))
corr = X.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\correlation_heatmap.png")
plt.show()

# ----------------------------- #
# 2. Feature Importance (Random Forest)
# ----------------------------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
features = X.columns
imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=imp_df)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\feature_importance_rf.png")
plt.show()

# ----------------------------- #
# 3. SHAP Summary Plot (Optional)
# ----------------------------- #

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type='bar', show=False)
plt.title("SHAP Summary (Bar Plot)")
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\shap_summary_bar.png")
plt.close()

shap.summary_plot(shap_values[1], X_test, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\shap_summary.png")
plt.close()



# Section 4.4 – Class Distribution Summary

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load original IEEE dataset
ieee_df = pd.read_csv(r"C:\Users\richa\Dataset\transaction_data.csv")

# Drop unnecessary columns
columns_to_drop = ['TransactionID', 'Unnamed: 0', 'nameOrig', 'nameDest', 'isFlaggedFraud']
ieee_df.drop(columns=[col for col in columns_to_drop if col in ieee_df.columns], axis=1, inplace=True, errors='ignore')
ieee_df.fillna(-999, inplace=True)

# Label encoding
for col in ieee_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    ieee_df[col] = le.fit_transform(ieee_df[col])

# Feature/target split
X = ieee_df.drop('isFraud', axis=1)
y = ieee_df['isFraud']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Class distribution before SMOTE
before_counts = y.value_counts().rename_axis('Class').reset_index(name='Count')
print("Class Distribution Before SMOTE:\n", before_counts)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Class distribution after SMOTE
after_counts = pd.Series(y_resampled).value_counts().rename_axis('Class').reset_index(name='Count')
print("Class Distribution After SMOTE:\n", after_counts)

# Optional: Save both as CSV
before_counts.to_csv(r"C:\Users\richa\Dataset\class_distribution_before_smote.csv", index=False)
after_counts.to_csv(r"C:\Users\richa\Dataset\class_distribution_after_smote.csv", index=False)




# Section 4.5 – Model Building and Evaluation

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
X = pd.read_csv(r"C:\Users\richa\Dataset\ieee_X_resampled.csv")
y = pd.read_csv(r"C:\Users\richa\Dataset\ieee_y_resampled.csv").values.ravel()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------- #
# Train Supervised Models
# ------------------------- #

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
        "AUC": roc_auc_score(y_test, proba)
    })

# ------------------------- #
# Isolation Forest
# ------------------------- #
iso = IsolationForest(contamination=0.02, random_state=42)
iso.fit(X_train)
iso_preds = iso.predict(X_test)
iso_preds = np.where(iso_preds == -1, 1, 0)  # Convert outlier flag to fraud

results.append({
    "Model": "Isolation Forest",
    "Accuracy": accuracy_score(y_test, iso_preds),
    "Precision": precision_score(y_test, iso_preds),
    "Recall": recall_score(y_test, iso_preds),
    "F1 Score": f1_score(y_test, iso_preds),
    "AUC": roc_auc_score(y_test, iso_preds)
})

# ------------------------- #
# Autoencoder
# ------------------------- #

autoencoder = MLPRegressor(hidden_layer_sizes=(32, 16, 32), max_iter=20, random_state=42)
autoencoder.fit(X_train, X_train)

reconstructions = autoencoder.predict(X_test)
mse = np.mean((X_test - reconstructions) ** 2, axis=1)

threshold = np.percentile(mse, 95)
auto_preds = (mse > threshold).astype(int)

results.append({
    "Model": "Autoencoder",
    "Accuracy": accuracy_score(y_test, auto_preds),
    "Precision": precision_score(y_test, auto_preds),
    "Recall": recall_score(y_test, auto_preds),
    "F1 Score": f1_score(y_test, auto_preds),
    "AUC": roc_auc_score(y_test, auto_preds)
})

# ------------------------- #
# Save Evaluation Table
# ------------------------- #

eval_df = pd.DataFrame(results)
eval_df.to_csv(r"C:\Users\richa\Dataset\model_performance_metrics.csv", index=False)
print(eval_df)

# ------------------------- #
# F1 Score Bar Chart
# ------------------------- #

plt.figure(figsize=(8, 6))
sns.barplot(data=eval_df, x='Model', y='F1 Score')
plt.title("F1 Score Comparison Across Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\f1_score_bar_chart.png")
plt.show()

# ------------------------- #
# ROC Curve for Supervised Models
# ------------------------- #

plt.figure(figsize=(8, 6))
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\roc_curves.png")
plt.show()

# ------------------------- #
# Confusion Matrix for Autoencoder
# ------------------------- #

conf_matrix = confusion_matrix(y_test, auto_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Autoencoder")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\confusion_matrix_autoencoder.png")
plt.show()



# Section 4.6 – Cross-Dataset Comparison

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
ieee_X = pd.read_csv(r"C:\Users\richa\Dataset\ieee_X_resampled.csv")
ieee_y = pd.read_csv(r"C:\Users\richa\Dataset\ieee_y_resampled.csv").values.ravel()
card_X = pd.read_csv(r"C:\Users\richa\Dataset\card_X_scaled.csv")
card_y = pd.read_csv(r"C:\Users\richa\Dataset\card_y.csv").values.ravel()

# Train/Test splits
ieee_X_train, ieee_X_test, ieee_y_train, ieee_y_test = train_test_split(ieee_X, ieee_y, test_size=0.3, random_state=42)
card_X_train, card_X_test, card_y_train, card_y_test = train_test_split(card_X, card_y, test_size=0.3, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

comparison_results = []

# Train and evaluate on both datasets
for model_name, model in models.items():
    # IEEE-CIS
    model.fit(ieee_X_train, ieee_y_train)
    ieee_preds = model.predict(ieee_X_test)
    ieee_proba = model.predict_proba(ieee_X_test)[:, 1]
    
    # Cardholder
    model.fit(card_X_train, card_y_train)
    card_preds = model.predict(card_X_test)
    card_proba = model.predict_proba(card_X_test)[:, 1]
    
    comparison_results.extend([
        {"Model": model_name, "Dataset": "IEEE-CIS", "F1 Score": f1_score(ieee_y_test, ieee_preds), "AUC": roc_auc_score(ieee_y_test, ieee_proba)},
        {"Model": model_name, "Dataset": "Cardholder", "F1 Score": f1_score(card_y_test, card_preds), "AUC": roc_auc_score(card_y_test, card_proba)}
    ])

# Autoencoder separately (unsupervised)
def evaluate_autoencoder(X_train, X_test, y_test, dataset_name):
    ae = MLPRegressor(hidden_layer_sizes=(32, 16, 32), max_iter=20, random_state=42)
    ae.fit(X_train, X_train)
    reconstructions = ae.predict(X_test)
    mse = np.mean((X_test - reconstructions)**2, axis=1)
    threshold = np.percentile(mse, 95)
    auto_preds = (mse > threshold).astype(int)
    return {
        "Model": "Autoencoder",
        "Dataset": dataset_name,
        "F1 Score": f1_score(y_test, auto_preds),
        "AUC": roc_auc_score(y_test, auto_preds)
    }

comparison_results.append(evaluate_autoencoder(ieee_X_train, ieee_X_test, ieee_y_test, "IEEE-CIS"))
comparison_results.append(evaluate_autoencoder(card_X_train, card_X_test, card_y_test, "Cardholder"))

# Create DataFrame
compare_df = pd.DataFrame(comparison_results)
compare_df.to_csv(r"C:\Users\richa\Dataset\cross_dataset_model_comparison.csv", index=False)
print(compare_df)

# ----------------------------- #
# Grouped Bar Chart
# ----------------------------- #

plt.figure(figsize=(10, 6))
sns.barplot(data=compare_df, x="Model", y="F1 Score", hue="Dataset")
plt.title("F1 Score Comparison Across Datasets")
plt.tight_layout()
plt.savefig(r"C:\Users\richa\Dataset\grouped_f1_bar_chart.png")
plt.show()
