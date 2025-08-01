# telco-churn-analysis-2
# 📌 Telco Customer Churn Analysis

# This notebook performs exploratory data analysis and predictive modeling on the Telco Customer Churn dataset.

# 🎯 Objectives:
# - Clean and preprocess the dataset
# - Explore key variables using visualizations
# - Build a classification model to predict customer churn
# - Present key findings and a dashboard
# 📥 Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

sns.set(style='whitegrid')
%matplotlib inline
# 📂 Load Dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
# 🔧 Data Cleaning

# Drop 'customerID' as it does not add value to analysis
df.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric and handle errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Map binary categorical variables to 0/1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 1, 0: 0})

# Encode remaining object-type categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
## 🔍 Data Cleaning

- **Why drop `customerID`?** It doesn’t contribute meaningful information to the analysis or modeling process.
- **Why drop missing values?** The missing data is limited to the `TotalCharges` column and dropping them avoids potential distortion from imputation.
- **Why map `Churn` and `SeniorCitizen` to 0/1?** It simplifies the analysis and is required for machine learning algorithms that expect numeric inputs.
# 📊 Correlation Heatmap

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
# 📈 Distribution of Tenure

sns.histplot(df['tenure'], kde=True, bins=30)
plt.title('Distribution of Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('Frequency')
plt.show()
## 📊 Exploratory Data Analysis

### Why each chart type?
- **Heatmap**: Helps identify correlation patterns across numerical features.
- **Distribution Plot**: Reveals skewness, spread, and shape of individual variables.
# 🎯 Modeling Setup

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
# 📋 Evaluation

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
### Insights:
- Features like `MonthlyCharges`, `tenure`, and `Contract` show moderate correlation with churn.
# Tenure distribution
sns.histplot(df['tenure'], kde=True, bins=30)
plt.title('Distribution of Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('Frequency')
plt.show()
# 📊 Dashboard Visualizations

# Pie chart of Churn
fig1 = px.pie(df, names='Churn', title='Churn Rate')
fig1.show()

# Bar chart of average tenure by churn
fig2 = px.bar(df.groupby('Churn')['tenure'].mean().reset_index(),
              x='Churn', y='tenure',
              title='Average Tenure by Churn Status')
fig2.show()
## 🎯 Modeling

- **X variables**: All features except `Churn`
- **y variable**: `Churn`
- **Goal**: Predict which customers are likely to churn
- **Why Logistic Regression?** Simple, interpretable, and effective for binary classification.
## 📊 Dashboard

- **Why Pie Chart?** To show overall churn distribution at a glance.
- **Why Bar Chart?** To compare average tenure across churn categories.

