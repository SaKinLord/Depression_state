# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:45:26 2025

@author: merto
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- Data Loading and Preprocessing -----------------------------

# Load the dataset
scores = pd.read_csv('scores.csv')

# Print dataset information and check for missing values
print(scores.info())
print(scores.describe())
print(scores.isnull().sum())

# Let's display the unique values ​​of categorical variables
categorical_columns = ["age", "edu", "gender", "afftype", "melanch", "inpatient", "marriage", "work"]
categorical_values = {col: scores[col].unique() for col in categorical_columns}


# Replace the spaces in the 'edu' column with NaN
scores['edu'] = scores['edu'].replace(' ', pd.NA)



# Determine missing data filling strategy
fill_strategies = {
    'afftype': scores['afftype'].mode()[0],   # Mode
    'melanch': scores['melanch'].mode()[0],   # Mode
    'inpatient': scores['inpatient'].mode()[0],  # Mode
    'edu': scores['edu'].mode()[0],  # Mod
    'marriage': scores['marriage'].mode()[0],  # Mode
    'work': scores['work'].mode()[0],  # Mode
    'madrs1': scores['madrs1'].median(),  # Median
    'madrs2': scores['madrs2'].median(),  # Median      
}
# Let's fill in the missing data
scores.fillna(fill_strategies, inplace=True)

# Define a function to convert ranges to mean for scores['edu']
def range_to_midpoint(range_str):
    try:
        lower, upper = map(int, range_str.split('-'))
        return (lower + upper) / 2
    except:
        return np.nan
scores['edu'] = scores['edu'].apply(range_to_midpoint)
# Define a function to convert ranges to mean for scores['age']
def range_to_midpoint(range_str):
    try:
        lower, upper = map(int, range_str.split('-'))
        return (lower + upper) / 2
    except:
        return np.nan
scores['age'] = scores['age'].apply(range_to_midpoint)
print(scores.isnull().sum())
scores.info()

# ----------------------------- Data Visualization -----------------------------

# Calculate the correlation matrix
corr = scores.corr(numeric_only=True)

# Plot heatmap to visualize correlations
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10}, linewidths=.5, linecolor='black')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Correlation Matrix", fontsize=18)
plt.show()

# Scatter plots for features vs madrs1
features = ['work', 'marriage', 'edu', 'inpatient', 'melanch', 'gender']
plt.figure(figsize=(20, 4))
for i, feat in enumerate(features):
    plt.subplot(1, 6, i + 1)
    plt.scatter(scores[feat], scores['madrs1'], alpha=0.7, edgecolor='k')
    plt.xlabel(feat)
    plt.ylabel('Y1')
    plt.title(f'{feat} vs madrs1')
plt.tight_layout()
plt.show()

# Scatter plots for features vs madrs2
plt.figure(figsize=(20, 4))
for i, feat in enumerate(features):
    plt.subplot(1, 6, i + 1)
    plt.scatter(scores[feat], scores['madrs2'], alpha=0.7, edgecolor='k')
    plt.xlabel(feat)
    plt.ylabel('Y2')
    plt.title(f'{feat} vs madrs2')
plt.tight_layout()
plt.show()

# Box plot for selected variables
columns = ['work', 'marriage', 'edu', 'inpatient', 'melanch', 'gender', 'madrs1', 'madrs2']
plt.figure(figsize=(12, 6))
scores[columns].boxplot(patch_artist=True,
                        boxprops=dict(facecolor='lightgreen'),
                        medianprops=dict(color='red'))
plt.title("Box Plot for Selected Variables")
plt.ylabel("Values")
plt.show()

# Pairplot for selected variables 
sns.pairplot(scores[columns], diag_kind='kde', corner=True)
plt.suptitle('Pair Plot: Relationships Among Selected Variables', fontsize=16, y=1.02) 
plt.show()

# Violin plot for selected variables 
columns_violin_features = ['work', 'marriage', 'inpatient', 'afftype', 'gender'] 
columns_violin_features = scores[columns_violin_features].melt(var_name='Variables', value_name='Value')
plt.figure(figsize=(12, 6)) 
sns.violinplot(x='Variables', y='Value', data=columns_violin_features, inner='quartile', color="skyblue") 
sns.stripplot(x='Variables', y='Value', data=columns_violin_features, color="black", alpha=0.2, jitter=True, size=3) 
plt.title("Violin Plot: Distributions of Smaller Scale Variables (Age Excluded)", fontsize=16) 
plt.ylabel("Values", fontsize=12)
plt.xlabel("Variables", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 3.5) 
plt.grid(axis='y', linestyle='-', alpha=0.3) 
sns.despine(left=True, bottom=True) 
plt.show()

# Violin Plot 2: MADRS Variables
columns_violin_madrs = ['madrs1', 'madrs2'] # MADRS variables
data_violin_madrs = scores[columns_violin_madrs].melt(var_name='Variables', value_name='Value')
plt.figure(figsize=(8, 6)) 
sns.violinplot(x='Variables', y='Value', data=data_violin_madrs, inner='quartile', color="lightcoral") 
sns.stripplot(x='Variables', y='Value', data=data_violin_madrs, color="black", alpha=0.3, jitter=True, size=4) 
plt.title("Violin Plot: Distributions of MADRS Scores", fontsize=16) 
plt.ylabel("MADRS Score", fontsize=12) 
plt.xlabel("MADRS Variables", fontsize=12) 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='-', alpha=0.3) 
sns.despine(left=True, bottom=True) 
plt.show()


# ----------------------------- Data Preparation -----------------------------

# Calculate the average of both MADRS scores
scores["madrs_avg"] = (scores["madrs1"] + scores["madrs2"]) / 2

# Define a function to assign a depression level based on the average MADRS score
def assign_depression_level(avg_score):
    if avg_score < 7:
        return "Normal (No Depression)"
    elif avg_score < 20:
        return "Mild Depression"
    elif avg_score < 35:
        return "Moderate Depression"
    else:
        return "Severe Depression"

# Apply the function to the average MADRS score to create the depression level column
scores["depression_level"] = scores["madrs_avg"].apply(assign_depression_level)

# ----------------------------- Data Split and Modelling -----------------------------
X = scores.iloc[:,1:10].values
y = scores['depression_level'].values

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
score_value1 = lr.score(X_test, y_test)
print("LogisticRegression Score:", score_value1)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 4, weights = 'distance')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
score_value2 = knn.score(X_test, y_test)
print("KNN Score:", score_value2)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_svc = grid_search.best_estimator_
y_pred_svc = best_svc.predict(X_test)
score_value3 = best_svc.score(X_test, y_test)
print("SVC Score:", score_value3)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred_tree = dtc.predict(X_test)
score_value4 = dtc.score(X_test, y_test)
print("DecisionTree Score:", score_value4)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred_random = rfc.predict(X_test)
score_value5 = rfc.score(X_test, y_test)
print("RandomForest Score:", score_value5)


# ----------------------------- Comparison Table -----------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example: Predictions from each model and their names
# Note: These variables should already be defined in your code.
# Make sure that the variable names match the predictions from your models.
models_predictions = {
    "Logistic Regression": y_pred_lr,
    "KNN": y_pred_knn,
    "SVC": y_pred_svc,
    "Decision Tree": y_pred_tree,
    "Random Forest": y_pred_random
}

# Create an empty dictionary to store the results
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Calculate the metrics for each model and add them to the dictionary
for model_name, y_pred in models_predictions.items():
    acc = accuracy_score(y_test, y_pred)
    # Use "weighted" average for multi-class classification; zero_division=0 avoids division errors
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results["Model"].append(model_name)
    results["Accuracy"].append(acc)
    results["Precision"].append(prec)
    results["Recall"].append(rec)
    results["F1 Score"].append(f1)

# Convert the dictionary to a DataFrame and print the results table
results_df = pd.DataFrame(results)
print(results_df)
# ----------------------------- Learning Curve -----------------------------

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Learning curve
def plot_learning_curve(estimator, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy'):
    """
    Plots the learning curve for ANY estimator to observe overfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training Score", marker='o')
    plt.plot(train_sizes, val_scores_mean, label="Validation Score", marker='o')
    plt.xlabel("Training Data Size")
    plt.ylabel("Accuracy Score")
    plt.title(f"Learning Curve for {estimator.__class__.__name__} (Best Model from GridSearchCV)") # Başlığa GridSearchCV bilgisi eklendi
    plt.legend()
    plt.grid(True)
    plt.show()

# Learning curve
plot_learning_curve(best_svc, X, y)


































