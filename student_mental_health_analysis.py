
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Student Mental health.csv')

df.head()

print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

df.dtypes

df.describe()

df.isnull().sum()

df.columns

df = df.drop('Timestamp', axis=1)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

categorical_columns = ['Choose your gender', 'What is your course?', 'Your current year of Study', 'Marital status', 'Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical_columns = ['Age',]  # replace this with your actual numerical columns

for column in numerical_columns:
    df[column] = scaler.fit_transform(df[[column]])

numerical_columns = ['Age', 'What is your CGPA?']

for column in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Variables")
plt.show()

# Boxplots for numerical variables vs categorical variables
categorical_columns = ['Choose your gender', 'What is your course?', 'Your current year of Study', 'Marital status', 'Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']

for cat_column in categorical_columns:
    for num_column in numerical_columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(data=df, x=cat_column, y=num_column)
        plt.title(f"Boxplot of {num_column} by {cat_column}")
        plt.xticks(rotation=45)
        plt.show()

#Count plots for categorical columns
categorical_columns = ['Choose your gender', 'What is your course?', 'Your current year of Study', 'Marital status', 'Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']

for column in categorical_columns:
    plt.figure(figsize=(16, 4))
    sns.countplot(data=df, x=column)
    plt.title(f"Count Plot for {column}")
    plt.xticks(rotation=45)
    plt.show()

# Create a new feature that counts the total number of mental health issues each student has
df['Total Mental Health Issues'] = df['Do you have Depression?'] + df['Do you have Anxiety?'] + df['Do you have Panic attack?']

df['CGPA Midpoint'] = df['What is your CGPA?'].map({
    '3.00 - 3.49': 3.25,
    '3.50 - 3.99': 3.75,
    '2.50 - 2.99': 2.75,
    '2.00 - 2.49': 2.25,
    '1.50 - 1.99': 1.75,
    '1.00 - 1.49': 1.25,
    '0.50 - 0.99': 0.75,
    '0.00 - 0.49': 0.25
})

df = df.drop('What is your CGPA?', axis=1)

# Convert 'Your current year of Study' to a numerical value
df['Study Year'] = df['Your current year of Study'].map({
    'year 1': 1,
    'year 2': 2,
    'year 3': 3,
    'year 4': 4
    # Add more years if necessary
})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define feature matrix X and target vector y
X = df.drop('Total Mental Health Issues', axis=1)
y = df['Total Mental Health Issues']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

num_cols = ['Age']
cat_cols = ['Choose your gender', 'What is your course?', 'Your current year of Study', 'Marital status', 'Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']

from sklearn.impute import SimpleImputer

# Impute missing values with the median for numerical columns
num_imputer = SimpleImputer(strategy='median')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Impute missing values with the most frequent category for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

median_value = X_train['CGPA Midpoint'].median()
X_train['CGPA Midpoint'].fillna(median_value, inplace=True)
X_test['CGPA Midpoint'].fillna(median_value, inplace=True)

# Drop the 'Study Year' column from the training and test sets
X_train = X_train.drop('Study Year', axis=1)
X_test = X_test.drop('Study Year', axis=1)

model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_evaluation_metrics(y_test, y_pred, average='micro'):
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average=average)}")
    print(f"Recall: {recall_score(y_test, y_pred, average=average)}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average=average)}")

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

for model_name, model in models.items():
    print(f"Model: {model_name}")
    y_pred = model.predict(X_test)
    print_evaluation_metrics(y_test, y_pred)
    print("\n")