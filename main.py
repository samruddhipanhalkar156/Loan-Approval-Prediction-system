import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Features and Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Remove loan_id
X = X.drop("loan_id", axis=1)

# Label Encoding
le = LabelEncoder()

X["education"] = le.fit_transform(X["education"])
X["self_employed"] = le.fit_transform(X["self_employed"])
y = le.fit_transform(y)

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Accuracy
print("\nModel Accuracy")
print("-" * 30)
print("Logistic Regression :", accuracy_score(y_test, lr.predict(X_test)))
print("Decision Tree       :", accuracy_score(y_test, dt.predict(X_test)))
print("Random Forest       :", accuracy_score(y_test, rf.predict(X_test)))

# Save Models
joblib.dump(lr, "logistic_regression.pkl")
joblib.dump(dt, "decision_tree.pkl")
joblib.dump(rf, "random_forest.pkl")

print("\nAll models saved successfully!")
print("feature_names.pkl")
print("logistic_regression.pkl")
print("decision_tree.pkl")
print("random_forest.pkl")