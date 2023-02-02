# for cleaning the data 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the cleaned data
df = pd.read_csv("phishing_emails_cleaned.csv")

# Split the data into training and test sets
X = df.drop(["is_phishing"], axis=1)
y = df["is_phishing"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate the classifier on the test data
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))

