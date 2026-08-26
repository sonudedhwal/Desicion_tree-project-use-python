import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Training Data
df = pd.DataFrame({
    "Age":    [15, 16, 17, 20, 22, 25, 30, 35],
    "Salary": [0,  0,  0,  25000, 30000, 50000, 60000, 70000],
    "Buy":    [0,  0,  0,      0,     0,     1,     1,     1]
})

# Features
X = df[["Age", "Salary"]]

# Target
y = df["Buy"]

# Create Decision Tree
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# New Person: Age = 28, Salary = 55000
result = model.predict([[28, 55000]])

print(result)

