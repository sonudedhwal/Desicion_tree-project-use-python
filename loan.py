from sklearn.tree import DecisionTreeClassifier

# Training Data
X = [
    [30000, 600],
    [40000, 650],
    [50000, 700],
    [60000, 720],
    [70000, 750],
    [45000, 680]
]

# Target
y = ["Reject", "Reject", "Approve", "Approve", "Approve", "Reject"]

# Create Model
model = DecisionTreeClassifier()

# Train Model
model.fit(X, y)

# Prediction
income = 65000
credit_score = 730

prediction = model.predict([[income, credit_score]])

print("Income:", income)
print("Credit Score:", credit_score)
print("Loan Status:", prediction[0])