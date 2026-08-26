from sklearn.tree import DecisionTreeClassifier

# Training Data
X = [
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 90]
]

# Target
y = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]

# Create Model
model = DecisionTreeClassifier()

# Train Model
model.fit(X, y)

# Prediction
study_hours = 6
attendance = 82

prediction = model.predict([[study_hours, attendance]])

print("Study Hours:", study_hours)
print("Attendance:", attendance)
print("Result:", prediction[0])