from sklearn.tree import DecisionTreeClassifier

# Features:
# [Experience (Years), Performance Score]

X = [
    [2, 60],
    [5, 80],
    [3, 65],
    [8, 95],
    [1, 50],
    [6, 90]
]

y = [
    "No Promotion",
    "Promotion",
    "No Promotion",
    "Promotion",
    "No Promotion",
    "Promotion"
]

model = DecisionTreeClassifier()

model.fit(X, y)

new_employee = [[7, 88]]

prediction = model.predict(new_employee)

print("Result:", prediction[0])