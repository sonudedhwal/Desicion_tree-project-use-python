from sklearn.tree import DecisionTreeClassifier

# Features:
# [Weight (grams), Color]
# Color: 1 = Red, 0 = Green

X = [
    [150, 1],
    [170, 1],
    [120, 0],
    [100, 0],
    [180, 1],
    [90, 0]
]

# Output
y = [
    "Apple",
    "Apple",
    "Guava",
    "Guava",
    "Apple",
    "Guava"
]

model = DecisionTreeClassifier()

model.fit(X, y)

# New Fruit
new_fruit = [[160, 1]]

prediction = model.predict(new_fruit)

print("Prediction:", prediction[0])