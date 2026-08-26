from sklearn.tree import DecisionTreeClassifier

# Training Data
# Features:
# [Weather, Temperature]
# Weather: 1 = Sunny, 0 = Rainy
# Temperature: 1 = Hot, 0 = Cool

X = [
    [1, 1],   # Sunny, Hot
    [1, 0],   # Sunny, Cool
    [0, 1],   # Rainy, Hot
    [0, 0],   # Rainy, Cool
    [1, 1],
    [0, 0]
]

# Output
y = [
    "Play",
    "Play",
    "Don't Play",
    "Don't Play",
    "Play",
    "Don't Play"
]

# Create Decision Tree Model
model = DecisionTreeClassifier()

# Train the Model
model.fit(X, y)

# New Weather
# Sunny = 1
# Rainy = 0
# Cool = 0
new_day = [[0, 1]]

# Predict
prediction = model.predict(new_day)

print("Today's Weather")
print("Sunny :", "Yes")
print("Temperature :", "Cool")
print()

print("Decision :", prediction[0])