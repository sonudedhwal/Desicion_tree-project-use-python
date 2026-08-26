from sklearn.tree import DecisionTreeClassifier

# Training data
# Features:
# [Contains "Free", Contains "Win", Number of Links]
X = [
    [1, 1, 5],   # Spam
    [1, 0, 3],   # Spam
    [0, 1, 4],   # Spam
    [0, 0, 1],   # Not Spam
    [0, 0, 0],   # Not Spam
    [1, 1, 6],   # Spam
    [0, 0, 2],   # Not Spam
    [1, 0, 4]    # Spam
]

# Target labels
y = [
    "Spam",
    "Spam",
    "Spam",
    "Not Spam",
    "Not Spam",
    "Spam",
    "Not Spam",
    "Spam"
]

# Create Decision Tree model
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# New email to test
# Email contains:
# "Free" = Yes (1)
# "Win" = Yes (1)
# Number of Links = 4
new_email = [[1, 1, 4]]

# Predict
prediction = model.predict(new_email)

print("Email Features:")
print("Contains 'Free': Yes")
print("Contains 'Win': Yes")
print("Number of Links:", 4)
print("\nPrediction:", prediction[0])