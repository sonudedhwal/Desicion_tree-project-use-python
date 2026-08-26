import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df = pd.DataFrame({
    "rain":[0,1,0],
    "sunday":[0,1,0],
    "Decision":[0,1,2]
})

#FEATURE
X = df[["rain","sunday"]]
#TARGET
Y = df["Decision"]
#creat decision tree
model= DecisionTreeClassifier()

#trean the model
model.fit(X,Y)
RESUT = model.predict([[0,0]])

print(RESUT)

#0 dont pay 
#1 pay cricket
#2  go to work