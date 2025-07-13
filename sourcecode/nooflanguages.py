# Language Detection
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
print(data.head())

# Check for missing values
data.isnull().sum()

# Show language counts
print("\nLanguage Counts:")
print(data["language"].value_counts())

# Print number of unique languages and the list of them
num_languages = data["language"].nunique()
languages = data["language"].unique()
print("\nNumber of Languages in the Dataset:", num_languages)
print("Languages:")
for lang in languages:
    print("-", lang)

# Preprocessing
x = np.array(data["Text"])
y = np.array(data["language"])
cv = CountVectorizer()
X = cv.fit_transform(x)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Training the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Model accuracy
print("\nModel Accuracy:", model.score(X_test, y_test))

# User input
user = input("\nEnter a Text: ")
data_input = cv.transform([user]).toarray()
output = model.predict(data_input)
print("Detected Language:", output[0])
