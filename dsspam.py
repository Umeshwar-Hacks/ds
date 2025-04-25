import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

df = pd.read_csv("D:/archive (3) (1)/spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['labels', 'text']

df['labels'] = df["labels"].map({"ham": 0, "spam": 1})

tfidf = TfidfVectorizer(stop_words="english")
df['text'] = df["text"].apply(clean_text)
x = tfidf.fit_transform(df["text"])
y = df["labels"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = MultinomialNB()
model.fit(x_train, y_train)
prediction = model.predict(x_test)

print(classification_report(y_test, prediction))

message = "Congradulations! You have won $10000"
message = clean_text(message)

X = tfidf.transform([message])
Y = model.predict(X)

print(Y[0])
