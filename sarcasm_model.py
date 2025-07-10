import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("sarcastic_dataset.csv")

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['label'].apply(lambda x: 1 if x == 'sarcastic' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("sarcasm_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
