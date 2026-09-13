import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("reviews_prepared.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"], df["sentiment"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_positive = sorted(zip(coefs, feature_names), reverse=True)[:15]
top_negative = sorted(zip(coefs, feature_names))[:15]

print("Top words signaling POSITIVE sentiment:")
for coef, word in top_positive:
    print(f"  {word:20s} {coef:.2f}")

print("\nTop words signaling NEGATIVE sentiment:")
for coef, word in top_negative:
    print(f"  {word:20s} {coef:.2f}")
