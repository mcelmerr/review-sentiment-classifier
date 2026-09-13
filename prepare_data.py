import pandas as pd

df = pd.read_csv("Reviews.csv")

df = df[df["Score"] != 3]

df["sentiment"] = df["Score"].apply(lambda s: "positive" if s >= 4 else "negative")

df = df[["Text", "sentiment"]].dropna()

df_pos = df[df["sentiment"] == "positive"].sample(15000, random_state=42)
df_neg = df[df["sentiment"] == "negative"].sample(15000, random_state=42)

df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)

df_balanced.to_csv("reviews_prepared.csv", index=False)

print(df_balanced["sentiment"].value_counts())
print(f"\nSaved {len(df_balanced)} rows to reviews_prepared.csv")
