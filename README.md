# Amazon Review Sentiment Classifier

Text classification project built during my Applied AI semester at
Hochschule Rosenheim. Trains a model to predict whether a product review
is positive or negative just from the review text.

## What I did

- Used the Amazon Fine Food Reviews dataset (~568K reviews) from Kaggle
- Dropped 3-star reviews (too ambiguous), labeled 4-5 stars as positive
  and 1-2 stars as negative
- Balanced sample of 30,000 reviews (15K positive, 15K negative) so the
  model doesn't just learn to predict the majority class
- Converted review text to numbers with TF-IDF
- Trained a Logistic Regression classifier

## Results

**87% accuracy** on 6,000 held-out reviews the model never saw during
training, with balanced precision/recall across both classes (~0.87-0.88
for both positive and negative).

The model also picked up on intuitive signals — "great," "perfect,"
"delicious" strongly predict positive reviews, while "disappointed,"
"worst," "waste," and "return" predict negative ones. Makes sense: people
mention wanting their money back or having wasted it when they're
unhappy with a purchase.

## How it works

- `prepare_data.py` — loads the raw CSV, labels sentiment, samples a
  balanced dataset
- `train_model.py` — trains the model and evaluates it (confusion
  matrix saved as PNG)
- `inspect_model.py` — pulls out which words the model weighs most
  heavily for each class

## Tools

Python, pandas, scikit-learn, matplotlib

## Data

[Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## Next steps

- Try a model that handles negation better (e.g. "not good" currently
  might confuse a bag-of-words approach)
- Test on a different domain (electronics reviews) to see if it
  generalizes
- Compare against a simple neural network to see if accuracy improves
