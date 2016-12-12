import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

news = fetch_20newsgroups(subset = "all")
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target)

clf_1 = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', MultinomialNB()),
        ])

clf_2 = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', MultinomialNB()),
        ])

clf_3 = Pipeline([
            ('vect', TfidfVectorizer(token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b")),
            ('clf', MultinomialNB()),
        ])

for i, clf in enumerate([clf_1, clf_2, clf_3]):
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    print("Model {0:d}: Mean score: {1:.3f} (+/-{2:.3f})").format(i, np.mean(scores), sem(scores))
    
