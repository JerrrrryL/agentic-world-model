#!/usr/bin/env python3
"""TF-IDF + logistic regression on the SICK sentence pairs.

Nowhere near the 0.905 SOTA (a fine-tuned RoBERTa); the point is a submission that is
clearly better than any constant one, so a reward that moves proves the grader works.
Runs in well under a minute on CPU.
"""

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = load_from_disk("./data/train")
test = load_from_disk("./data/test")

a_tr = [s.lower() for s in train["sentence_A"]]
b_tr = [s.lower() for s in train["sentence_B"]]
a_te = [s.lower() for s in test["sentence_A"]]
b_te = [s.lower() for s in test["sentence_B"]]

word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
word.fit(a_tr + b_tr)


def feats(a, b):
    """Entailment is about the *difference* between the two sentences, so the pairwise
    features (elementwise product and absolute difference) carry most of the signal."""
    A, B = word.transform(a), word.transform(b)
    prod = A.multiply(B)
    diff = abs(A - B)
    return sparse.hstack([A, B, prod, diff]).tocsr()


X = feats(a_tr, b_tr)
y = np.asarray(train["label"])
clf = LogisticRegression(max_iter=2000, C=4.0)
clf.fit(X, y)
print("train accuracy:", clf.score(X, y))

pred = clf.predict(feats(a_te, b_te))
pd.DataFrame({"label": pred}).to_csv("/app/submission.csv", index=False)
print("wrote /app/submission.csv", pred.shape, np.bincount(pred))
