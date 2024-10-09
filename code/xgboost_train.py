# Import necessary libraries
import re
import pandas as pd
import pickle
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import X_train, X_test, y_train, y_test, label_encoder

RANDOM_STATE = 42
TRAIN_SPLIT = 0.8
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
SELECT_K = 1500
LEARNING_RATE = 0.1
MAX_DEPTH = 7
N_ESTIMATORS = 500

# Vectorize text
vectorizer = TfidfVectorizer(lowercase=True, max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Feature selection
selector = SelectKBest(chi2, k=SELECT_K)
X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_selected = selector.transform(X_test_tfidf)


# Train XGBoost classifier
xgb_clf = XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE
)
xgb_clf.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred = xgb_clf.predict(X_test_selected)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
    
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)
