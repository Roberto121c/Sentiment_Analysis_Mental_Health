import pickle

with open('xgboost_model.pkl', 'rb') as f:
    xgb_clf = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('selector.pkl', 'rb') as f:
    selector = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def predict_labels(new_text):
    
    # Transform the new text data
    new_X_tfidf = vectorizer.transform(new_text)
    new_X_selected = selector.transform(new_X_tfidf)

    # Make predictions
    new_y_pred = xgb_clf.predict(new_X_selected)

    # Decode the predicted labels
    new_y_pred_labels = label_encoder.inverse_transform(new_y_pred)

    return new_y_pred_labels

#Anxiety
new_text = ['trouble sleeping, confused mind, restless heart. All out of tune']
predicted_labels = predict_labels(new_text)
print(predicted_labels)


new_text = ['WKWKWKWK my teacher is so cute']
predicted_labels = predict_labels(new_text)
print(predicted_labels)
