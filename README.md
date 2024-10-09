# Sentiment_Analysis_Mental_Health

This project aim to develop a sentiment analysis model to classify mental health-related statements into different statuses (e.g., anxiety, depression, normal).

The main steps in the project include:

### Data Exploration and Visualization (EDA)

* Analyzed the distribution of mental health statuses in the dataset
* Visualized word clouds for different mental health statuses
* Analyzed the distribution of statement lengths and winsorized the data
  
### Data Preprocessing
* Encoded the mental health status labels
* Split the data into training and test sets
* Resampled the training data to address class imbalance
* Preprocessed the text data by converting to lowercase, removing non-alphanumeric characters, and applying stemming

### Model Training
* Used TF-IDF vectorization to convert the text data into numerical features
* Performed feature selection using SelectKBest and chi-square
* Trained an XGBoost classifier on the selected features
* Evaluated the model performance using classification report

Model Testing
* Loaded the trained XGBoost model, TF-IDF vectorizer, and feature selector
* Implemented a function to predict the mental health status of new text inputs
* Tested the model on sample inputs and printed the predicted labels

The project demonstrates the application of natural language processing and machine learning techniques to address the problem of mental health status classification.
