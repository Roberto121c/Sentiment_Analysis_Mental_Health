# Import necessary libraries
import re
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
FILE_PATH = '/Users/roancagu/Desktop/Python/NLP_project/Sentiment_Analysis_Mental_Health/sentiments_data.csv'
RANDOM_STATE = 42
TRAIN_SPLIT = 0.8

def shuffle_and_split(df, train_frac=TRAIN_SPLIT):
    df_shuffled = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train_frac)
    train_df = df_shuffled.iloc[:split_idx].reset_index(drop=True)
    test_df = df_shuffled.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df

def resample_data(df, target='status'):
    max_count = df[target].value_counts().max()
    resampled = pd.concat([
        resample(group, replace=True, n_samples=max_count, random_state=RANDOM_STATE)
        if len(group) < max_count else group
        for _, group in df.groupby(target)
    ]).reset_index(drop=True)
    return resampled

def preprocess_text(text, stemmer):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Load data
df = pd.read_csv(FILE_PATH)

# Filter and encode labels
df = df[~df['status'].isin(['Stress', 'Personality disorder'])][['status', 'statement']].reset_index(drop=True)
label_encoder = LabelEncoder()
df['encoded_status'] = label_encoder.fit_transform(df['status'])

print("Classes:", label_encoder.classes_)
print("Encoded classes:", label_encoder.transform(label_encoder.classes_))

# Split data
train_df, test_df = shuffle_and_split(df)

# Resample training data
train_df = resample_data(train_df)

# Preprocess text
stemmer = PorterStemmer()
train_df['statement'] = train_df['statement'].apply(lambda x: preprocess_text(x, stemmer))
test_df['statement'] = test_df['statement'].apply(lambda x: preprocess_text(x, stemmer))

# Prepare features and labels
X_train = train_df['statement'].tolist()
y_train = train_df['encoded_status']
X_test = test_df['statement'].tolist()
y_test = test_df['encoded_status']

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
