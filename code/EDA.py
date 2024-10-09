# Import necessary libraries
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from sklearn.utils import resample
from scipy.stats import mstats
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

# Create a function to generate and display a word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

file_path = 'sentiments_data.csv'

# Load data
df = pd.read_csv(file_path)

print("Missing Values:")
print(df.isnull().sum())

print('--------------')
print(df['status'].value_counts(normalize=True))

# Group by status and get a random statement from each group
random_statements = df.groupby('status')['statement'].apply(lambda x: x.sample(n=1).iloc[0])

# Print the results
print('--------------')
for status, statement in random_statements.items():
    print(f"Status: {status}")
    print(f"Statement: {statement}\n")
    print("-" * 10)
    
# Histogram for status
print('--------------')
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='status')
plt.title('Distribution of Mental Health Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

print('--------------')
# Winsorize the data at the 5th and 95th percentiles
df['nb_words'] = df['statement'].apply(lambda x: len(str(x).split()))
df["nb_chars"] = df["statement"].apply(lambda x: len(x))

df['statement_length_winsorized'] = mstats.winsorize(df['nb_words'], limits=[0.05, 0.05])

# Plot the winsorized data
df['statement_length_winsorized'].hist(bins=100)
plt.title('Winsorized Distribution of Statement Lengths')
plt.xlabel('Length of Statements')
plt.ylabel('Frequency')
plt.show()

# Generate word clouds for each status
print('--------------')
statuses = df['status'].unique()

for status in statuses:
    status_text = ' '.join(df[df['status'] == status]['statement'])
    generate_word_cloud(status_text, title=f'Word Cloud for {status} state')
