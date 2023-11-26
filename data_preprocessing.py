# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load Sentiment140 dataset
df = pd.read_csv('trainin1_data.csv', encoding='latin-1', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# Keep only relevant columns
df = df[['target', 'text']]

# Map target labels to 'positive', 'negative', and 'neutral'
df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})

# Text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Save TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = nb_model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Example prediction for new text

new_text = "i hate you "
new_text_clean = preprocess_text(new_text)
new_text_tfidf = tfidf_vectorizer.transform([new_text_clean])

prediction = nb_model.predict(new_text_tfidf)
print(f"Predicted Sentiment: {prediction}")

# Save the model
joblib.dump(nb_model, 'sentiment_model.joblib')


