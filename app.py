from flask import Flask, render_template, request
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
nb_model = joblib.load('sentiment_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')  # Save your TF-IDF vectorizer during training

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        cleaned_input = preprocess_text(user_input)
        input_tfidf = tfidf_vectorizer.transform([cleaned_input])
        prediction = nb_model.predict(input_tfidf)[0]
        return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
