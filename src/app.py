from flask import Flask, request, render_template
import joblib
import re
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

download("wordnet")
download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r'[^a-z ]', " ", text)
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\^[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\s+', " ", text.lower())
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    return text.split()

def lemmatize_text(words):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():

    input_text = request.form.get('input_text')

    processed_text = preprocess_text(input_text)
    tokens = lemmatize_text(processed_text)
    tokens_joined = " ".join(tokens)

    X = vectorizer.transform([tokens_joined]).toarray()

    prediction = model.predict(X)

    if prediction[0] == 0:
        sentiment = "Negative sentiment"
    else:
        sentiment = "Positive sentiment"

    # Mostrar el resultado
    return render_template('index.html', input_text=input_text, prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
