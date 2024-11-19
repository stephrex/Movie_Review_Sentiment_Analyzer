from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import subprocess
import sys
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

app = Flask(__name__)
CORS(app)

# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


class Predict():
    def __init__(self, text):
        self.text = text

    class Preprocess():
        def __init__(self, text):
            self.text = text

        def remove_punctuations(self):
            self.text = re.sub(r'[^\w\s]', '', self.text)
            return self

        def remove_stopwords(self):
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            self.text = ' '.join(
                [word for word in self.text.split() if word.lower() not in stop_words])
            return self

        def remove_tags_and_urls(self):
            self.text = re.sub(r'<[^>]+>.*?</[^>]+>', '', self.text)
            self.text = re.sub(r'http[s]?://\S+|www\.\S+', '', self.text)
            return self

        def lemmatization(self):
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer()
            self.text = ' '.join([lemmatizer.lemmatize(word)
                                 for word in self.text.split()])
            return self

        def preprocess(self):
            self.remove_punctuations().remove_stopwords().remove_tags_and_urls().lemmatization()
            return self.text

    def predict(self, model):
        '''
        This function preprocesses the text, then makes predictions using the model.
        '''
        self.text = self.Preprocess(self.text).preprocess()
        print(f'Preprocessed Text: {self.text}')
        y_pred = tf.cast(tf.round(tf.squeeze(model.predict(
            tf.cast([self.text], dtype=tf.string)))), dtype=tf.int32)
        classes = ['Negative', 'Postive']
        y_pred_label = classes[y_pred]
        return y_pred_label


@app.route('/')
def home():
    return render_template('app.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['content']
    model = tf.keras.models.load_model('models/model_1.keras')
    prediction = Predict(text).predict(model)
    print(f'Prediction:{prediction}')
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
