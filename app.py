from flask import Flask, request, jsonify, render_template
import pickle
from model import SentimentPrediction

app = Flask(__name__)
sentiment = SentimentPrediction()
sentiment.load('./model')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/review', methods = ['POST'])
def review():
    review = [request.form['review']]

    prediction = sentiment.model.predict(sentiment.vect.transform(review))
    #ret = '{"prediction":' + str(float(prediction)) + '}'   

    return render_template('index.html',prediction = prediction, review = review[0])



if __name__ == "__main__":
    app.run(debug=True)