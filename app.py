from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
        review_text=str(request.form.get('review'))
        cv = joblib.load("model/cv.pkl")
        num = cv.transform([review_text]).toarray()
        model=joblib.load("model/classifier1.pkl")
        prediction=model.predict(num)
        return render_template("output.html", prediction=prediction)


if __name__ =="__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')