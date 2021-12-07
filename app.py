import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)

    output = [pred[0]]

    mostra = 'TESTE'
    if output == [1]:
        mostra = 'O POKEMON É LENDÁRIO!'
    else:
        mostra = 'O POKEMON NÃO É LENDARIO :( '

    return render_template("index.html", prediction_text = output, prediction_result = mostra)

@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    output = [pred[0]]
    return jsonify(output)
