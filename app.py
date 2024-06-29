from flask import Flask, request, jsonify

from model import ner


app = Flask(__name__)

@app.route('/')
def home():
    resume= request.files["resume"].read().decode("utf-8")
    entities = ner(resume)

    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True)
