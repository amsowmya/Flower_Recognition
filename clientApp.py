from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from com_utils.utils import decodeImage
from predict import flower

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.fileName = 'sample_flowers/Rose.png'
        self.classifier = flower(self.fileName)

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.fileName)
    result = clApp.classifier.predict_flower()
    return jsonify(result)

if __name__ == '__main__':
    clApp = ClientApp()
    app.run(host='127.0.0.1', port=8000, debug=False)

