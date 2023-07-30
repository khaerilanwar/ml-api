# define function predict
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, make_response
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import os

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/coba")
def coba():
    data = {
        "pesan": "API sudah bisa"
    }

    return jsonify(data)


@app.route("/kampus", methods=['POST'])
def kampus():
    data = {
        'nama': 'UBSI Kota Tegal',
        'prodi': 'Sistem Informasi',
        'beasiswa': 'KIP Kuliah'
    }
    data['alamat'] = request.form['alamat']

    return jsonify(data)


@app.route("/latihan", methods=['GET', 'POST'])
def latihan():
    if request.method == 'POST':
        data = {
            "gambar": request.form["gambar"],
            "test post": "code sudah berubah"
        }

        return jsonify(data)


def b64_to_img(image_b64, img_resize):
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    image_re = image.resize(img_resize)

    return image_re


@app.route("/rps", methods=['GET', 'POST'])
def rps_predict():
    if request.method == 'POST':
        model = load_model('rps_model.h5')
        # image = b64_to_img(request.form['gambar'], (150, 150))
        image_data = base64.b64decode(request.form['gambar'])
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((150, 150))
        x = img_to_array(image)
        x /= 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=32)
        result = np.argmax(classes, axis=1)
        print(classes)
        print(result)

        if result[0] == 0:
            prediction = {'label': 'ROCK'}
            return jsonify(prediction)

        elif result[0] == 1:
            # print('ROCK')
            prediction = {'label': 'PAPER'}
            return jsonify(prediction)

        else:
            # print('SCISSORS')
            prediction = {'label': 'SCISSORS'}
            return jsonify(prediction)


@app.route("/gender", methods=['POST'])
def gender_predict():
    if request.method == 'POST':
        model = load_model('1689656509.h5')
        image_re = b64_to_img(request.form["gambar"], (64, 64))

        x = img_to_array(image_re)
        x /= 255
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=1)
        result = np.argmax(classes, axis=1)

        if classes[0] > 0.5:
            # print("This is a male")
            prediction = {
                "gender": "Laki - laki"
            }
        else:
            # print( "This is a female")
            prediction = {
                "gender": "Perempuan"
            }

        return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)
