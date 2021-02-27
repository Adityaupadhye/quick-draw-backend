import os
from flask import Flask, request
import json
from flask_cors import CORS
import base64
import runmodel
from datetime import datetime as dt

# initialize flask app
app = Flask(__name__)
cors = CORS(app)
datasetPath = 'data'


def findtime():
    now = dt.now()
    return now.strftime('%d%m%Y_%H%M%S')


@app.route('/api/test')
def test_fun():
    return "this is test function"


@app.route('/api/result', methods=['POST'])
def result():
    data = json.loads(request.data.decode('utf-8'))
    print(type(data))
    image = data['url'].split(',')[1].encode('utf-8')
    # print('got this url= ', imgurl)

    os.makedirs('userdata', exist_ok=True)
    filename = findtime()+'.png'

    with open(f'userdata/{filename}', 'wb') as f:
        f.write(base64.decodebytes(image))

    output = runmodel.test(f'userdata/{filename}')
    print(output)

    return 'got image url ' + output


@app.route('/upload_canvas', methods=['POST'])
def upload_canvas():
    data = json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    fileName = data['filename']
    className = data['class_name']
    os.makedirs(f'{datasetPath}/{className}/image', exist_ok=True)
    with open(f'{datasetPath}/{className}/image/{fileName}', 'wb') as fh:
        fh.write(base64.decodebytes(image_data))

    return "got the image"
