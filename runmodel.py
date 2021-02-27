import numpy as np
import onnxruntime as ort
from PIL import Image

classes = ['Bird', 'Flower', 'Hand', 'House', 'Pencil', 'Spectacles', 'Spoon', 'Sun', 'Tree', 'Umbrella']

ort_session = ort.InferenceSession('savedModel/model_final.onnx')  # load the saved onnx model

imgpath = 'data2/Pencil/image/Pencil_1609258655396.png'
path2 = 'data2/Umbrella/image/Umbrella_1609258441186.png'
path3 = 'data/Umbrella/image/Umbrella_1612969775350.png'
path4 = 'data/Tree/image/Tree_1612970057695.png'
path5 = 'data2/Spoon/image/Spoon_1609258989982.png'
path6 = 'data2/House/image/House_1609259235083.png'
path7 = 'data/Spectacles/image/Spectacles_1612958478329.png'
path8 = 'data/Bird/image/Bird_1613020665245.png'
path9 = 'data2/Pencil/image/Pencil_1609258676164.png'
path0 = 'data2/Sun/image/Sun_1609257615331.png'
path10 = 'data2/Flower/image/Flower_1609258374874.png'
path11 = 'data/Flower/image/Flower_1613021533630.png'
path12 = 'data/Hand/image/Hand_1612968751876.png'
path13 = 'data/House/image/House_1613047285747.png'


# pre processing
def process(path):
    # pre process same as training
    image = Image.open(path)
    arr = np.asarray(image)
    image = Image.fromarray(arr[:, :, 3])  # read alpha channel
    image = image.resize((32, 32))
    image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]

    return image[None]


# tes the model
def test(path):
    image = process(path)
    output = ort_session.run(None, {'data': image})[0].argmax()

    print(classes[output], output)

    return classes[output]


if __name__ == '__main__':
    test(path13)
