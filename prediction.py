from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

model = None

input_shape = (224, 224)


def load_model():
    model = tf.keras.models.load_model('resnet50_scoretestitem')
    print("Model loaded")
    return model


def predict(image: np.ndarray):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    class_labels = ['0', '1', '2', '3', '4', '5', '6', 'L']

    pred = np.argmax(model.predict(image), axis=-1)
    print(pred)
    return class_labels[pred[0]]


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image