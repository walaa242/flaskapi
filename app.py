from flask import Flask, request
from keras.models import load_model
import io
import numpy as np
import tensorflow as tf
from PIL import Image


model = tf.keras.models.load_model('CNN_Cancer.h5')
def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    return 1 if model.predict(img)[0][0] > 0.5 else 0

result='prediction'

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def infer_image():
    # Catch the image file from a POST request
    if request.method == 'POST':
        file = request.files['my_image']

    if not file:
        result='there is no image uploaded'

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)
    
    # return result
    prediction = predict_result(img)
    if prediction == 1 :
        result='Malignant'
    else:
        result='Normal'
    return result

if __name__ =='__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)