import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model = load_model('trained_model.h5')
test_data_dir = 'allfrog/'
img_height, img_width = 32, 32

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def determine_label(prediction):
    return "bad" if prediction[0][0] > 0.5 else "good"

with open('labels.txt', 'w') as labels_file:
    for img_name in sorted(os.listdir(test_data_dir)):

        img_path = os.path.join(test_data_dir, img_name)
        img_array = load_and_preprocess_image(img_path)
        prediction = model.predict(img_array)

        label = determine_label(prediction)

        labels_file.write(f'{label}\n')
        print(f'Image: {img_name}, Prediction: {prediction[0][0]}, Label: {label}')