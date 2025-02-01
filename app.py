from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model('plant_disease_model.h5')

# Preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

class_labels=[
    "Tomato_healthy",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Target_Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Leaf_Mold",
    "Tomato_Late_blight",
    "Tomato_Early_blight",
    "Tomato_Bacterial_spot",
    "Potato___healthy",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Pepper__bell___healthy",
    "Pepper__bell___Bacterial_spot"
]
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # Fix: Use 'methods' instead of 'method'
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file temporarily
    img_path = 'temp.jpg'
    file.save(img_path)

    # Preprocess the image and make a prediction
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Clean up the temporary file
    os.remove(img_path)

    # Return the result
    return jsonify({
        'predicted_class': class_labels[predicted_class],
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)