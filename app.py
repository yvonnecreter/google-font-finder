# app.py (Backend - Flask)
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import io
import cv2
import torch
from torchvision import transforms

app = Flask(__name__, static_folder='static')

# Initialize font recognition model (placeholder)
# In practice, use a pre-trained model or train your own
font_model = torch.jit.load('font_model.pt') if os.path.exists('font_model.pt') else None

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get and process image
    file = request.files['image']
    img = Image.open(file.stream)
    
    # Preprocess image
    processed = preprocess_image(img)
    characters = segment_characters(processed)
    
    # Predict font (mock implementation)
    predictions = predict_font(characters)
    
    return jsonify(predictions[:5])

def preprocess_image(img):
    # Convert to grayscale
    img = ImageOps.grayscale(img)
    
    # Invert if needed
    if np.mean(img) > 127:
        img = ImageOps.invert(img)
    
    # Autocrop
    img = autocrop(img)
    
    return img

def autocrop(img):
    # Implement autocrop using OpenCV
    cv_img = np.array(img)
    mask = cv_img < 127
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img.crop((x0, y0, x1, y1))

def segment_characters(img):
    # Use OpenCV contour detection
    cv_img = np.array(img)
    contours, _ = cv2.findContours(cv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted([(x, y, w, h) for (x,y,w,h) in [cv2.boundingRect(c) for c in contours]], key=lambda c: c[0])

def predict_font(characters):
    # Mock prediction - implement actual model inference
    return [{'font': 'Roboto', 'probability': 0.95},
            {'font': 'Open Sans', 'probability': 0.87}]

if __name__ == '__main__':
    app.run(debug=True)