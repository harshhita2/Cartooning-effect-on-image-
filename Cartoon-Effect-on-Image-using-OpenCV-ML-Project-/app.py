from flask import Flask, request, render_template, send_file, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Cartoon effect functions
def edge_detection(img, line_size=7, blur_value=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, line_size, blur_value
    )
    return edges

def color_quantization(img, k=9):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    return result.reshape(img.shape)

def cartoon_effect(img):
    edges = edge_detection(img)
    quantized = color_quantization(img)
    blurred = cv2.bilateralFilter(quantized, d=7, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Process the image
    img = cv2.imread(filepath)
    cartoon_img = cartoon_effect(img)

    # Save processed image
    output_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(output_path, cartoon_img)

    # Return the processed image
    return send_file(output_path, mimetype='image/jpeg')

@app.route('/show_image/<filename>')
def show_image(filename):
    # Generate the processed image URL dynamically
    image_url = url_for('processed', filename=filename)
    return render_template('index.html', image_url=image_url)

@app.route('/processed/<filename>')
def processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8999)
