from flask import Flask, render_template, request
import os
from tester_exe import convert

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index_nav2.html')
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Extract image data from the JSON request
        image_data = request.json.get('image_data')
        
        # Check if image_data exists and process it
        if image_data:
            processed_text = convert(image_data)  # Perform your OCR processing here
            return processed_text
        else:
            return "No image data found in the request", 400
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
   app.run()


