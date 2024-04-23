import cv2
import numpy as np
import base64

"""just as tester py file to simulate the trained application, take a image base64 convert it to np"""
def convert(image_data):
    # Convert base64 image data to OpenCV image
    decoded_data = base64.b64decode(image_data)
    np_data = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # Perform processing using OpenCV
    # Example: Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform OCR or any other processing as required
    processed_text = "Text extracted from image: ABC123"

    return processed_text
