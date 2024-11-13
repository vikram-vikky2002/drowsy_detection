from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("drowsiness_detection_model.h5")

# Define function to prepare image for prediction
def prepare_image(image_path):
    img = load_img(image_path, target_size=(64, 64))  # Resize image to the model's input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image is uploaded
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Save the uploaded image
            image_path = os.path.join("static", "uploaded_image.png")
            file.save(image_path)

            # Prepare the image and make prediction
            img_array = prepare_image(image_path)
            prediction = model.predict(img_array)
            result = "Active (Eyes Open)" if prediction[0][0] > 0.5 else "Drowsy (Eyes Closed)"

            # Render result
            return render_template("index.html", result=result, image_path=image_path)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
