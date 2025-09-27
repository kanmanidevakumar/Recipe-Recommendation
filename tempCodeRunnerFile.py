from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import base64
import cv2
import requests

app = Flask(__name__)

# ✅ Load Model and Labels
model = tf.keras.models.load_model("trained_model.h5")
with open("class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

# ✅ Replace with your Spoonacular API Key

API_KEY = "67ca4fcc94314553bc0f930349ca9b35"

# ✅ Preprocess base64 image
def preprocess_image(base64_str):
    image_data = base64.b64decode(base64_str.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (150, 150)) / 255.0
    return np.expand_dims(image, axis=0)

# ✅ Fetch Recipe from Spoonacular
def get_recipe_from_api(ingredient):
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ingredient,
        "number": 1,
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data:
        recipe_id = data[0]["id"]
        recipe_url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
        info = requests.get(recipe_url, params={"apiKey": API_KEY})
        return info.json().get("sourceUrl", None)
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # ✅ Preprocess Image
    image_array = preprocess_image(data["image"])
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    predicted_label = class_labels[class_index]

    # ✅ Make sure API understands the ingredient
    ingredient = predicted_label.lower()

    # ✅ Debugging (optional, check backend console)
    print("Raw prediction:", prediction)
    print("Predicted label:", predicted_label)

    # ✅ Get Recipe
    recipe_url = get_recipe_from_api(ingredient)

    return jsonify({
        "ingredient": ingredient,
        "recipe_url": recipe_url if recipe_url else "No recipe found"
    })

if __name__ == "__main__":
    app.run(debug=True)
