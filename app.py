# from flask import Flask, render_template, request
# import numpy as np
# import tensorflow as tf
# import cv2

# app = Flask(__name__)

# # Load trained model
# model = tf.keras.models.load_model("model/sign_model.h5")

# # Labels
# labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():

#     file = request.files["image"]

#     img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

#     img = cv2.resize(img,(64,64))

#     img = img/255.0

#     img = np.reshape(img,(1,64,64,3))

#     prediction = model.predict(img)

#     index = np.argmax(prediction)

#     result = labels[index]

#     return {"prediction": result}

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Load trained model with error handling
try:
    model = tf.keras.models.load_model("model/sign_model_final.h5")
    print("Model loaded successfully!")
except:
    model = tf.keras.models.load_model("model/sign_model.h5")
    print("Loaded original model (consider retraining for better accuracy)")

# Labels - update if you have 29 classes
# If you have special characters, modify this list accordingly
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL", "NOTHING"]  # Example for 29 classes

def preprocess_image(image_bytes):
    """
    Enhanced image preprocessing for better prediction
    """
    # Decode image
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing for better feature extraction
    # 1. Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    # 2. Apply slight Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Resize to model input size
    img = cv2.resize(img, (128, 128))  # Match your training size
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if image was uploaded
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400
        
        # Read and preprocess image
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_preds = [(labels[i], float(predictions[0][i])) for i in top_3_idx]
        
        # Get best prediction
        best_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][best_idx])
        
        # Return prediction with confidence
        return jsonify({
            "prediction": labels[best_idx],
            "confidence": f"{confidence*100:.2f}%",
            "top_3_predictions": [
                {"label": label, "confidence": f"{conf*100:.2f}%"} 
                for label, conf in top_3_preds
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
