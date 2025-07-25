from flask import Flask, request, jsonify, render_template
from inference import analyze_image
import io, base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    img_file = request.files.get("image")
    if not img_file:
        return jsonify({"error": "No se subi� ninguna imagen"}), 400

    data, img_bytes = analyze_image(img_file.read())

    # Convertimos la imagen procesada a Base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    # Preparamos data para enviar:
    data["annotated_image"] = f"data:image/png;base64,{img_b64}"

    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
