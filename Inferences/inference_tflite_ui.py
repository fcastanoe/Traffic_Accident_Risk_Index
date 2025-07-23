import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import time

def load_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input details:", input_details)
    return interpreter, input_details, output_details

def preprocess_image(path, size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size) / 255.0
    return np.expand_dims(img.numpy().astype(np.float32), axis=0)

def infer_tflite(interpreter, input_details, output_details, img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# —————————————————————————
# 2- Configuración de modelo y etiquetas
# —————————————————————————

TFLITE_PATH = "C:/Users/fcast/OneDrive - Universidad Nacional de Colombia/UNIVERSIDAD/PDI/MODELS/model_clean.tflite"
IMG_SIZE    = (512, 512)

label_names = [
    'NONE', 'Surface_Dry', 'Surface_Unknown', 'Surface_Wet',
    'Weather_Clear', 'Weather_Fog', 'Weather_Rain', 'Weather_Unknown'
]
PREFIXES = ['Surface_', 'Weather_']

interp, in_det, out_det = load_tflite_model(TFLITE_PATH)

# —————————————————————————
# 3- GUI 
# —————————————————————————

window = tk.Tk()
window.title("Clasificación TFLite – Weather y Surface")

def select_image():
    # Selección
    path = filedialog.askopenfilename(
        filetypes=[("Imagen", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return

    # Mostrar preview
    img_disp = Image.open(path).convert("RGB")
    tk_img = ImageTk.PhotoImage(img_disp.resize((300, 300)))
    image_label.configure(image=tk_img)
    image_label.image = tk_img

    # Inference estilo Kaggle
    img_tensor = preprocess_image(path, IMG_SIZE)
    start = time.time()
    preds = infer_tflite(interp, in_det, out_det, img_tensor)
    elapsed = time.time() - start

    # Debug: imprime vector completo
    print("\nPreds vector completo:")
    for lab, p in zip(label_names, preds):
        print(f"  {lab}: {p:.4f}")

    # Post‑procesado por prefijos
    results = []
    for prefix in PREFIXES:
        group = [(lab, float(p)) for lab, p in zip(label_names, preds) if lab.startswith(prefix)]
        best = max(group, key=lambda x: x[1])
        results.append(best)

    # Mostrar resultados
    text = (
        f"{results[0][0]}: {results[0][1]:.2f}\n"
        f"{results[1][0]}: {results[1][1]:.2f}\n"
        f"Tiempo: {elapsed*1000:.1f} ms"
    )
    result_text.set(text)

# Widgets
btn = tk.Button(window, text="Seleccionar imagen", command=select_image)
btn.pack(pady=5)

image_label = tk.Label(window)
image_label.pack(pady=5)

result_text = tk.StringVar()
result_label = tk.Label(window, textvariable=result_text, font=("Arial", 12), justify="left")
result_label.pack(pady=5)

window.mainloop()