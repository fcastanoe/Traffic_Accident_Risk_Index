import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import time

# —————————————————————————
# 1) Parámetros y modelos
# —————————————————————————

# --- Clasificador TFLite ---
TFLITE_PATH_CLS = "/home/fcastanoe/Downloads/model_f16.tflite"
IMG_SIZE    = (512, 512)
interp_cls  = tf.lite.Interpreter(model_path=TFLITE_PATH_CLS)
interp_cls.allocate_tensors()
in_cls  = interp_cls.get_input_details()
out_cls = interp_cls.get_output_details()

label_names = [
    'None','Surface_Dry', 'Surface_Unknown', 'Surface_Wet',
    'Weather_Clear', 'Weather_Fog', 'Weather_Rain', 'Weather_Unknown'
]
PREFIXES = ['Surface_', 'Weather_']

# Risk indices históricos
weather_risk = {
    "Weather_Rain":   0.99785,
    "Weather_Fog":    0.87875,
    "Weather_Unknown":0.56813,
    "Weather_Clear":  0.30912,
}
surface_risk = {
    "Surface_Wet":     0.94239,
    "Surface_Dry":     0.11007,
    "Surface_Unknown": 0.00529,
}

# --- Detector TFLite de baches ---
TFLITE_PATH_DET = "/home/fcastanoe/Downloads/best_1_float32.tflite"
MODEL_SIZE_DET  = (512, 512)
CONF_THRESH_DET = 0.4
IOU_THRESH_DET  = 0.5
interpreter_det = tf.lite.Interpreter(model_path=TFLITE_PATH_DET)
interpreter_det.allocate_tensors()
in_det  = interpreter_det.get_input_details()[0]
out_det = interpreter_det.get_output_details()[0]
FONT = ImageFont.load_default()

# Pesos para combinación final
W1, W2, W3 = 0.3, 0.4, 0.6
MAX_HOLES   = 20

# —————————————————————————
# 2) Funciones auxiliares
# —————————————————————————

def preprocess_cls(path):
    arr = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    arr = tf.image.resize(arr, IMG_SIZE) / 255.0
    return np.expand_dims(arr.numpy().astype(np.float32), axis=0)

def run_cls(path):
    tensor = preprocess_cls(path)
    interp_cls.set_tensor(in_cls[0]['index'], tensor)
    interp_cls.invoke()
    preds = interp_cls.get_tensor(out_cls[0]['index'])[0]
    results = []
    for p in PREFIXES:
        group = [(lab, float(preds[i])) for i, lab in enumerate(label_names) if lab.startswith(p)]
        results.append(max(group, key=lambda x: x[1]))
    return results  # [(s_lbl,s_p),(w_lbl,w_p)]

def nms_numpy(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter/(areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return keep

def infer_tflite_det(path):
    # Preprocesado
    img = Image.open(path).convert("RGB").resize(MODEL_SIZE_DET)
    arr = np.array(img, dtype=np.float32)/255.0
    inp = np.expand_dims(arr, axis=0)
    interpreter_det.set_tensor(in_det['index'], inp)
    t0 = time.time()
    interpreter_det.invoke()
    t_inf = time.time() - t0

    out = interpreter_det.get_tensor(out_det['index'])[0].T  # (N,5)
    mask = out[:,4] > CONF_THRESH_DET
    dets = out[mask]
    if dets.shape[0]==0:
        return img, 0, t_inf

    xc,yc,w,h,conf = dets.T
    x1 = xc - w/2;  y1 = yc - h/2
    x2 = xc + w/2;  y2 = yc + h/2
    boxes  = np.stack([x1,y1,x2,y2],axis=1)
    scores = conf
    keep = nms_numpy(boxes, scores, IOU_THRESH_DET)
    # Dibujar y contar
    draw = ImageDraw.Draw(img)
    for i in keep:
        bx1,by1,bx2,by2 = boxes[i]
        # clamp
        x1p = int(max(0, min(1,bx1))*MODEL_SIZE_DET[0])
        y1p = int(max(0, min(1,by1))*MODEL_SIZE_DET[1])
        x2p = int(max(0, min(1,bx2))*MODEL_SIZE_DET[0])
        y2p = int(max(0, min(1,by2))*MODEL_SIZE_DET[1])
        sc = scores[i]
        draw.rectangle([x1p,y1p,x2p,y2p], outline="red", width=2)
        draw.text((x1p, y1p-10), f"{sc:.2f}", fill="red", font=FONT)
    return img, len(keep), t_inf

def compute_risk(w_lbl, s_lbl, holes):
    wr = weather_risk.get(w_lbl, 0.0)
    sr = surface_risk.get(s_lbl, 0.0)
    pr = min(holes, MAX_HOLES)/MAX_HOLES
    return W1*wr + W2*sr + W3*pr

def recommendation(score):
    if score >= 0.7: return "ALTO riesgo – manejar con MUCHA precaución"
    if score >= 0.4: return "Riesgo moderado – con precaución"
    return "Bajo riesgo – condiciones razonables"

# —————————————————————————
# 3) Interfaz Tkinter
# —————————————————————————

window = tk.Tk()
window.title("Análisis de Riesgo de Carretera")

def on_select():
    path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.png")])
    if not path: return

    # Clasificación
    (s_lbl,s_p),(w_lbl,w_p) = run_cls(path)

    # Detección TFLite de baches
    det_img, holes, det_time = infer_tflite_det(path)

    # Cálculo de riesgo
    score = compute_risk(w_lbl, s_lbl, holes)
    rec   = recommendation(score)

    # Mostrar original y detección
    img = Image.open(path).convert("RGB").resize((300,300))
    det_img_tk = det_img.resize((300,300))
    tk_orig = ImageTk.PhotoImage(img)
    tk_det  = ImageTk.PhotoImage(det_img_tk)
    lbl_orig.configure(image=tk_orig); lbl_orig.image = tk_orig
    lbl_det.configure(image=tk_det);   lbl_det.image = tk_det

    # Mensajes por condición
    soil_msgs = {
        "Surface_Wet":     "Reducir velocidad: el suelo está mojado.",
        "Surface_Dry":     "Superficie seca: condiciones normales.",
        "Surface_Unknown": "Estado de la superficie desconocido: extrema precaución."
    }
    weather_msgs = {
        "Weather_Rain":    "Lluvia: mantener distancia de seguridad y usar limpiaparabrisas.",
        "Weather_Fog":     "Niebla: encender luces antiniebla y reducir velocidad.",
        "Weather_Clear":   "Cielo despejado: condiciones buenas.",
        "Weather_Unknown": "Clima desconocido: conducir con precaución."
    }
    hole_msgs = {
        range(0, 1):   "Sin huecos detectados.",
        range(1, 6):   "Algunos huecos: reducir velocidad ligeramente.",
        range(6, 21):  "Muchos huecos: máxima precaución y velocidad muy baja."
    }

    msg_soil    = soil_msgs.get(s_lbl, "")
    msg_weather = weather_msgs.get(w_lbl, "")
    msg_holes   = next(v for k, v in hole_msgs.items() if holes in k)

    info = (
        "PREDICCIONES:\n"
        f"  • Clima: {w_lbl} ({w_p:.2f})\n"
        f"  • Condición de carretera: {s_lbl} ({s_p:.2f})\n"
        f"  • Huecos: {holes}\n\n"
        "RIESGO:\n"
        f"  • Índice de riesgo: {score:.2f} → {rec}\n\n"
        "RECOMENDACIONES ADICIONALES:\n"
        f"  • {msg_soil}\n"
        f"  • {msg_weather}\n"
        f"  • {msg_holes}"
    )
    txt.set(info)

btn = tk.Button(window, text="Seleccionar imagen", command=on_select)
btn.pack(pady=5)

frame = tk.Frame(window)
frame.pack()
lbl_orig = tk.Label(frame); lbl_orig.grid(row=0, column=0, padx=10)
lbl_det  = tk.Label(frame); lbl_det .grid(row=0, column=1, padx=10)

txt = tk.StringVar()
tk.Label(window, textvariable=txt, font=("Arial",12), justify="left").pack(pady=10)

window.mainloop()
