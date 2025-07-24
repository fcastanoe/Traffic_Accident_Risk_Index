import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# —————————————————————————
# 1) Parámetros y modelos
# —————————————————————————

# --- Clasificador TFLite ---
TFLITE_PATH = "/home/fcastanoe/Downloads/model_f16.tflite"
IMG_SIZE    = (512, 512)
interp_cls = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp_cls.allocate_tensors()
in_cls  = interp_cls.get_input_details()
out_cls = interp_cls.get_output_details()

label_names = [
    'None','Surface_Dry', 'Surface_Unknown', 'Surface_Wet',
    'Weather_Clear', 'Weather_Fog', 'Weather_Rain', 'Weather_Unknown'
]
PREFIXES = ['Surface_', 'Weather_']

# Risk indices
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

# --- Detector TFLite en tiempo real ---
TFLITE_PATH_DET = "/home/fcastanoe/Downloads/best_1_float32.tflite"
MODEL_SIZE_DET  = (512, 512)
CONF_THRESH     = 0.25
IOU_THRESH      = 0.5
interpreter_det = tf.lite.Interpreter(model_path=TFLITE_PATH_DET)
interpreter_det.allocate_tensors()
in_det  = interpreter_det.get_input_details()[0]
out_det = interpreter_det.get_output_details()[0]
from PIL import ImageDraw, ImageFont
FONT = ImageFont.load_default()

# Pesos y límites
W1, W2, W3 = 0.3, 0.4, 0.6
MAX_HOLES   = 20

# —————————————————————————
# 2) Funciones auxiliares adaptadas a array
# —————————————————————————

def preprocess_cls_arr(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = tf.image.resize(rgb, IMG_SIZE) / 255.0
    return np.expand_dims(arr.numpy().astype(np.float32), axis=0)

def run_cls_arr(frame):
    tensor = preprocess_cls_arr(frame)
    interp_cls.set_tensor(in_cls[0]['index'], tensor)
    interp_cls.invoke()
    preds = interp_cls.get_tensor(out_cls[0]['index'])[0]
    results = []
    for p in PREFIXES:
        group = [(lab, float(preds[i])) for i, lab in enumerate(label_names) if lab.startswith(p)]
        results.append(max(group, key=lambda x: x[1]))
    return results

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

def run_det_arr(frame):
    """
    Detección en tiempo real usando TFLite:
    - redimensiona el frame a MODEL_SIZE_DET,
    - normaliza y corre el intérprete,
    - realiza NMS y dibuja rectángulos en el frame redimensionado.
    """
    # 1) convertir y redimensionar
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, MODEL_SIZE_DET)
    inp = np.expand_dims(resized.astype(np.float32)/255.0, axis=0)

    # 2) inferir
    interpreter_det.set_tensor(in_det['index'], inp)
    interpreter_det.invoke()

    # 3) leer salidas y filtrar
    out = interpreter_det.get_tensor(out_det['index'])[0].T  # (N,5)
    mask = out[:,4] > CONF_THRESH
    dets = out[mask]
    if dets.shape[0] == 0:
        return resized, 0

    xc,yc,w,h,confs = dets.T
    x1 = xc - w/2;  y1 = yc - h/2
    x2 = xc + w/2;  y2 = yc + h/2
    boxes  = np.stack([x1,y1,x2,y2], axis=1)
    scores = confs
    keep = nms_numpy(boxes, scores, IOU_THRESH)

    # 4) dibujar sobre la imagen redimensionada
    img_pil = Image.fromarray(resized)
    draw = ImageDraw.Draw(img_pil)
    for i in keep:
        bx1,by1,bx2,by2 = boxes[i]
        x1p = int(max(0, min(1, bx1))*MODEL_SIZE_DET[0])
        y1p = int(max(0, min(1, by1))*MODEL_SIZE_DET[1])
        x2p = int(max(0, min(1, bx2))*MODEL_SIZE_DET[0])
        y2p = int(max(0, min(1, by2))*MODEL_SIZE_DET[1])
        sc = scores[i]
        draw.rectangle([x1p,y1p,x2p,y2p], outline="red", width=2)
        draw.text((x1p, y1p-10), f"{sc:.2f}", fill="red", font=FONT)

    return np.array(img_pil), int(len(keep))

def compute_risk(w_lbl, s_lbl, holes):
    wr = weather_risk.get(w_lbl, 0.0)
    sr = surface_risk.get(s_lbl, 0.0)
    pr = min(holes, MAX_HOLES) / MAX_HOLES
    return W1*wr + W2*sr + W3*pr

def recommendation(score):
    if score >= 0.7: return "ALTO riesgo – manejar con MUCHA precaución"
    if score >= 0.4: return "Riesgo moderado – con precaución"
    return "Bajo riesgo – condiciones razonables"

# —————————————————————————
# 3) Interfaz en tiempo real
# —————————————————————————

window = tk.Tk()
window.title("Riesgo en Tiempo Real")

canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()
txt = tk.StringVar()
info_label = tk.Label(window, textvariable=txt, font=("Arial",12), justify="left")
info_label.pack(pady=5)

cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    # Clasificación y detección
    (s_lbl, s_p), (w_lbl, w_p) = run_cls_arr(frame)
    det_frame, holes      = run_det_arr(frame)

    # Riesgo y recomendación
    score = compute_risk(w_lbl, s_lbl, holes)
    rec   = recommendation(score)

    # Mostrar en Tkinter
    disp = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(disp).resize((640,480))
    imgtk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.imgtk = imgtk

    info = (
        f"Clima: {w_lbl} ({w_p:.2f})\n"
        f"Superficie: {s_lbl} ({s_p:.2f})\n"
        f"Huecos: {holes}\n"
        f"Índice riesgo: {score:.2f} – {rec}\n"
    )
    txt.set(info)

    window.after(50, update_frame)

window.after(0, update_frame)
window.mainloop()
