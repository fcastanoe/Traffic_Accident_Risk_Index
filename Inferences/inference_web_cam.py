import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import torch
import cv2
import time
from torchvision.ops import nms
from tensorflow.lite.experimental import load_delegate

# —————————————————————————
# 1) Parámetros y modelos
# —————————————————————————

# --- Clasificador TFLite ---
TFLITE_PATH = "C:/Users/fcast/OneDrive - Universidad Nacional de Colombia/UNIVERSIDAD/PDI/MODELS/model_clean.tflite"
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

# --- Detector TorchScript ---
TS_PATH     = "C:/Users/fcast/OneDrive - Universidad Nacional de Colombia/UNIVERSIDAD/PDI/MODELS/best_1.torchscript"
model_det   = torch.jit.load(TS_PATH).eval()
INPUT_SIZE  = 512
CONF_THRESH = 0.4
IOU_THRESH  = 0.5

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


def letterbox(img):
    h, w = img.shape[:2]
    r = min(INPUT_SIZE/h, INPUT_SIZE/w)
    new_unpad = (int(w*r), int(h*r))
    dw, dh = (INPUT_SIZE-new_unpad[0]) / 2, (INPUT_SIZE-new_unpad[1]) / 2
    resized = cv2.resize(img, new_unpad)
    padded = cv2.copyMakeBorder(resized,
                                int(dh), int(dh),
                                int(dw), int(dw),
                                cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, r, dw, dh


def run_det_arr(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    padded, r, dw, dh = letterbox(rgb)
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        preds = model_det(tensor)[0].permute(1,0).cpu()
    mask = preds[:,4] > CONF_THRESH
    det = preds[mask]
    if det.numel() == 0:
        return frame, 0
    xcs, ycs, ws, hs, confs = det.t()
    x1 = xcs - ws/2; y1 = ycs - hs/2
    x2 = xcs + ws/2; y2 = ycs + hs/2
    boxes = torch.stack([x1, y1, x2, y2], 1)
    keep = nms(boxes, confs, IOU_THRESH)
    count = keep.numel()
    out = rgb.copy()
    for idx in keep:
        b = boxes[idx].numpy()
        x1n = int((b[0] - dw) / r); y1n = int((b[1] - dh) / r)
        x2n = int((b[2] - dw) / r); y2n = int((b[3] - dh) / r)
        cv2.rectangle(out, (x1n, y1n), (x2n, y2n), (0,255,0), 2)
    return out, int(count)


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

    (s_lbl, s_p), (w_lbl, w_p) = run_cls_arr(frame)
    det_frame, holes = run_det_arr(frame)
    score = compute_risk(w_lbl, s_lbl, holes)
    rec = recommendation(score)

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
