import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import torch
import cv2
import time
from torchvision.ops import nms

# —————————————————————————
# 1) Parámetros y modelos
# —————————————————————————

# --- Clasificador TFLite ---
TFLITE_PATH = "C:/Users/fcast/OneDrive - Universidad Nacional de Colombia/UNIVERSIDAD/PDI/MODELS/model_clean.tflite"
IMG_SIZE    = (512, 512)
interp_cls, in_cls, out_cls = tf.lite.Interpreter(model_path=TFLITE_PATH), None, None
interp_cls.allocate_tensors()
in_cls  = interp_cls.get_input_details()
out_cls = interp_cls.get_output_details()

label_names = [
    'None','Surface_Dry', 'Surface_Unknown', 'Surface_Wet',
    'Weather_Clear', 'Weather_Fog', 'Weather_Rain', 'Weather_Unknown'
]
PREFIXES = ['Surface_', 'Weather_']

# Risk indices de tu análisis histórico
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

# Pesos para la combinación final
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
    # elegir mejor Surface_ y Weather_
    results = []
    for p in PREFIXES:
        group = [(lab, float(preds[i])) for i, lab in enumerate(label_names) if lab.startswith(p)]
        results.append(max(group, key=lambda x: x[1]))
    return results

def letterbox(img):
    h, w = img.shape[:2]
    r = min(INPUT_SIZE/h, INPUT_SIZE/w)
    new_unpad = (int(w*r), int(h*r))
    dw, dh = (INPUT_SIZE-new_unpad[0])/2, (INPUT_SIZE-new_unpad[1])/2
    resized = cv2.resize(img, new_unpad)
    padded = cv2.copyMakeBorder(resized,
                                int(dh), int(dh),
                                int(dw), int(dw),
                                cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, r, dw, dh

def run_det(path):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded, r, dw, dh = letterbox(rgb)
    tensor = torch.from_numpy(padded.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        preds = model_det(tensor)[0].permute(1,0).cpu()
    mask = preds[:,4] > CONF_THRESH
    det = preds[mask]
    if det.numel()==0:
        return rgb, 0
    # decode + NMS
    xcs,ycs,ws,hs,confs = det.t()
    x1 = xcs-ws/2; y1=ycs-hs/2; x2=xcs+ws/2; y2=ycs+hs/2
    boxes = torch.stack([x1,y1,x2,y2],1)
    keep = nms(boxes, confs, IOU_THRESH)
    count = keep.numel()
    # dibujar
    out = rgb.copy()
    for idx in keep:
        b = boxes[idx].numpy()
        x1n = int((b[0]-dw)/r); y1n = int((b[1]-dh)/r)
        x2n = int((b[2]-dw)/r); y2n = int((b[3]-dh)/r)
        cv2.rectangle(out,(x1n,y1n),(x2n,y2n),(0,255,0),2)
    return out, int(count)

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

    # Detección
    orig_det, holes = run_det(path)

    # Cálculo de riesgo
    score = compute_risk(w_lbl, s_lbl, holes)
    rec   = recommendation(score)

    # Mostrar original y detección
    img = Image.open(path).convert("RGB").resize((300,300))
    det_img = Image.fromarray(orig_det).resize((300,300))
    tk_orig = ImageTk.PhotoImage(img);    tk_det = ImageTk.PhotoImage(det_img)
    lbl_orig.configure(image=tk_orig);    lbl_orig.image = tk_orig
    lbl_det.configure(image=tk_det);      lbl_det.image = tk_det
    
    # Mensajes específicos por condición
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
        # si hay muchos potholes
        range(0, 1):   "Sin huecos detectados.",
        range(1, 6):   "Algunos huecos: reducir velocidad ligeramente.",
        range(6, 21):  "Muchos huecos: máxima precaución y velocidad muy baja."
    }

    # Obtener mensaje de superficie
    msg_soil = soil_msgs.get(s_lbl, "")
    # Mensaje de clima
    msg_weather = weather_msgs.get(w_lbl, "")
    # Mensaje de huecos
    msg_holes = next(v for k, v in hole_msgs.items() if holes in k)

    # Texto de resultados
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

lbl_orig = tk.Label(frame)
lbl_orig.grid(row=0, column=0, padx=10)
lbl_det  = tk.Label(frame)
lbl_det.grid(row=0, column=1, padx=10)

txt = tk.StringVar()
tk.Label(window, textvariable=txt, font=("Arial",12), justify="left").pack(pady=10)

window.mainloop()