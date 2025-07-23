import torch
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time
from torchvision.ops import nms

# --- CARGA DEL MODELO TORCHSCRIPT ---

# Ruta al archivo TorchScript (.torchscript) exportado desde PyTorch
model_path = "C:/Users/fcast/OneDrive - Universidad Nacional de Colombia/UNIVERSIDAD/PDI/MODELS/best_1.torchscript"

# Cargar el modelo TorchScript
# TorchScript es una forma serializada y optimizada para ejecutar modelos PyTorch sin necesidad
# del entorno completo de Python o entrenamiento.
model = torch.jit.load(model_path)
model.eval()  # Poner modelo en modo evaluación (sin gradientes)

# Tamaño de entrada del modelo
INPUT_SIZE = 512
CONF_THRESH = 0.4
IOU_THRESH  = 0.5


def letterbox(image, new_shape=(INPUT_SIZE, INPUT_SIZE), color=(114, 114, 114)):
    shape = image.shape[:2]  # height, width
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, int(dh), int(dh), int(dw), int(dw),
                               cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, dw, dh


def run_inference(image_path):
    # Leer imagen y convertir a RGB
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]

    # Letterbox a INPUT_SIZE
    img, ratio, dw, dh = letterbox(rgb)
    # Normalizar y tensor
    tensor = img.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    tensor = torch.from_numpy(tensor).unsqueeze(0)

    # Inferencia
    start = time.time()
    with torch.no_grad():
        preds = model(tensor)[0]   # [1,5,N] -> preds[0]: (5, N)
    inference_time = time.time() - start

    # Convertir a (N,5)
    preds = preds.permute(1, 0).cpu()  # (N,5): [xc, yc, w, h, conf]
    # Filtrar por conf
    mask = preds[:,4] > CONF_THRESH
    if not mask.any():
        return rgb, rgb, {}, inference_time
    det = preds[mask]

    # Convertir a x1,y1,x2,y2
    xcs, ycs, ws, hs, confs = det.t()
    x1 = xcs - ws/2;  y1 = ycs - hs/2
    x2 = xcs + ws/2;  y2 = ycs + hs/2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = confs

    # NMS
    keep = nms(boxes, scores, IOU_THRESH)
    final = boxes[keep].numpy()
    final_scores = scores[keep].numpy()

    class_counts = {0: len(keep)}  # solo clase 0: pothole

    # Ajustar coordenadas al tamaño original
    img_out = rgb.copy()
    for box, score in zip(final, final_scores):
        x1, y1, x2, y2 = box
        # remap from padded coords to original
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio
        # clamp
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_out, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Guardar detecciones TXT
    txt_path = image_path.rsplit('.',1)[0] + '_detections.txt'
    with open(txt_path, 'w') as f:
        for box, score in zip(final, final_scores):
            x1n, y1n, x2n, y2n = [coord / sz for coord, sz in zip(box, [INPUT_SIZE]*4)]
            f.write(f"{x1n:.6f} {y1n:.6f} {x2n:.6f} {y2n:.6f} {score:.6f} 0\n")

    return rgb, img_out, class_counts, inference_time


def select_image():
    fp = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.jpeg *.png")])
    if not fp:
        return
    orig, proc, counts, inf_time = run_inference(fp)
    # mostrar
    orig_img = ImageTk.PhotoImage(Image.fromarray(orig).resize((320,320)))
    proc_img = ImageTk.PhotoImage(Image.fromarray(proc).resize((320,320)))
    original_label.configure(image=orig_img)
    original_label.image = orig_img
    processed_label.configure(image=proc_img)
    processed_label.image = proc_img
    class_text.set(f"Huecos detectados: {counts.get(0,0)}\nTiempo: {inf_time*1000:.1f} ms")

# --- UI ---
window = tk.Tk()
window.title("Detección Potholes TorchScript")
btn = tk.Button(window, text="Seleccionar imagen", command=select_image)
btn.pack(pady=5)
frame = tk.Frame(window)
frame.pack()
original_label = tk.Label(frame)
original_label.grid(row=0, column=0, padx=10)
processed_label = tk.Label(frame)
processed_label.grid(row=0, column=1, padx=10)
class_text = tk.StringVar()
class_label = tk.Label(frame, textvariable=class_text, font=("Arial", 12), justify="left")
class_label.grid(row=0, column=2, padx=10)

window.mainloop()


