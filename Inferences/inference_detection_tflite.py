import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import time


TFLITE_PATH = "/home/fcastanoe/Downloads/best_1_float32.tflite"
MODEL_SIZE = (512, 512)    # tamano (H, W) que espera tu TFLite
CONF_THRESH = 0.25
IOU_THRESH  = 0.5

# Carga TFLite
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
in_details  = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]

# Fuente para dibujar texto
FONT = ImageFont.load_default()


def preprocess_image(path, size):
    """Lee imagen, convierte a RGB, redimensiona y normaliza."""
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img  # devolvemos tambien el PIL original redimensionado

def nms_numpy(boxes, scores, iou_thr):
    """NMS en NumPy. boxes=(N,4), scores=(N,)"""
    x1, y1, x2, y2 = boxes.T
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        # IoU contra el resto
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # filtrar
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return keep

def infer_tflite(path):
    # Preprocesado
    inp_array, pil_resized = preprocess_image(path, MODEL_SIZE)

    # Ejecutar interprete
    interpreter.set_tensor(in_details['index'], inp_array)
    t0 = time.time()
    interpreter.invoke()
    t_inf = time.time() - t0

    # Leer salida 
    out = interpreter.get_tensor(out_details['index'])  # shape (1,5,5376)
    dets = out[0].T                                    # shape (5376,5)

    # Filtrar por confianza
    mask = dets[:, 4] > CONF_THRESH
    dets = dets[mask]
    if dets.shape[0] == 0:
        return pil_resized, [], t_inf

    # De xc,yc,w,h,conf  a x1,y1,x2,y2,conf
    xc, yc, w, h, conf = dets.T
    x1 = xc - w/2;  y1 = yc - h/2
    x2 = xc + w/2;  y2 = yc + h/2
    boxes  = np.stack([x1, y1, x2, y2], axis=1)
    scores = conf

    # NMS
    keep = nms_numpy(boxes, scores, IOU_THRESH)
    dets_out = [(boxes[i], scores[i]) for i in keep]

    return pil_resized, dets_out, t_inf

window = tk.Tk()
window.title("Deteccion TFLite Potholes")

def select_image():
    fp = filedialog.askopenfilename(filetypes=[("Imagen","*.jpg *.jpeg *.png")])
    if not fp: return

    pil_img, detections, t_inf = infer_tflite(fp)

    # dibujar resultados sobre una copia
    draw = ImageDraw.Draw(pil_img)
    for (x1,y1,x2,y2), score in detections:
        draw.rectangle([x1,y1,x2,y2], outline="lime", width=2)
        txt = f"{score:.2f}"
        draw.text((x1, y1-10), txt, fill="yellow", font=FONT)

    # mostrar en Tk
    tk_img = ImageTk.PhotoImage(pil_img.resize((320,320)))
    img_label.configure(image=tk_img); img_label.image = tk_img

    # texto resumen
    res_text.set(f"Baches: {len(detections)}\nTiempo inf.: {t_inf*1000:.1f} ms")

btn = tk.Button(window, text="Seleccionar imagen", command=select_image)
btn.pack(pady=5)

img_label = tk.Label(window)
img_label.pack(pady=5)

res_text = tk.StringVar()
tk.Label(window, textvariable=res_text, font=("Arial",12)).pack(pady=5)

window.mainloop()
