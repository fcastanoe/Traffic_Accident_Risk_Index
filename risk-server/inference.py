import io
from PIL import Image
import numpy as np
import tensorflow as tf
from Risk_Index import run_cls, infer_tflite_det, compute_risk, recommendation

def analyze_image(img_bytes: bytes):
    # 1) Cargar imagen desde bytes
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 2) Guardar temporal si tus funciones necesitan ruta
    #    (o mejor reescr�belas para que acepten PIL.Image directamente).
    img.save("/tmp/last.jpg")

    # 3) Ejecutar clasificaci�n y detecci�n
    (s_lbl,s_p),(w_lbl,w_p) = run_cls("/tmp/last.jpg")
    det_img, holes, _ = infer_tflite_det("/tmp/last.jpg")

    # 4) Calcular riesgo
    score = compute_risk(w_lbl, s_lbl, holes)
    rec   = recommendation(score)

    # 5) Devolver JSON con datos + la imagen anotada como bytes
    buf = io.BytesIO()
    det_img.save(buf, format="PNG")
    buf.seek(0)
    return {
        "surface":   (s_lbl, round(s_p,2)),
        "weather":   (w_lbl, round(w_p,2)),
        "holes":     holes,
        "risk":      round(score,2),
        "recommend": rec
    }, buf.read()
