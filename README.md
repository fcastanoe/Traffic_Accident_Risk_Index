# Detección y Evaluación de Riesgo Vial en Raspberry Pi 5
---
## Autores:

- Fredy Andres Castaño Escobar C.C. 1003828997
- Edgar Ivan Calpa Cuacialpud C.C. 1004577329

---

Este repositorio contiene todo lo necesario para entrenar, analizar e inferir dos modelos (clasificación y detección), y para calcular un índice de riesgo vial a partir de imágenes o de una cámara en tiempo real, todo ejecutándose en una Raspberry Pi 5.

---

## 📁 Estructura de Carpetas
``` bash
├─ Analysis/ # Notebooks de entrenamiento y análisis de datos
│ ├─ analisis-dataset-de-accidentes-de-trafico.ipynb # Análisis exploratorio: condiciones de calle vs. accidentes
│ └─ train-classifier-road-conditions.ipynb # Entrenamiento y validación del modelo de clasificacion (Weather y Surface)
│ └─ train-detection -yolov8-pothole.ipynb # Entrenamiento y validación del modelo de deteccion de huecos
│
├─ Models/ # Documentación y enlaces para descargar modelos
│ └─ MODELS.md # Enlaces a pesos de:
│ • Clasificación (8 clases binarias)
│ • Detección de huecos
│
├─ Inferences/ # Scripts de inferencia y cálculo de índice de riesgo
│ ├─ inference_clasification_tflite.py # Inferencia del modelo de clasificación
│ ├─ inference_detection_tflite.py # Inferencia del modelo de detección de huecos
│ ├─ Risk_Index.py # Cálculo de índice de riesgo en imágenes estáticas
│ └─ inference_web_cam_1.py # Cálculo de índice de riesgo en video/cámara
│
├─ requirements.txt # Lista de dependencias de Python
└─ README.md # Documentación (este archivo)
```
---

## 🎯 Descripción del Proyecto

1. **Clasificación de Condiciones de Calle**  
   - Modelo entrenado con 2 600 imágenes anotadas en Roboflow.  
   - 8 clases binarias organizadas bajo dos prefijos:  
     - `Weather_`: (p.ej. `Weather_Rain`, `Weather_Clear`)  
     - `Defect_`:  (p.ej. `Defect_Dry`, `Defect_Wet`)

2. **Detección de Huecos (Potholes)**  
   - Localiza y cuenta huecos en la vía para medir su incidencia.

3. **Cálculo de Índice de Riesgo**  
   - Fusiona resultados de clasificación y detección.  
   - Genera un puntaje de riesgo para imágenes estáticas y secuencias en tiempo real.

---

## 🛠️ Entorno y Dependencias

### Python & Sistema Operativo

- **Python**: 3.9 o superior  
- **SO en Raspberry Pi 5**: Raspberry Pi OS 64‑bit  
- **Recomendación**: crear entorno virtual

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-tk
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
---

## ⚙️ Hardware y Software

| Componente            | Detalle                                 |
| --------------------- | --------------------------------------- |
| **Dispositivo**       | Raspberry Pi 5                          |
| **RAM**               | 8 GB DDR4                               |
| **Almacenamiento**    | SD Card 32 GB (clase 10)                |
| **Cámara**            | Módulo CSI o USB (opcional, para video) |
| **Sistema Operativo** | Raspberry Pi OS (64‑bit)                |

---

## 🚀 Guía Paso a Paso

1. Clonar Repositorio
   
```bash
git clone https://github.com/fcastanoe/Traffic_Accident_Risk_Index.git
cd Traffic_Accident_Risk_Index
```
2. Crear y activar entorno (ver sección anterior).

3. Instalar dependencias

```bash
pip install -r requirements.txt
```

4. Descarga de modelos

   - Revisar Models/MODELS.md.
   - Colocar los modelos en Downloads.

5. Corrección de ruta del modelo en la inferencia. Entrar al codigo de inferencia que desea ejecutar, y modificar las rutas de inferencias de donde se encuentra alojado su modelo.

6. Inferencia estática

```bash
python Inferences/inference_clasification_tflite.py
python Inferences/inference_detection_tflite.py
python Inferences/Risk_Index.py
```

7. Inferencia en tiempo real

```bash
python Inferences/inference_web_cam_1.py
```

---

## 📊 Matriz de Requerimientos

| Indicador                | Pruebas (Descripción)                               | Criterios de Éxito                        | Agrupación Funcional        | Corridas | Quién – Cuándo     | Revisado Por  |
| ------------------------ | --------------------------------------------------- | ----------------------------------------- | --------------------------- | -------- | ------------------ | ------------- |
| Weather\_Clear           | Entrenar con `Weather_Clear` y validar precisión    | Accuracy ≥ 90% en validación              | Clasificación Weather       | 3        | Fredy – 2025‑07‑20 | Cristian Quenguan|
| Defect\_Wet              | Entrenar con `Defect_Wet` y validar precisión       | Accuracy ≥ 90%                            | Clasificación Defect        | 3        | Fredy – 2025‑07‑20 | Cristian Quenguan     |
| Detección de Huecos      | Inferir sobre 50 imágenes con huecos anotados       | Recall ≥ 95%, F1‑score ≥ 0.90             | Detección de Huecos         | 5        | Fredy – 2025‑07‑22 | Cristian Quenguan  |
| Latencia Inferencia Live | Medir tiempo por frame en `risk_live.py`            | ≤ 200 ms por frame en Pi 5                | Rendimiento                 | 10       | Fredy – 2025‑07‑24 | Gabriela Romo |
| Robustez a Condiciones   | Pruebas nocturnas y lluvia simulada                 | Métricas ≥ 85% en todas clases `Weather_` | Pruebas de Robustez         | 4        | Fredy – 2025‑07‑25 | Gabriela Romo |
| Entorno Raspberry Pi     | Comprobar que la GUI de Tkinter arranca sin errores | Venta­na de prueba aparece en ≤ 3 s       | Software de Infraestructura | 1        | Fredy – 2025‑07‑25 | Gabriela Romo  |

 



