# 📄 Detección y Evaluación de Riesgo Vial en Raspberry Pi 5
---
## 🎯 Objetivo

Prevenir accidentes y aumentar la seguridad vial mediante un sistema embebido capaz de clasificar condiciones meteorológicas y de superficie, detectar huecos en la vía y calcular un índice de riesgo en tiempo real.

---
## 👥 Autores

- Fredy Andres Castaño Escobar C.C. 1003828997
- Edgar Ivan Calpa Cuacialpud C.C. 1004577329

---
📝 Descripción General

Este proyecto implementa en una Raspberry Pi 5 un flujo completo para:

1. Clasificar condiciones de clima y superficie (seco, mojado, lluvia, niebla, etc.) usando un modelo TensorFlow Lite.

2. Detectar huecos (potholes) en la carretera y contarlos con un detector TFLite.

3. Calcular un Índice de Riesgo combinando probabilidades de clasificación y densidad de huecos, y generar recomendaciones de manejo.

4. Exponer esta lógica como una API REST en Flask con una interfaz web ligera y responsiva.

5. Desplegar la aplicación como un servicio systemd, arrancando automáticamente al encender la Pi.

Con esto obtenemos un sistema remoto accesible desde cualquier navegador para subir imágenes o transmitir video y obtener, en < 200 ms por frame en Pi 5, un análisis completo de riesgo vial.

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
│ ├─ inference_risk_index.py # Cálculo de índice de riesgo en imágenes estáticas
│ └─ inference_web_cam_1.py # Cálculo de índice de riesgo en video/cámara
|
├─ risk-server/              # API REST Flask + servicio systemd  
│   ├─ app.py                # Servidor Flask  
│   ├─ inference.py          # Función analyze_image para el endpoint /analyze  
│   ├─ templates/index.html  # Interfaz web  
│   └─ Risk_Index.py         # Inferencia de indice de riesgo para la API 
│
├─ requirements.txt # Lista de dependencias de Python
└─ README.md # Documentación (este archivo)
```
---
## 🛠️ Entorno y Dependencias

### Python & Sistema Operativo

- **Python**: 3.9 o superior  
- **SO en Raspberry Pi 5**: Raspberry Pi OS 64‑bit  
- **Paquetes del sistema**: python3-venv, python3-pip, python3-tk

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-tk
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r Traffic_Accident_Risk_Index/requirements.txt
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

3. Descargar Modelos según ```Models/MODELS.md``` y colocarlos en ```Downloads/```.

4. Ajustar rutas en los scripts de ```Inferences/``` o en ```risk-server/Risk_Index.py``` para apuntar a tus archivos ```.tflite```.

5. Probar local:

```bash
# Inferencia estática
source venv/bin/activate
python Inferences/inference_risk_index.py

# API web
cd risk-server
source venv/bin/activate
python app.py
# Luego abre en tu navegador: http://IP-de-tu-Pi:5000
```
6. Crear servicio systemd (fuera del repositorio):

- Crea ```/etc/systemd/system/risk-server.service``` con contenido:

```ini
[Unit]
Description=Risk Index Web Service
After=network.target

[Service]
Type=simple
User=fcastanoe
WorkingDirectory=/home/TU-USUARIO/Traffic_Accident_Risk_Index/risk-server

ExecStart=/home/TU-USUARIO/venv/bin/python \
/home/TU-USUARIO/Traffic_Accident_Risk_Index/risk-server/app.py

Restart=on-failure
RestartSec=5s
Environment=FLASK_ENV=production

[Install]
WantedBy=multi-user.target
```

- Recarga y habilita:

```bash
sudo systemctl daemon-reload
sudo systemctl enable risk-server.service
sudo systemctl start risk-server.service
sudo systemctl status risk-server.service
```

7. Acceder desde cualquier navegador a http://IP-de-tu-Pi:5000/.
---
## 🔍 Ejemplos de Uso

1. Inferencia local en la Raspberry Pi.

En la carpeta raíz, tras activar el entorno virtual, basta con llamar el .py y te aparecera una interfaz para Seleccionar imagen.

```bash
source venv/bin/activate
python Inferences/inference_risk_index.py
```

<img width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/8ee25ec8-8079-487b-9351-b5e04827e93f" />

Al ejecutarlo verás en consola:

- Las etiquetas clasificadas y sus probabilidades.

- El número de huecos detectados.

- El índice de riesgo calculado y la recomendación asociada.

- Recomendaciones Adicionales

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/1408e79f-7bf3-47db-bbb2-8a4eac516e29" />

2. Uso vía API REST con interfaz web

   1. Navega en tu navegador a http://IP-de-tu-Pi:5000/.
   2. Haz clic en “Seleccionar imagen” y busca el archivo local que desees analizar.
   3. Pulsa el botón “Analizar Imagen”.
   4. Al momento, la página mostrará:
      - Una tarjeta con Superficie, Clima, Huecos, Índice de riesgo y Recomendación.
      - La imagen procesada con los recuadros de detección de huecos.

<img width="640" height="640" alt="image" src="https://github.com/user-attachments/assets/bb08e359-ecd8-4d98-8911-08470760d1da" />

---

## 📊 Matriz de Requerimientos

| Task | Descripción                                                                     | Criterio de Éxito                                                                 | Runs | Fecha         | Revisado Por |
| ---- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------- | --------------------- | ------------ |
| 1    | Validar e inferir correctamente modelo de **clasificación** convertido a TFLite | Clasificaciones `Weather_` y `Surface_` con accuracy ≥ 90% en inferencia estática | 1        | 25/06/2025 09:18 p.m.| Cristian Quenguan    |
| 2    | Validar e inferir correctamente modelo de **detección** exportado a TFLite      | Detección de huecos con recall ≥ 95% y F1 ≥ 0.90 en inferencia estática           | 1        | 25/06/2025 04:55 p.m. | Gabriela Romo     |
| 3    | Validar inferencia conjunta y cálculo de **índice de riesgo**                   | Índice coherente con resultados individuales y recomendaciones plausibles         | 1        | 03/07/2025 08:34 p.m.| Cristian Quenguan      |
| 4    | Clonar repositorio y descargar dependencias paso a paso en la Raspberry Pi      | Entorno listo y `venv` activado sin errores, `pip install` completo               | 1        | 09/07/2025 10:43 a.m. | Gabriela Romo    |
| 5    | Verificar inferencias individuales de ambos modelos en la Raspberry Pi          | Scripts de inferencia funcionan correctamente                                     | 1        | 09/07/2025 11:27 a.m.| Gabriela Romo    |
| 6    | Verificar correcto cálculo del índice de riesgo por **imágenes** en la Pi       | `Risk_Index.py` muestra índice y recomendaciones adecuadas                        | 1        | 09/07/2025 11:39 a.m.| Gabriela Romo     |
| 7    | Testear la **API REST** localmente para la inferencia de índice de riesgo       | Endpoint `/analyze` responde HTTP 200 con JSON y data URI de imagen               | 1        | 24/07/2025 05:53 p.m. | Cristian Quenguan     |
| 8    | Verificar funcionamiento del **servicio systemd** al iniciar la Raspberry Pi    | `systemctl status risk-server.service` muestra `active (running)` tras reboot     | 1        | 24/07/2025 06:32 p.m. | Cristian Quenguan     |



 



