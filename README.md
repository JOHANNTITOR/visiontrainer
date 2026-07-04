# 🧾 Clasificador de Documentos

**Clasifica automáticamente tickets de peaje, boletas y facturas a partir de una foto o escaneo**, usando Machine Learning clásico (HOG + color) y una interfaz web hecha con Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ ¿Qué hace?

Sube fotos de documentos, entrena un modelo con un clic y clasifícalos automáticamente. Todo desde el navegador, sin necesidad de GPU ni datasets gigantes.

| Paso | Descripción |
|------|-------------|
| 📥 **1. Agregar imágenes** | Sube fotos/escaneos y asígnalas a una categoría (`peaje`, `boleta`, `factura` u otra personalizada) |
| 🧠 **2. Entrenar modelo** | Un botón extrae características de las imágenes y entrena un `RandomForestClassifier` en segundos |
| 🔍 **3. Clasificar** | Sube un documento nuevo y obtén la categoría predicha con su nivel de confianza |

---

## 🖼️ Vista previa

```
┌─────────────────────────────────────┐
│  🧾  Clasificador de Documentos      │
├─────────────────────────────────────┤
│  1. Agregar imágenes de entrenamiento│
│     [Categoría ▾]  [Subir archivos]  │
│                                       │
│  2. Entrenar modelo                  │
│     [ Entrenar modelo ]  ✅ 94% acc  │
│                                       │
│  3. Clasificar un documento          │
│     [Subir imagen]  →  "boleta" 97%  │
└─────────────────────────────────────┘
```

---

## 🧠 ¿Cómo funciona por dentro?

En vez de entrenar una red neuronal desde cero (que necesita miles de imágenes), este proyecto usa **visión computacional clásica**, ideal para datasets pequeños (decenas o cientos de fotos):

1. **HOG (Histogram of Oriented Gradients)** — captura la forma, bordes y estructura del documento (un ticket angosto de peaje no tiene la misma silueta que una factura con tablas).
2. **Histograma de color RGB** — captura diferencias de impresión (papel térmico, blanco y negro, logos a color, etc).
3. **RandomForestClassifier** — combina ambas señales para predecir la categoría.

Esto entrena en segundos, corre en cualquier laptop y funciona sorprendentemente bien para distinguir tipos de documentos con diseños visualmente distintos.

> 💡 ¿Tienes cientos de imágenes por categoría y buscas más precisión? El proyecto está preparado para migrar a *deep learning* (fine-tuning de una red preentrenada) sin cambiar la interfaz.

---

## 🚀 Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/tu-usuario/clasificador-documentos.git
cd clasificador-documentos

# 2. Crea un entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate

# 3. Instala las dependencias
pip install -r requirements.txt

# 4. Ejecuta la app
streamlit run clasificador_documentos.py
```

La app se abrirá automáticamente en `http://localhost:8501`.

---

## 📁 Estructura del proyecto

```
clasificador-documentos/
├── clasificador_documentos.py   # App principal de Streamlit
├── requirements.txt             # Dependencias del proyecto
├── dataset_documentos/          # Se crea automáticamente al subir imágenes
│   ├── manifest.csv             # Registro de cada imagen y su categoría
│   ├── peaje/
│   ├── boleta/
│   └── factura/
└── modelo_clasificador_documentos.pkl   # Modelo entrenado (se genera al entrenar)
```

---

## 🛠️ Tecnologías usadas

- [Streamlit](https://streamlit.io/) — interfaz web interactiva
- [scikit-learn](https://scikit-learn.org/) — `RandomForestClassifier`
- [scikit-image](https://scikit-image.org/) — extracción de características HOG
- [Pillow](https://python-pillow.org/) — procesamiento de imágenes
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — manejo de datos

---

## 🗺️ Roadmap

- [ ] Soporte para PDF además de imágenes
- [ ] Exportar el dataset etiquetado a un `.zip`
- [ ] Opción de entrenar con una red preentrenada (transfer learning)
- [ ] Métricas de entrenamiento con matriz de confusión visual
- [ ] Despliegue en Streamlit Community Cloud

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Eres libre de usarlo, modificarlo y distribuirlo.

---

<p align="center">
  Hecho con ❤️ y 🧠 en Chile
</p>
