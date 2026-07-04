# importar librerías
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# scikit-learn: modelo de clasificación
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# scikit-image: para transformar imágenes en características numéricas
from skimage.feature import hog
from skimage.color import rgb2gray

# cache de forma nativa
import joblib

# --------------------------
# CONFIGURACIÓN / RUTAS
# --------------------------

CARPETA_DATASET = "dataset_documentos"
ARCHIVO_MANIFEST = os.path.join(CARPETA_DATASET, "manifest.csv")
ARCHIVO_MODELO = "modelo_clasificador_documentos.pkl"
TAMANO_IMG = (128, 128)  # tamaño al que se redimensiona toda imagen antes de procesar

CATEGORIAS_BASE = ["peaje", "boleta", "factura"]

os.makedirs(CARPETA_DATASET, exist_ok=True)


def cargar_manifest():
    """Carga el csv que relaciona cada imagen guardada con su categoría."""
    if os.path.exists(ARCHIVO_MANIFEST):
        return pd.read_csv(ARCHIVO_MANIFEST)
    return pd.DataFrame(columns=["archivo", "categoria"])


def guardar_manifest(df):
    df.to_csv(ARCHIVO_MANIFEST, index=False)


# --------------------------
# EXTRACCIÓN DE CARACTERÍSTICAS
# --------------------------
# Como el dataset de este tipo de proyectos suele ser chico (decenas o
# cientos de imágenes, no miles), en vez de entrenar una red neuronal desde
# cero usamos "features" clásicas de visión computacional (HOG + histograma
# de color) y las alimentamos a un RandomForest, igual que en prediktor.py.
# Esto entrena en segundos y funciona bien para distinguir documentos con
# diseños/colores distintos (ticket de peaje angosto y térmico, boleta,
# factura con más texto y tablas, etc).

def extraer_caracteristicas(imagen_pil):
    img = imagen_pil.convert("RGB").resize(TAMANO_IMG)
    arr = np.array(img)

    # forma / bordes del documento
    gris = rgb2gray(arr)
    hog_feat = hog(
        gris,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # distribución de color (papel térmico, blanco y negro, con logo a color, etc)
    hist_r = np.histogram(arr[:, :, 0], bins=16, range=(0, 255))[0]
    hist_g = np.histogram(arr[:, :, 1], bins=16, range=(0, 255))[0]
    hist_b = np.histogram(arr[:, :, 2], bins=16, range=(0, 255))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    hist = hist / (hist.sum() + 1e-6)

    return np.concatenate([hog_feat, hist])


# --------------------------
# INTERFAZ
# --------------------------

st.title("Clasificador de Documentos")
st.subheader("Peajes, boletas y facturas a partir de una imagen")

manifest = cargar_manifest()

# --------------------------------
# 1. AGREGAR IMÁGENES AL DATASET
# --------------------------------

st.header("1. Agregar imágenes de entrenamiento")
st.write(
    "Sube fotos o escaneos de tickets de peaje, boletas y facturas, "
    "indicando a qué categoría pertenecen."
)

categoria_sel = st.selectbox("Categoría", CATEGORIAS_BASE + ["otro"])

if categoria_sel == "otro":
    categoria_final = st.text_input("Nombre de la nueva categoría", max_chars=30).strip().lower()
else:
    categoria_final = categoria_sel

archivos_subidos = st.file_uploader(
    "Selecciona una o varias imágenes",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.button("Agregar imágenes al dataset"):
    if not categoria_final:
        st.error("Debes indicar una categoría válida.")
    elif not archivos_subidos:
        st.error("Debes subir al menos una imagen.")
    else:
        carpeta_categoria = os.path.join(CARPETA_DATASET, categoria_final)
        os.makedirs(carpeta_categoria, exist_ok=True)

        nuevas_filas = []
        for archivo in archivos_subidos:
            img = Image.open(archivo)
            nombre_archivo = f"{len(os.listdir(carpeta_categoria))}_{archivo.name}"
            ruta_relativa = os.path.join(categoria_final, nombre_archivo)
            ruta_absoluta = os.path.join(CARPETA_DATASET, ruta_relativa)
            img.convert("RGB").save(ruta_absoluta)
            nuevas_filas.append({"archivo": ruta_relativa, "categoria": categoria_final})

        manifest = pd.concat([manifest, pd.DataFrame(nuevas_filas)], ignore_index=True)
        guardar_manifest(manifest)
        st.success(f"Se agregaron {len(archivos_subidos)} imagen(es) a la categoría '{categoria_final}'.")
        st.rerun()

# --------------------------------
# ESTADO ACTUAL DEL DATASET
# --------------------------------

st.subheader("Dataset actual")
if manifest.empty:
    st.info("Todavía no hay imágenes cargadas.")
else:
    conteo = manifest["categoria"].value_counts()
    st.dataframe(manifest)
    st.bar_chart(conteo)

# --------------------------------
# 2. ENTRENAR MODELO
# --------------------------------

st.header("2. Entrenar modelo")

if st.button("Entrenar modelo"):
    if manifest.empty or manifest["categoria"].nunique() < 2:
        st.error(
            "Necesitas imágenes de al menos 2 categorías distintas para poder entrenar."
        )
    else:
        with st.spinner("Extrayendo características y entrenando..."):
            X = []
            y = []
            for _, fila in manifest.iterrows():
                ruta = os.path.join(CARPETA_DATASET, fila["archivo"])
                if not os.path.exists(ruta):
                    continue
                img = Image.open(ruta)
                X.append(extraer_caracteristicas(img))
                y.append(fila["categoria"])

            X = np.array(X)
            y = np.array(y)

            # separa un pequeño set de validación si hay datos suficientes
            reporte_texto = None
            if len(X) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, y_train = X, y
                X_test, y_test = None, None

            modelo = RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
            modelo.fit(X_train, y_train)

            if X_test is not None:
                y_pred = modelo.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                reporte_texto = classification_report(y_test, y_pred, zero_division=0)

            joblib.dump(modelo, ARCHIVO_MODELO)

        st.success("Modelo entrenado correctamente.")
        if reporte_texto:
            st.write(f"Precisión en set de validación: **{acc:.0%}**")
            st.text(reporte_texto)
        else:
            st.info(
                "El dataset es pequeño (menos de 10 imágenes), así que se entrenó "
                "con todos los datos sin separar un set de validación."
            )

# --------------------------------
# 3. PREDECIR / CLASIFICAR
# --------------------------------

st.header("3. Clasificar un documento")

archivo_predecir = st.file_uploader(
    "Sube la imagen del documento que quieres clasificar",
    type=["png", "jpg", "jpeg"],
    key="predecir"
)

if archivo_predecir is not None:
    imagen_pred = Image.open(archivo_predecir)
    st.image(imagen_pred, caption="Documento a clasificar", width=300)

if st.button("Clasificar documento"):
    if archivo_predecir is None:
        st.error("Primero sube una imagen.")
    elif not os.path.exists(ARCHIVO_MODELO):
        st.error("Primero debes entrenar el modelo.")
    else:
        modelo = joblib.load(ARCHIVO_MODELO)
        caracteristicas = extraer_caracteristicas(imagen_pred).reshape(1, -1)

        prediccion = modelo.predict(caracteristicas)[0]
        st.success(f"Categoría predicha: **{prediccion}**")

        if hasattr(modelo, "predict_proba"):
            probas = modelo.predict_proba(caracteristicas)[0]
            df_probas = pd.DataFrame({
                "categoria": modelo.classes_,
                "probabilidad": probas
            }).sort_values("probabilidad", ascending=False)
            st.write("Confianza por categoría:")
            st.dataframe(df_probas, hide_index=True)
