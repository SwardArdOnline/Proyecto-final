# Crear la app (Creación Rápida de Archivos de Script fuera del notebook)
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Inferencia de Ingresos", layout="centered")
st.title("Inferencia de Ingresos de Clientes")
st.markdown("Esta aplicación estima el **ingreso** de un cliente y lo ubica en un **nivel de "
            "ingreso** (Bajo / Medio / Alto), a partir de sus datos demográficos y financieros.")

# Cargar modelos
try:
    with open("modelo_regresion_ingresos.pkl", "rb") as file:
        modelo_regresion = pickle.load(file)
    with open("modelo_clasificacion_ingresos.pkl", "rb") as file:
        modelo_clasificacion = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"Error: no se encontró un archivo de modelo ({e}). "
             "Asegúrese de que los .pkl estén en la misma carpeta que app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# Diccionario de categorías (mismo orden que en el entrenamiento)
mapa_inverso = {0: "Bajo", 1: "Medio", 2: "Alto"}
COLOR_CATEGORIA = {"Bajo": "#C00000", "Medio": "#ED7D31", "Alto": "#70AD47"}

# Controles de entrada
st.sidebar.header("Ingrese los datos del cliente")

edad = st.sidebar.slider("Edad", min_value=18, max_value=80, value=35)
anios_direccion = st.sidebar.slider("Años en la dirección actual", min_value=0, max_value=40, value=5)
gasto_coche = st.sidebar.number_input("Gasto en coche", min_value=0.0, max_value=200.0, value=20.0, step=0.5)
anios_empleo = st.sidebar.slider("Años de empleo", min_value=0, max_value=50, value=8)
anios_residen = st.sidebar.slider("Años de residencia", min_value=0, max_value=40, value=3)

# Botón de predicción
if st.button("Predecir ingreso"):
    # 1. Preparación de los datos de entrada (mismas columnas y orden que en el entrenamiento)
    input_data = pd.DataFrame({
        'edad':           [edad],
        'AniosDireccion': [anios_direccion],
        'Gastocoche':     [gasto_coche],
        'Aniosempleo':    [anios_empleo],
        'Aniosresiden':   [anios_residen]
    })

    # 2. Predicción del ingreso exacto (regresión)
    try:
        ingreso_predicho = modelo_regresion.predict(input_data)[0]
    except Exception as e:
        st.error(f"Error al predecir el ingreso: {e}")
        st.stop()

    # 3. Predicción del nivel de ingreso (clasificación)
    try:
        categoria_cod = modelo_clasificacion.predict(input_data)[0]
        categoria_proba = modelo_clasificacion.predict_proba(input_data)[0]
    except Exception as e:
        st.error(f"Error al predecir la categoría: {e}")
        st.stop()

    categoria = mapa_inverso[categoria_cod]
    color = COLOR_CATEGORIA[categoria]

    # 4. Visualización de resultados

    # A. Ingreso estimado
    st.success(f"Ingreso estimado: **{ingreso_predicho:,.1f}**")

    # B. Nivel de ingreso
    st.markdown(
        f"""
        <div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;'>
            <b>Nivel de ingreso: {categoria}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # C. Probabilidades por categoría
    st.markdown("### Probabilidad por categoría")
    proba_df = pd.DataFrame({
        'Categoría': ['Bajo', 'Medio', 'Alto'],
        'Probabilidad': categoria_proba
    })
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(proba_df['Categoría'], proba_df['Probabilidad'],
           color=[COLOR_CATEGORIA[c] for c in proba_df['Categoría']])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidad")
    st.pyplot(fig)

    # D. Ficha de validación
    st.markdown("### Ficha de validación")
    etiquetas = ['Edad', 'Años en dirección actual', 'Gasto en coche', 'Años de empleo', 'Años de residencia']
    valores = [edad, anios_direccion, gasto_coche, anios_empleo, anios_residen]
    perfil_df = pd.DataFrame({'Variable': etiquetas, 'Valor ingresado': valores})
    st.table(perfil_df.set_index('Variable'))
