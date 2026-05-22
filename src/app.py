"""
app.py — Aplicación ETL con API Cognitiva de Hugging Face
Computación Cognitiva para Big Data — Actividad 4

Flujo:
  1. EXTRAER   → leer comentarios desde un CSV local
  2. TRANSFORMAR → clasificar cada texto con Hugging Face API
  3. CARGAR    → guardar el dataset enriquecido con etiqueta y confianza

El análisis corre desde código local: los datos NO se suben
directamente a ningún chat de IA, lo que protege la privacidad
de los pacientes según la Ley 1581 de 2012 (Colombia).
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ── Cargar variables de entorno ────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "No se encontró HF_TOKEN en el archivo .env\n"
        "Copia .env.example a .env y agrega tu token de Hugging Face."
    )

# ── Configuración del modelo ───────────────────────────────────────────────────
API_URL = (
    "https://api-inference.huggingface.co/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Rutas de archivos
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "comentarios_entrada.csv")
OUT_CSV   = os.path.join(BASE_DIR, "data", "dataset_sentimientos.csv")


# ── Función de clasificación ───────────────────────────────────────────────────
def clasificar_sentimiento(texto: str) -> tuple[str, float]:
    """
    Envía un texto a la API de Hugging Face y retorna la etiqueta
    y el nivel de confianza del modelo.

    Parámetros
    ----------
    texto : str
        Texto del comentario a clasificar.

    Retorna
    -------
    etiqueta  : str   — 'POSITIVE' o 'NEGATIVE'
    confianza : float — probabilidad asignada por el modelo (0.0 – 1.0)
    """
    try:
        respuesta = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": texto},
            timeout=15
        )
        respuesta.raise_for_status()
        resultado = respuesta.json()[0][0]
        return resultado["label"], round(resultado["score"], 4)
    except requests.exceptions.RequestException as e:
        print(f"  ⚠  Error al procesar: '{texto[:40]}...' → {e}")
        return "ERROR", 0.0


# ── Función principal: ETL completo ───────────────────────────────────────────
def ejecutar_etl(ruta_entrada: str, ruta_salida: str) -> pd.DataFrame:
    """
    Ejecuta el pipeline ETL completo:
      Extraer → Transformar → Cargar

    Parámetros
    ----------
    ruta_entrada : str  — CSV con columna 'texto' (y opcionalmente 'categoria')
    ruta_salida  : str  — CSV de salida con etiqueta y confianza añadidas

    Retorna
    -------
    DataFrame con el dataset enriquecido.
    """

    # ── EXTRAER ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  PASO 1 — EXTRAER: leyendo datos locales")
    print("=" * 60)

    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {ruta_entrada}\n"
            "Asegúrate de tener el archivo 'comentarios_entrada.csv' en data/"
        )

    df = pd.read_csv(ruta_entrada, encoding="utf-8")
    print(f"  Registros cargados  : {len(df)}")
    print(f"  Columnas disponibles: {list(df.columns)}")
    print(f"\n  Primeras 3 filas:")
    print(df.head(3).to_string(index=False))

    if "texto" not in df.columns:
        raise ValueError("El CSV debe tener una columna llamada 'texto'.")

    # Descartar filas con texto vacío
    df = df[df["texto"].notna() & (df["texto"].str.strip() != "")]
    print(f"\n  Registros válidos después de limpieza: {len(df)}")

    # ── TRANSFORMAR ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PASO 2 — TRANSFORMAR: clasificando con Hugging Face API")
    print("=" * 60)

    etiquetas, confianzas = [], []
    fecha_hoy = datetime.today().strftime("%Y-%m-%d")

    for idx, fila in df.iterrows():
        texto_raw = str(fila["texto"])
        print(f"  [{idx + 1:02d}/{len(df)}] {texto_raw[:55]:<55}", end=" → ")
        etiqueta, confianza = clasificar_sentimiento(texto_raw)
        print(f"{etiqueta:<10} ({confianza:.3f})")
        etiquetas.append(etiqueta)
        confianzas.append(confianza)

    df["etiqueta_principal"]  = etiquetas
    df["confianza"]           = confianzas
    df["fecha_procesamiento"] = fecha_hoy
    df["observaciones"]       = "Procesado con Hugging Face Inference API"

    # ── CARGAR ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PASO 3 — CARGAR: guardando dataset enriquecido")
    print("=" * 60)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    print(f"  Dataset guardado en: {ruta_salida}")
    print(f"  Total registros    : {len(df)}")

    # ── RESUMEN ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN DEL PROCESAMIENTO")
    print("=" * 60)
    print(df["etiqueta_principal"].value_counts().to_string())
    print(f"\n  Confianza media : {df['confianza'].mean():.3f}")
    print(f"  Confianza mín.  : {df['confianza'].min():.3f}")
    print(f"  Confianza máx.  : {df['confianza'].max():.3f}")

    errores = (df["etiqueta_principal"] == "ERROR").sum()
    if errores > 0:
        print(f"\n  ⚠  {errores} registros no pudieron procesarse (ver columna 'etiqueta_principal' = ERROR)")

    return df


# ── Punto de entrada ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Aplicación ETL — Análisis de Sentimientos con Hugging Face")
    print("  Computación Cognitiva para Big Data — Actividad 4\n")

    df_resultado = ejecutar_etl(INPUT_CSV, OUT_CSV)

    print("\n  ¡ETL completado! Ejecuta 'python src/analisis_eda.py' para ver el análisis.")
