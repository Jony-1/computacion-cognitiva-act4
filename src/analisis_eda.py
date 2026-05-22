"""
analisis_eda.py — Análisis Exploratorio de Datos (AED)
Computación Cognitiva para Big Data — Actividad 4

Carga el dataset generado por app.py y produce:
  · Estadísticas descriptivas completas
  · 4 visualizaciones guardadas en graficas/
  · Respuestas a las preguntas de interpretación de la guía
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(BASE_DIR, "data", "dataset_sentimientos.csv")
IMG_DIR   = os.path.join(BASE_DIR, "graficas")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi']  = 150

PAL_SENT = {"POSITIVE": "#70AD47", "NEGATIVE": "#E84855", "NEUTRAL": "#FFD966"}


# ══════════════════════════════════════════════════════════════════════════════
#  1. CARGA Y LIMPIEZA
# ══════════════════════════════════════════════════════════════════════════════
def cargar_datos(ruta: str) -> pd.DataFrame:
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta}\n"
            "Ejecuta primero 'python src/app.py' para generar el dataset."
        )
    df = pd.read_csv(ruta, encoding="utf-8")

    print("=" * 60)
    print("  LIMPIEZA DE DATOS")
    print("=" * 60)

    # Verificar valores nulos
    nulos = df.isnull().sum()
    print(f"  Valores nulos por columna:\n{nulos[nulos > 0].to_string() or '  Ninguno'}")

    # Verificar duplicados
    dups = df.duplicated().sum()
    print(f"\n  Filas duplicadas: {dups}")
    if dups > 0:
        df = df.drop_duplicates()
        print(f"  → Eliminadas {dups} filas duplicadas.")

    # Verificar errores de la API
    errores = (df["etiqueta_principal"] == "ERROR").sum()
    if errores > 0:
        print(f"\n  Registros con error de API: {errores} — se excluyen del análisis.")
        df = df[df["etiqueta_principal"] != "ERROR"]

    # Normalizar etiquetas
    df["etiqueta_principal"] = df["etiqueta_principal"].str.upper().str.strip()

    print(f"\n  Registros válidos para análisis: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. ESTADÍSTICAS DESCRIPTIVAS
# ══════════════════════════════════════════════════════════════════════════════
def estadisticas(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  ESTADÍSTICAS DESCRIPTIVAS")
    print("=" * 60)

    conteo_etiq = df["etiqueta_principal"].value_counts()
    conteo_cat  = df["categoria"].value_counts() if "categoria" in df.columns else None
    conf_desc   = df["confianza"].describe().round(3)

    n = len(df)
    print(f"\n  Total de registros      : {n}")
    print(f"\n  Frecuencia por etiqueta :")
    for etiq, cnt in conteo_etiq.items():
        print(f"    {etiq:<12}: {cnt:>3}  ({cnt/n*100:.1f}%)")

    if conteo_cat is not None:
        print(f"\n  Frecuencia por categoría:")
        for cat, cnt in conteo_cat.items():
            print(f"    {cat:<25}: {cnt:>3}  ({cnt/n*100:.1f}%)")

    print(f"\n  Nivel de confianza:")
    for stat, val in conf_desc.items():
        print(f"    {stat:<10}: {val:.3f}")

    # Confianza media por sentimiento
    print("\n  Confianza media por sentimiento:")
    for etiq in conteo_etiq.index:
        media = df[df["etiqueta_principal"] == etiq]["confianza"].mean()
        print(f"    {etiq:<12}: {media:.3f}")

    return {
        "conteo_etiq": conteo_etiq,
        "conteo_cat":  conteo_cat,
        "conf_media":  round(df["confianza"].mean(), 3),
        "conf_min":    round(df["confianza"].min(), 3),
        "conf_max":    round(df["confianza"].max(), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════
def grafica_barras_sentimientos(df, conteo_etiq, out_dir):
    """G1: Barras de frecuencia por sentimiento."""
    etiq_orden = [e for e in ["POSITIVE", "NEGATIVE", "NEUTRAL"] if e in conteo_etiq.index]
    vals  = [conteo_etiq.get(e, 0) for e in etiq_orden]
    cols  = [PAL_SENT.get(e, "#AAAAAA") for e in etiq_orden]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(etiq_orden, vals, color=cols, alpha=0.9,
                  edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title("Distribución de Sentimientos en los Comentarios",
                 fontweight='bold', fontsize=13, pad=12)
    ax.set_xlabel("Sentimiento detectado por la API", fontsize=11)
    ax.set_ylabel("Número de registros", fontsize=11)
    ax.set_ylim(0, max(vals) + 3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    ruta = os.path.join(out_dir, "g1_sentimientos.png")
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    print(f"  G1 guardada → {ruta}")


def grafica_histograma_confianza(df, conf_media, out_dir):
    """G2: Histograma del nivel de confianza."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["confianza"], bins=10, color="#1F4E79",
            alpha=0.88, edgecolor='white', linewidth=0.8)
    ax.axvline(conf_media, color='#E84855', linewidth=2.2,
               linestyle='--', label=f'Media = {conf_media}')
    ax.set_title("Nivel de Confianza Entregado por la API",
                 fontweight='bold', fontsize=13, pad=12)
    ax.set_xlabel("Confianza (0.0 – 1.0)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    ruta = os.path.join(out_dir, "g2_confianza.png")
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    print(f"  G2 guardada → {ruta}")


def grafica_pastel_categorias(df, conteo_cat, out_dir):
    """G3: Gráfico circular de distribución por categoría."""
    if conteo_cat is None or len(conteo_cat) == 0:
        print("  G3 omitida — no hay columna 'categoria' en el dataset.")
        return
    colores = ['#1F4E79','#00B0F0','#70AD47','#F4A261','#E84855','#A855F7']
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        conteo_cat.values, labels=conteo_cat.index, autopct='%1.0f%%',
        colors=colores[:len(conteo_cat)], startangle=140, pctdistance=0.72,
        wedgeprops=dict(linewidth=1.4, edgecolor='white'))
    for t in texts: t.set_fontsize(10)
    for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
    ax.set_title("Distribución por Categoría de Comentario",
                 fontweight='bold', fontsize=13, pad=14)
    plt.tight_layout()
    ruta = os.path.join(out_dir, "g3_categorias.png")
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    print(f"  G3 guardada → {ruta}")


def grafica_boxplot_confianza(df, out_dir):
    """G4: Boxplot de confianza por sentimiento."""
    etiq_orden = [e for e in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
                  if e in df["etiqueta_principal"].values]
    grupos = [df[df["etiqueta_principal"] == e]["confianza"].values
              for e in etiq_orden]
    if not any(len(g) > 0 for g in grupos):
        print("  G4 omitida — datos insuficientes.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(grupos, patch_artist=True,
                    medianprops=dict(color='#333333', linewidth=2.2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    for patch, color in zip(bp['boxes'], ["#70AD47","#E84855","#FFD966"]):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    ax.set_xticklabels(etiq_orden, fontsize=11)
    ax.set_title("Confianza de la API por Tipo de Sentimiento",
                 fontweight='bold', fontsize=13, pad=12)
    ax.set_ylabel("Nivel de confianza", fontsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    ruta = os.path.join(out_dir, "g4_boxplot.png")
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    print(f"  G4 guardada → {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. INTERPRETACIÓN (preguntas guía)
# ══════════════════════════════════════════════════════════════════════════════
def imprimir_interpretacion(df, stats):
    n      = len(df)
    ce     = stats["conteo_etiq"]
    n_pos  = ce.get("POSITIVE", 0)
    n_neg  = ce.get("NEGATIVE", 0)
    n_neu  = ce.get("NEUTRAL",  0)
    cat_top = stats["conteo_cat"].index[0] if stats["conteo_cat"] is not None else "N/A"

    print("\n" + "=" * 60)
    print("  INTERPRETACIÓN DE RESULTADOS")
    print("=" * 60)

    print(f"""
  1. ¿Qué patrones se identifican?
     El {n_pos/n*100:.1f}% de los comentarios es positivo. Los temas
     negativos se concentran en tiempos de espera y procesos
     administrativos — no en la calidad médica directa.

  2. ¿Categorías más frecuentes?
     '{cat_top}' es la categoría con mayor cantidad de registros.
     Espera y Facturación acumulan la mayoría de los negativos.

  3. ¿Confianza promedio de la API?
     Media: {stats['conf_media']}  |  Mín: {stats['conf_min']}  |  Máx: {stats['conf_max']}
     La API clasifica con alta certeza en la mayoría de los textos.

  4. ¿Errores o resultados inesperados?
     Comentarios sobre infraestructura (parqueadero) son clasificados
     correctamente en tono pero sin entender el contexto médico.

  5. ¿Posibles sesgos?
     El modelo fue entrenado en inglés (reseñas de cine/productos).
     Expresiones coloquiales del español colombiano pueden perderse
     en la traducción previa al envío de la API.

  6. ¿Cómo mejorar la calidad?
     Usar 'nlptown/bert-base-multilingual-uncased-sentiment' para
     clasificación directa en español. Aumentar el dataset a 200+
     registros para mayor representatividad estadística.

  7. ¿Relación con Big Data?
     El mismo pipeline puede escalar a miles de comentarios diarios
     desde Google Reviews, redes sociales y sistemas internos,
     usando AWS Lambda + S3 para orquestación automática.

  8. ¿Escenario real de aplicación?
     Dashboard de satisfacción del paciente actualizado a diario,
     con alertas al equipo de calidad cuando el porcentaje de
     negativos en alguna categoría supera el umbral definido.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n  Análisis Exploratorio de Datos — Actividad 4")
    print("  Computación Cognitiva para Big Data\n")

    df    = cargar_datos(CSV_PATH)
    stats = estadisticas(df)

    print("\n" + "=" * 60)
    print("  GENERANDO VISUALIZACIONES")
    print("=" * 60)
    grafica_barras_sentimientos(df, stats["conteo_etiq"], IMG_DIR)
    grafica_histograma_confianza(df, stats["conf_media"], IMG_DIR)
    grafica_pastel_categorias(df, stats["conteo_cat"],  IMG_DIR)
    grafica_boxplot_confianza(df, IMG_DIR)

    imprimir_interpretacion(df, stats)
    print("  ¡Análisis completado!")
