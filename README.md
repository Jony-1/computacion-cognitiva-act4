# Actividad 4 — Herramientas Inteligentes para el Análisis de Datos

**Curso:** Computación Cognitiva para Big Data  
**Docente:** Joaquín Fernando Sánchez Cifuentes  
**Estudiante:** Jonathan Dario Sierra Galindo  
**Fecha:** Mayo 2026

---

## Descripción

Análisis exploratorio de datos (AED) de comentarios de pacientes de una clínica colombiana,
utilizando la **API de Hugging Face** para clasificar automáticamente el sentimiento de cada texto.

El pipeline ETL corre completamente desde código local: los datos **no se exponen** directamente
a ningún chat de IA, protegiendo la privacidad de los pacientes según la **Ley 1581 de 2012**.

---

## API cognitiva utilizada

| Campo | Detalle |
|-------|---------|
| Plataforma | [Hugging Face Inference API](https://huggingface.co/inference-api) |
| Modelo | `distilbert-base-uncased-finetuned-sst-2-english` |
| Tarea | Clasificación de sentimientos: POSITIVE / NEGATIVE |
| Acceso | Gratuito con token personal (30 000 llamadas/mes) |

---

## Estructura del proyecto

```
cognitiva-act4/
├── src/
│   ├── app.py              # ETL principal: leer CSV → API → guardar resultado
│   └── analisis_eda.py     # Análisis exploratorio + 4 visualizaciones
├── data/
│   ├── comentarios_entrada.csv   # Dataset de entrada (25 comentarios)
│   └── dataset_sentimientos.csv  # Dataset enriquecido con etiqueta y confianza
├── graficas/
│   ├── g1_sentimientos.png       # Barras de distribución de sentimientos
│   ├── g2_confianza.png          # Histograma del nivel de confianza
│   ├── g3_categorias.png         # Gráfico circular por categoría
│   └── g4_boxplot.png            # Boxplot de confianza por sentimiento
├── .env.example            # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Jony-1/computacion-cognitiva-act4.git
cd computacion-cognitiva-act4
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar el token de Hugging Face

```bash
# Copiar la plantilla
cp .env.example .env
```

Editar `.env` y reemplazar `hf_xxx...` con tu token real.  
Consíguelo en: https://huggingface.co/settings/tokens

> **Importante:** El archivo `.env` está en `.gitignore` — nunca se sube a GitHub.
> GitHub escanea automáticamente los commits buscando tokens y bloquea el push si los detecta.

### 4. Ejecutar el ETL

```bash
python src/app.py
```

Este script:
1. **Extrae** los comentarios de `data/comentarios_entrada.csv`
2. **Transforma** cada texto llamando a la API de Hugging Face
3. **Carga** el resultado en `data/dataset_sentimientos.csv`

### 5. Ejecutar el análisis exploratorio

```bash
python src/analisis_eda.py
```

Genera estadísticas descriptivas, las 4 visualizaciones en `graficas/`
y responde las preguntas de interpretación de la guía.

---

## Escenario de aplicación

**Clínica del Norte – Bogotá, Colombia**

El equipo de calidad recibe cientos de reseñas de pacientes cada mes desde
Google Reviews, redes sociales y formularios internos. Sin automatización,
revisarlas toma días y las decisiones de mejora llegan tarde.

Este pipeline clasifica cada comentario en segundos, genera métricas por
categoría (espera, personal, instalaciones, facturación) y permite al equipo
detectar tendencias negativas antes de que escalen.

---

## Dataset generado

El dataset de salida `dataset_sentimientos.csv` tiene las siguientes columnas:

| Columna | Descripción |
|---------|-------------|
| `id` | Identificador único del registro |
| `texto` | Comentario original del paciente |
| `categoria` | Área del servicio evaluada |
| `etiqueta_principal` | Sentimiento: POSITIVE o NEGATIVE |
| `confianza` | Nivel de certeza del modelo (0.0 – 1.0) |
| `fecha_procesamiento` | Fecha de ejecución del ETL |
| `observaciones` | Nota sobre el origen del procesamiento |

---

## Resultados del análisis (muestra de 25 registros)

- **48%** comentarios positivos
- **40%** comentarios negativos  
- **12%** comentarios neutros
- Confianza media de la API: **0.830**
- Categorías con más negativos: Espera y Facturación

---

## Referencias

- Hugging Face. (2024). *DistilBERT base uncased finetuned SST-2*. https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english  
- Congreso de Colombia. (2012). *Ley 1581 de 2012: Protección de datos personales*. Diario Oficial No. 48.587.  
- García Alsina, M. (2017). *Big data: gestión y explotación de grandes volúmenes de datos*. Editorial UOC.
