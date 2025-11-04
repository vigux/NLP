"""
Chatbot de autoaprendizaje (Active Learning) con TensorFlow/Keras.
- Clasifica entradas del usuario en 'intenciones' (intents).
- Responde con plantillas por intención.
- Si la confianza es baja, solicita etiqueta/respuesta y re-entrena en caliente.
"""

# =========================
# 1) IMPORTS Y CONFIGURACIÓN
# =========================
import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Para reproducibilidad (opcional)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Directorio donde se guardarán archivos del modelo y datos aprendidos
ARTIFACTS_DIR = "chatbot_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Umbral de confianza: por debajo de este valor, el bot pide ayuda y aprende
CONFIDENCE_THRESHOLD = 0.55

# Hiperparámetros base
MAX_TOKENS = 10000        # Tamaño de vocabulario del TextVectorization
MAX_LEN = 20              # Longitud máxima de secuencia (tokens por frase)
EMBED_DIM = 64            # Dimensión de embeddings
LSTM_UNITS = 64           # Unidades de la BiLSTM
BATCH_SIZE = 16
EPOCHS_INIT = 20          # Épocas de entrenamiento inicial
EPOCHS_INCREMENTAL = 5    # Épocas cuando se aprende nuevo dato


# ===================================
# 2) INTENTS: PATRONES Y RESPUESTAS
# ===================================
# Conjunto inicial mínimo (puede ampliarse). Cada intent tiene:
# - "patterns": ejemplos de entradas del usuario para esa intención
# - "responses": listas de respuestas posibles (se elige aleatoriamente)
# NOTA: este dataset crecerá durante la sesión con nuevos ejemplos.
intents = {
    "saludo": {
        "patterns": [
            "hola", "buenos días", "buenas tardes", "qué tal", "hey"
        ],
        "responses": [
            "¡Hola! ¿En qué puedo ayudarte?",
            "¡Un gusto saludarte! ¿Qué te gustaría saber?"
        ]
    },
    "despedida": {
        "patterns": [
            "adiós", "hasta luego", "gracias, nos vemos", "me voy", "bye"
        ],
        "responses": [
            "¡Hasta luego! Que tengas un excelente día.",
            "¡Nos vemos! Vuelve cuando quieras."
        ]
    },
    "clima": {
        "patterns": [
            "¿cómo está el clima?", "¿va a llover hoy?", "pronóstico del tiempo", "clima actual"
        ],
        "responses": [
            "No tengo datos del clima en tiempo real, pero puedo ayudarte a consultarlo si me indicas tu ciudad.",
            "El clima depende de tu ubicación. ¿En qué ciudad te encuentras?"
        ]
    },
    "agradecimiento": {
        "patterns": [
            "gracias", "muchas gracias", "te lo agradezco", "mil gracias"
        ],
        "responses": [
            "¡Con gusto! ¿Algo más en lo que te pueda apoyar?",
            "Para servirte, ¿quieres preguntar otra cosa?"
        ]
    }
}

# Guardamos/actualizamos intents en disco para persistir el aprendizaje
INTENTS_PATH = os.path.join(ARTIFACTS_DIR, "intents.json")

def save_intents(intents_dict):
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(intents_dict, f, ensure_ascii=False, indent=2)

def load_intents():
    if os.path.exists(INTENTS_PATH):
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Si ya hay intents previos aprendidos, los cargamos
loaded = load_intents()
if loaded:
    intents = loaded


# ======================================
# 3) PREPARACIÓN DE DATOS (X: texto, y: etiqueta)
# ======================================
def build_dataset_from_intents(intents_dict):
    """Crea listas de textos y etiquetas a partir de intents."""
    texts, labels = [], []
    for intent, data in intents_dict.items():
        for p in data["patterns"]:
            texts.append(p.lower().strip())
            labels.append(intent)
    return texts, labels

texts, labels = build_dataset_from_intents(intents)
unique_labels = sorted(list(set(labels)))

# Mapas etiqueta <-> índice para salida softmax
label2idx = {lab: i for i, lab in enumerate(unique_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

# Convertimos las etiquetas a índices enteros
y = np.array([label2idx[l] for l in labels], dtype=np.int32)


# ====================================================
# 4) CAPA DE TEXTUALIZACIÓN: TEXTVECTORIZATION (Keras)
# ====================================================
# Esta capa tokeniza, normaliza y convierte texto a secuencias de enteros.
text_vec = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",  # Minúsculas + elimina puntuación
    split="whitespace",                         # Tokenización por espacio
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

# Adaptamos la capa al corpus actual (puede re-adaptarse con nuevos datos)
text_vec.adapt(np.array(texts))

# Para inspección opcional: vocabulario aprendido
# vocab = text_vec.get_vocabulary()


# ==================================
# 5) DEFINICIÓN DEL MODELO (Keras)
# ==================================
def build_model(num_classes):
    """
    Modelo secuencial:
    - TextVectorization (fuera del modelo, para poder re-adaptar)
    - Embedding
    - Bidirectional LSTM
    - Dense Softmax
    """
    inputs = layers.Input(shape=(MAX_LEN,), dtype=tf.int64, name="input_ids")
    x = layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBED_DIM, name="embed")(inputs)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS), name="bilstm")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="cls")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Creamos y entrenamos el modelo inicial
model = build_model(num_classes=len(unique_labels))

# Transformamos textos -> IDs con la capa text_vec (fuera del grafo)
X = text_vec(np.array(texts))
model.fit(X, y, epochs=EPOCHS_INIT, batch_size=BATCH_SIZE, verbose=0)


# ===========================================
# 6) INFERENCIA: PREDICCIÓN DE INTENCIÓN
# ===========================================
def predict_intent(user_text):
    """
    Devuelve (intent_predicho, confianza, vector_probabilidades),
    usando softmax del modelo.
    """
    seq = text_vec(np.array([user_text.lower().strip()]))
    probs = model.predict(seq, verbose=0)[0]         # vector softmax
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    intent = idx2label[idx]
    return intent, conf, probs


# ===========================================
# 7) RESPUESTA Y LÓGICA DE AUTOAPRENDIZAJE
# ===========================================
def get_response_for_intent(intent):
    """Devuelve una respuesta aleatoria asociada a la intención."""
    candidates = intents[intent]["responses"]
    return random.choice(candidates)

def active_learning_cycle(user_text):
    """
    Si la confianza del modelo es baja:
      - Pide al tutor que etiquete la intención correcta o cree una nueva.
      - (Opcional) Permite registrar una respuesta ejemplo.
      - Actualiza dataset y re-entrena por unas pocas épocas.
    """
    print("\n[Aprendizaje activo] No estoy seguro de la clase correcta.")
    print("Por favor, ingresa la intención correcta para esta frase o escribe 'nueva' para crear una intención:")
    print(f"Entrada del usuario: \"{user_text}\"")
    print(f"Intenciones existentes: {list(intents.keys())}")

    # Etiquetado por el tutor
    etiqueta = input("Etiqueta correcta (o 'nueva'): ").strip().lower()

    if etiqueta == "nueva":
        etiqueta = input("Nombre de la NUEVA intención: ").strip().lower()
        if etiqueta not in intents:
            intents[etiqueta] = {"patterns": [], "responses": []}

    # Guardamos el patrón en la intención elegida
    intents[etiqueta]["patterns"].append(user_text)

    # Permita al tutor agregar una respuesta ejemplo para esa intención
    print("Opcional: agrega una respuesta adecuada para esta intención (enter para saltar):")
    nueva_resp = input("Respuesta: ").strip()
    if nueva_resp:
        intents[etiqueta]["responses"].append(nueva_resp)

    # Persistimos intents en disco
    save_intents(intents)

    # Re-entrenamos rápidamente con el nuevo ejemplo (mini-ajuste)
    incremental_retrain()

    return f"¡Gracias! He aprendido algo nuevo sobre la intención '{etiqueta}'."

def incremental_retrain():
    """
    Re-adapta la capa de texto, recompila el modelo si cambió el #clases,
    y entrena unas pocas épocas adicionales.
    """
    global model, text_vec, unique_labels, label2idx, idx2label

    # Reconstruimos dataset a partir de intents actualizados
    texts, labels = build_dataset_from_intents(intents)
    unique_labels = sorted(list(set(labels)))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    y = np.array([label2idx[l] for l in labels], dtype=np.int32)

    # Re-adaptamos el TextVectorization (vocab cambia al crecer el corpus)
    text_vec = layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LEN
    )
    text_vec.adapt(np.array(texts))
    X = text_vec(np.array(texts))

    # Si cambió el número de clases, reconstruimos el modelo
    if model.output_shape[-1] != len(unique_labels):
        model = build_model(num_classes=len(unique_labels))

    # Entrenamiento incremental corto
    model.fit(X, y, epochs=EPOCHS_INCREMENTAL, batch_size=BATCH_SIZE, verbose=0)


# ===========================================
# 8) BUCLE DE DIÁLOGO (CLI)
# ===========================================
def chat():
    print("Chatbot de autoaprendizaje (escribe 'salir' para terminar).")
    while True:
        user = input("\nTú: ").strip()
        if user.lower() in ["salir", "exit", "quit"]:
            print("Bot: ¡Hasta luego! 👋")
            break

        # Predicción y confianza
        intent, conf, _ = predict_intent(user)
        # Si confianza baja, pedir ayuda y aprender
        if conf < CONFIDENCE_THRESHOLD:
            msg = active_learning_cycle(user)
            print(f"Bot: {msg}")
            continue

        # Si confianza aceptable, responder
        resp = get_response_for_intent(intent)
        print(f"Bot: ({intent}, conf={conf:.2f}) {resp}")

# (Opcional) Guardado del modelo y vocabulario para reutilizar más tarde
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.keras")
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.txt")

def save_all():
    """Guarda el modelo Keras y el vocabulario de TextVectorization."""
    # Guardar modelo
    model.save(MODEL_PATH)
    # Guardar vocabulario del TextVectorization
    vocab = text_vec.get_vocabulary()
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    # Guardar intents ya lo hacemos en cada aprendizaje activo
    save_intents(intents)

def load_vocab_for_textvec(path):
    """Reconstruye un TextVectorization cargando vocabulario desde archivo."""
    vec = layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LEN
    )
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f.readlines() if line.strip()]
        # Asignamos el vocabulario (API privada pero práctica para talleres)
        vec.set_vocabulary(vocab)
    return vec


# =====================
# 9) EJECUCIÓN PRINCIPAL
# =====================
if __name__ == "__main__":
    # Inicio del chat. (Use Ctrl+C para detener y luego save_all() si desea persistir)
    try:
        chat()
    finally:
        # Guardamos todo al salir
        save_all()