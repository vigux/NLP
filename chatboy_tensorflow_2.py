"""
Chatbot que emplea el aprendizaje obtenido previamente (TensorFlow/Keras).
- Carga intents, modelo y vocabulario desde 'chatbot_artifacts/' si existen.
- Si falta algo o cambió el número de clases, reconstruye el modelo y re-entrena desde intents.json.
- Mantiene aprendizaje activo (active learning) para seguir mejorando en tiempo real.

Requisitos:
  - tensorflow>=2.10
  - Artefactos previos en ./chatbot_artifacts/ (intents.json, model.keras, vocab.txt)
    * Si faltan model/vocab, se reconstruye el modelo desde intents.json.
"""

import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ======================
# 1) CONFIGURACIÓN BÁSICA
# ======================
ARTIFACTS_DIR = "chatbot_artifacts"
INTENTS_PATH  = os.path.join(ARTIFACTS_DIR, "intents.json")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "model.keras")
VOCAB_PATH    = os.path.join(ARTIFACTS_DIR, "vocab.txt")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Hiperparámetros
MAX_TOKENS = 10000
MAX_LEN    = 20
EMBED_DIM  = 64
LSTM_UNITS = 64
BATCH_SIZE = 16
EPOCHS_INIT = 15            # para entrenamiento/reentrenamiento completo
EPOCHS_INCREMENTAL = 5      # cuando se agrega un ejemplo nuevo
CONFIDENCE_THRESHOLD = 0.55 # confianza mínima para no activar aprendizaje

# Respuestas de respaldo
FALLBACK_RESPONSES = [
    "No estoy seguro de entender. ¿Podrías reformularlo?",
    "Aún estoy aprendiendo. Intenta con otro enunciado o dame más contexto."
]

# =========================
# 2) UTILIDADES DE INTENTS
# =========================
def load_intents():
    """Carga intents desde disco. Si no existe, crea uno mínimo."""
    if os.path.exists(INTENTS_PATH):
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Intents mínimos si no hay archivo previo
    intents = {
        "saludo": {
            "patterns": ["hola", "buenos días", "buenas tardes", "qué tal", "hey"],
            "responses": ["¡Hola! ¿En qué puedo ayudarte?", "¡Un gusto saludarte! ¿Qué te gustaría saber?"]
        },
        "despedida": {
            "patterns": ["adiós", "hasta luego", "nos vemos", "bye"],
            "responses": ["¡Hasta luego! Que tengas un excelente día.", "¡Nos vemos! Vuelve cuando quieras."]
        },
        "ayuda": {
            "patterns": ["ayuda", "¿qué puedes hacer?", "¿cómo funcionas?", "necesito ayuda"],
            "responses": ["Puedo responder saludos y despedidas, y aprender nuevas intenciones contigo."]
        }
    }
    save_intents(intents)
    return intents

def save_intents(intents):
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(intents, f, ensure_ascii=False, indent=2)

def build_dataset_from_intents(intents):
    texts, labels = [], []
    for intent, data in intents.items():
        for p in data.get("patterns", []):
            texts.append(p.lower().strip())
            labels.append(intent)
    if not texts:
        # Evitar dataset vacío
        texts, labels = ["hola"], ["saludo"]
    return texts, labels

# =========================================
# 3) TEXTVECTORIZATION: CREAR O RECONSTRUIR
# =========================================
def new_text_vectorizer():
    return layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LEN
    )

def load_vocab_for_textvec(path):
    vec = new_text_vectorizer()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
        # API privada pero útil para talleres: asignar vocabulario directamente
        vec.set_vocabulary(vocab)
    return vec

def save_vocab(text_vec, path):
    vocab = text_vec.get_vocabulary()
    with open(path, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(tok + "\n")

# ===========================
# 4) MODELO (BiLSTM + Dense)
# ===========================
def build_model(num_classes: int):
    inputs = layers.Input(shape=(MAX_LEN,), dtype=tf.int64, name="input_ids")
    x = layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBED_DIM, name="embed")(inputs)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS), name="bilstm")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="cls")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ==================================================
# 5) CARGA/RECONSTRUCCIÓN DEL MODELO Y TEXT VEC
# ==================================================
def prepare_model_and_vectorizer(intents):
    texts, labels = build_dataset_from_intents(intents)
    unique_labels = sorted(list(set(labels)))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    y = np.array([label2idx[l] for l in labels], dtype=np.int32)

    # Intento de cargar vocabulario y modelo previos
    if os.path.exists(VOCAB_PATH):
        text_vec = load_vocab_for_textvec(VOCAB_PATH)
    else:
        text_vec = new_text_vectorizer()
        text_vec.adapt(np.array(texts))
        save_vocab(text_vec, VOCAB_PATH)

    X = text_vec(np.array(texts))

    # Intento de cargar el modelo previo:
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            # Si cambió el # de clases, reconstruimos y reentrenamos
            out_dim = model.output_shape[-1]
            if out_dim != len(unique_labels):
                model = build_model(len(unique_labels))
                model.fit(X, y, epochs=EPOCHS_INIT, batch_size=BATCH_SIZE, verbose=0)
                model.save(MODEL_PATH)
        except Exception:
            # Si falla la carga, reconstruimos
            model = build_model(len(unique_labels))
            model.fit(X, y, epochs=EPOCHS_INIT, batch_size=BATCH_SIZE, verbose=0)
            model.save(MODEL_PATH)
    else:
        # No había modelo: entrenar desde intents.json
        model = build_model(len(unique_labels))
        model.fit(X, y, epochs=EPOCHS_INIT, batch_size=BATCH_SIZE, verbose=0)
        model.save(MODEL_PATH)

    return model, text_vec, label2idx, idx2label

# ======================================================
# 6) INFERENCIA, RESPUESTAS Y APRENDIZAJE EN CALIENTE
# ======================================================
def predict_intent(user_text, model, text_vec, idx2label):
    seq = text_vec(np.array([user_text.lower().strip()]))
    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx2label[idx], float(probs[idx]), probs

def get_response(intent, intents):
    return random.choice(intents[intent]["responses"])

def incremental_retrain(intents, model, text_vec):
    """Re-adapta vocabulario, reconstruye si cambia #clases, y entrena unas épocas."""
    texts, labels = build_dataset_from_intents(intents)
    unique_labels = sorted(list(set(labels)))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    y = np.array([label2idx[l] for l in labels], dtype=np.int32)

    # Re-adaptar text_vec al corpus actualizado y persistir vocabulario
    text_vec = new_text_vectorizer()
    text_vec.adapt(np.array(texts))
    save_vocab(text_vec, VOCAB_PATH)
    X = text_vec(np.array(texts))

    # Si cambió el #clases, reconstruir modelo
    if model.output_shape[-1] != len(unique_labels):
        model = build_model(len(unique_labels))

    # Entrenamiento corto incremental
    model.fit(X, y, epochs=EPOCHS_INCREMENTAL, batch_size=BATCH_SIZE, verbose=0)
    model.save(MODEL_PATH)

    return model, text_vec, label2idx, idx2label

def active_learning(user_text, intents, model, text_vec, idx2label):
    print("\n[Aprendizaje activo] No estoy seguro de la intención.")
    print(f"Entrada: \"{user_text}\"")
    print("Intenciones existentes:", list(intents.keys()))
    etiqueta = input("Ingresa la intención correcta o escribe 'nueva' para crear una nueva: ").strip().lower()

    if etiqueta == "nueva":
        etiqueta = input("Nombre de la NUEVA intención: ").strip().lower()
        if etiqueta not in intents:
            intents[etiqueta] = {"patterns": [], "responses": []}

    # Añadimos el patrón
    intents[etiqueta]["patterns"].append(user_text)

    # Añadir una respuesta de ejemplo
    nueva_resp = input("Agrega una respuesta adecuada (Enter para omitir): ").strip()
    if nueva_resp:
        intents[etiqueta]["responses"].append(nueva_resp)

    # Persistir intents
    save_intents(intents)

    # Reentrenamiento incremental
    model, text_vec, label2idx, idx2label = incremental_retrain(intents, model, text_vec)

    return model, text_vec, label2idx, idx2label, f"¡Gracias! He aprendido sobre la intención '{etiqueta}'."

# ==========================
# 7) BUCLE DE DIÁLOGO (CLI)
# ==========================
def chat():
    intents = load_intents()
    model, text_vec, label2idx, idx2label = prepare_model_and_vectorizer(intents)

    print("Chatbot (aprendizaje reutilizado). Escribe 'salir' para terminar.")
    while True:
        user = input("\nTú: ").strip()
        if user.lower() in ["salir", "exit", "quit"]:
            print("Bot: ¡Hasta luego! 👋")
            break

        intent, conf, _ = predict_intent(user, model, text_vec, idx2label)

        if conf < CONFIDENCE_THRESHOLD:
            model, text_vec, label2idx, idx2label, msg = active_learning(
                user, intents, model, text_vec, idx2label
            )
            print("Bot:", msg)
            continue

        # Responder con base en la intención predicha
        try:
            resp = get_response(intent, intents)
            print(f"Bot: ({intent}, conf={conf:.2f}) {resp}")
        except Exception:
            # Si no hay respuestas registradas para esa intención
            print(f"Bot: ({intent}, conf={conf:.2f}) {random.choice(FALLBACK_RESPONSES)}")

if __name__ == "__main__":
    chat()
