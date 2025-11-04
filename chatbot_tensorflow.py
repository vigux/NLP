"""
Chatbot sencillo con TensorFlow/Keras
- Clasifica la intención del usuario y responde con una frase adecuada.
- Enfoque: pipeline minimalista (TextVectorization + Embedding + Pooling + Dense).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# 1) INTENCIONES (DATASET)
# =========================
# Conjunto pequeño de intents para un bot básico.
INTENTS = {
    "saludo": {
        "patterns": [
            "hola", "hola, ¿qué tal?", "buenos días", "buenas tardes", "hey", "qué onda"
        ],
        "responses": [
            "¡Hola! ¿En qué puedo ayudarte?",
            "¡Un gusto saludarte! ¿Qué te gustaría saber?"
        ]
    },
    "ayuda": {
        "patterns": [
            "¿qué puedes hacer?", "ayuda", "necesito ayuda", "¿cómo funcionas?", "no sé qué preguntar"
        ],
        "responses": [
            "Puedo responder saludos, despedidas y dudas básicas. ¡Inténtalo!",
            "Puedo orientarte de manera general. Por ejemplo: salúdame o despídete."
        ]
    },
    "agradecimiento": {
        "patterns": [
            "gracias", "muchas gracias", "te lo agradezco", "mil gracias"
        ],
        "responses": [
            "¡Con gusto! ¿Algo más en lo que te pueda apoyar?",
            "Para servirte. ¿Deseas preguntar otra cosa?"
        ]
    },
    "despedida": {
        "patterns": [
            "adiós", "hasta luego", "nos vemos", "bye", "me voy"
        ],
        "responses": [
            "¡Hasta luego! Que tengas un excelente día.",
            "¡Nos vemos! Vuelve cuando quieras."
        ]
    }
}

FALLBACK_RESPONSES = [
    "No estoy seguro de entender. ¿Podrías reformularlo?",
    "Aún estoy aprendiendo. Intenta con: 'hola', 'ayuda', 'gracias' o 'adiós'."
]

# =====================================
# 2) CONSTRUCCIÓN DEL CONJUNTO DE DATOS
# =====================================
def build_dataset(intents_dict):
    """Crea listas X (textos) e y (etiquetas) a partir de INTENTS."""
    texts, labels = [], []
    for intent, data in intents_dict.items():
        for p in data["patterns"]:
            texts.append(p.lower().strip())
            labels.append(intent)
    return texts, labels

texts, labels = build_dataset(INTENTS)

# Creamos mapas etiqueta <-> índice
unique_labels = sorted(list(set(labels)))
label2idx = {lab: i for i, lab in enumerate(unique_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
y = np.array([label2idx[l] for l in labels], dtype=np.int32)

# ======================================
# 3) TEXTVECTORIZATION (TOKENIZACIÓN/ID)
# ======================================
MAX_TOKENS = 10000          # vocabulario máximo
MAX_LEN = 16                # longitud máxima por frase

text_vec = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",  # minúsculas + sin puntuación
    split="whitespace",
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN
)
text_vec.adapt(np.array(texts))                 # aprende vocabulario del dataset

X = text_vec(np.array(texts))                   # textos -> secuencias de IDs

# ============================
# 4) MODELO (Keras, muy simple)
# ============================
# Arquitectura minimalista:
# Embedding -> GlobalAveragePooling1D -> Dense(hidden) -> Dense(softmax)
EMBED_DIM = 64

inputs = layers.Input(shape=(MAX_LEN,), dtype=tf.int64, name="input_ids")
x = layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBED_DIM, name="embed")(inputs)
x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(unique_labels), activation="softmax", name="cls")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento rápido (dataset pequeño)
model.fit(X, y, epochs=30, batch_size=8, verbose=0)

# =====================================
# 5) PREDICCIÓN + RESPUESTA DEL CHATBOT
# =====================================
CONFIDENCE_THRESHOLD = 0.6  # si la confianza es menor, usamos FALLBACK

def predict_intent(user_text: str):
    """Retorna (intent, confianza)."""
    seq = text_vec(np.array([user_text.lower().strip()]))
    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx2label[idx], float(probs[idx])

def get_bot_response(intent: str) -> str:
    """Elige una respuesta aleatoria asociada a la intención."""
    import random
    return random.choice(INTENTS[intent]["responses"])

# ============================
# 6) BUCLE DE DIÁLOGO EN CONSOLA
# ============================
def chat():
    print("Chatbot TensorFlow (sencillo). Escribe 'salir' para terminar.")
    while True:
        user = input("\nTú: ").strip()
        if user.lower() in ["salir", "exit", "quit"]:
            print("Bot: ¡Hasta luego! 👋")
            break

        intent, conf = predict_intent(user)
        if conf < CONFIDENCE_THRESHOLD:
            # Respuesta de reserva si el modelo no está seguro
            print(f"Bot: {np.random.choice(FALLBACK_RESPONSES)}")
        else:
            # Respuesta asociada a la intención detectada
            print(f"Bot: ({intent}, conf={conf:.2f}) {get_bot_response(intent)}")

if __name__ == "__main__":
    chat()