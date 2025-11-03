<<<<<<< HEAD
from transformers import pipeline

# Modelo multilingüe con etiquetas de 1 a 5 estrellas
analizador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

texto = "Me gusto este producto; la calidad es buena y el envío fue muy rápido."
resultado = analizador(texto)[0]

print("Texto:", texto)
print("Etiqueta del modelo:", resultado['label'])  # '5 stars', '4 stars', etc.
print("Confianza:", round(resultado['score'], 3))

# Normalización a positivo/neutral/negativo (opcional)
label = resultado['label']
if label in ["4 stars", "5 stars"]:
    etiqueta = "positivo"
elif label in ["1 star", "2 stars"]:
    etiqueta = "negativo"
else:
    etiqueta = "neutro"
=======
from transformers import pipeline

# Modelo multilingüe con etiquetas de 1 a 5 estrellas
analizador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

texto = "Me gusto este producto; la calidad es buena y el envío fue muy rápido."
resultado = analizador(texto)[0]

print("Texto:", texto)
print("Etiqueta del modelo:", resultado['label'])  # '5 stars', '4 stars', etc.
print("Confianza:", round(resultado['score'], 3))

# Normalización a positivo/neutral/negativo (opcional)
label = resultado['label']
if label in ["4 stars", "5 stars"]:
    etiqueta = "positivo"
elif label in ["1 star", "2 stars"]:
    etiqueta = "negativo"
else:
    etiqueta = "neutro"
>>>>>>> 50084c9784e74406b7e9c27c7b8e1690a2597b34
print("Etiqueta (simplificada):", etiqueta)