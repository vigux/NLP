import nltk
nltk.download('punkt')
# nltk.download('all')

#Función para dividir el texto en palabras
from nltk.tokenize import word_tokenize

# Texto de ejemplo
texto = "La inteligencia artificial está transformando el mundo."

# Tokenización del texto mediante la función word_tokenize
tokens = word_tokenize(texto)

# Impresión de la lista de tokens
print(tokens)