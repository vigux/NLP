<<<<<<< HEAD
#Importar el módulo Translator de la biblioteca translate
from translate import Translator

# Crear un objeto translator especificando el idioma destino
translator = Translator(to_lang="es")
# Traducción de una frase
translation = translator.translate("How are you?")
=======
#Importar el módulo Translator de la biblioteca translate
from translate import Translator

# Crear un objeto translator especificando el idioma destino
translator = Translator(to_lang="es")
# Traducción de una frase
translation = translator.translate("How are you?")
>>>>>>> 50084c9784e74406b7e9c27c7b8e1690a2597b34
print(translation)  # Output: ¿Cómo estás?