from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text
text = "Natural Language Processing enables computers to understand human language."

# Tokenize the text
tokens = text.split()

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize the tokens
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

print("Original Tokens:")
print(tokens)

print("\\nLemmatized Tokens:")
print(lemmatized_tokens)