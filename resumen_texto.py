import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

#nltk.download('punkt') #https://www.nltk.org/nltk_data/
nltk.download('stopwords')

# Sample text
text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence
concerned with the interactions between computers and human language, in particular how to program computers to process
and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech
recognition, natural language understanding, and natural language generation."""

# Preprocess the text
sentences = sent_tokenize(text)
stop_words = set(stopwords.words('english'))

def preprocess_sentence(sentence):
    words = word_tokenize(sentence.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

# Sentence scoring based on term frequency
def score_sentences(sentences):
    sentence_scores = []
    word_frequencies = FreqDist([word for sentence in sentences for word in preprocess_sentence(sentence)])

    for sentence in sentences:
        words = preprocess_sentence(sentence)
        sentence_score = sum(word_frequencies[word] for word in words)
        sentence_scores.append((sentence, sentence_score))

    return sentence_scores

# Select top-ranked sentences
def select_sentences(sentence_scores, num_sentences=2):
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    selected_sentences = [sentence[0] for sentence in sentence_scores[:num_sentences]]
    return selected_sentences

# Generate summary
sentence_scores = score_sentences(sentences)
summary_sentences = select_sentences(sentence_scores)
summary = ' '.join(summary_sentences)

print("Summary:")
print(summary)