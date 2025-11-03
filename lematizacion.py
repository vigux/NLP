import spacy
nlp = spacy.load("es_core_news_sm")

doc = nlp("Los estudiantes están programando aplicaciones inteligentes.")
for token in doc:
    print(token.text, "→", token.lemma_)