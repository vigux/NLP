<<<<<<< HEAD
from textblob import TextBlob
# Sample text
text = "I love this product! It's amazing."
# Create a TextBlob object
blob = TextBlob(text)
# Perform sentiment analysis
sentiment = blob.sentiment
=======
from textblob import TextBlob
# Sample text
text = "I love this product! It's amazing."
# Create a TextBlob object
blob = TextBlob(text)
# Perform sentiment analysis
sentiment = blob.sentiment
>>>>>>> 50084c9784e74406b7e9c27c7b8e1690a2597b34
print(sentiment)  # Output: Sentiment(polarity=0.65, subjectivity=0.6)