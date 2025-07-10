from textblob import TextBlob

def get_emoji(sentiment):
    if sentiment == 'positive':
        return '😊'
    elif sentiment == 'negative':
        return '😠'
    else:
        return '😐'

def analyze_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.2:
        return "positive"
    elif blob.sentiment.polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def convert_to_literal(text):
    examples = {
        "Oh great, another Monday.": "I dislike Mondays.",
        "I just love getting ignored.": "I feel bad being ignored.",
        "Thank you for actually doing your job!": "You were expected to do this.",
        "My life is a comedy show.": "My life feels ridiculous.",
        "Nice job breaking it, hero.": "You made a mistake.",
    }
    return examples.get(text, "This might mean the opposite of what is said.")
