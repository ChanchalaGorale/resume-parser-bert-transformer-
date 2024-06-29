import re
import spacy
nlp= spacy.load("en-core-web-sm")


from datacollection import resume

def process_text(text):

    text = re.sub(r'\s+', " ", text)

    text = text.lower()

    doc= nlp(text)

    tokens= [token.text for token in doc]

    return tokens

pre_processed_text = [process_text(t) for t in resume]


