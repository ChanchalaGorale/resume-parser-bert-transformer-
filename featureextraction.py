
import spacy
nlp= spacy.load("en-core-web-sm")

from datapreprocess import pre_processed_text


def feat_extract(text):

    doc = nlp(text)

    features = {
        "tokens":[token.text for token in doc],
        "pos_tags":[token.pos_ for token in doc],
        "entities":[( ent.text, ent.label_) for ent in doc.ents]
    }

    return features

features = [feat_extract(t) for t in pre_processed_text]

