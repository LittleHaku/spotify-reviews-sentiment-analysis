import numpy as np
import pandas as pd
import string
import re
import tensorflow as tf
from tensorflow.keras import layers
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from termcolor import colored

model_bow = load_model('model_bow.keras')
model_ids = load_model('model_ids.keras')

vocab_df = pd.read_csv('vocab.csv')
vocab = vocab_df['word'].values

abbreviations = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My Ass Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My Ass Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The Fuck",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don’t care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can’t stop laughing"
}

def replace_abbreviation(text):
    result = [abbreviations.get(word.upper(), word) for word in text.split()]
    return " ".join(result)

def clean_text(sentence):
    pattern = r"[^a-zA-Z0-9\s]"
    clean = re.sub(pattern, "", sentence)
    spaces_pattern = r"\s+"
    return re.sub(spaces_pattern, " ", clean).lower()

def tokenize_text(text):
    return text.split()

def remove_stopwords(words):
    stop_words = stopwords.words('english')
    result = [word for word in words if word not in stop_words]
    return result

def basic_stemmer(words):
    stemmed_words = []
    for word in words:
        original = word
        word = re.sub(r'ing$', '', word)
        word = re.sub(r'ed$', '', word)
        word = re.sub(r'es$', '', word)
        word = re.sub(r's$', '', word)
        word = re.sub(r'ly$', '', word)
        word = re.sub(r'ment$', '', word)
        word = re.sub(r'able$', '', word)
        word = re.sub(r'ful$', '', word)
        word = re.sub(r'ness$', '', word)

        if word[-1:] == 'i':
            word = word[:-1] + 'y'

        if len(word) > 1 and word[-1] == word[-2] and word[-1] not in 'aeiou':
            word = word[:-1]

        if len(word) <= 2:
            word = original

        stemmed_words.append(word)

    return stemmed_words

def preprocess_sentences(sentences):
    mod_sentences = [replace_abbreviation(sentence) for sentence in sentences]
    mod_sentences = [clean_text(sentence) for sentence in mod_sentences]
    mod_sentences = [tokenize_text(sentence) for sentence in mod_sentences]
    mod_sentences = [remove_stopwords(sentence) for sentence in mod_sentences]
    mod_sentences = [basic_stemmer(sentence) for sentence in mod_sentences]
    return mod_sentences

def create_bow(sentences, vocab):
    word_to_index = {word: i for i, word in enumerate(vocab)}
    bow_matrix = np.zeros((len(sentences), len(vocab)))

    for i, sentence in enumerate(sentences):
        word_counts = defaultdict(int)
        for word in sentence:
            if word in word_to_index:
                word_counts[word] += 1

        for word, count in word_counts.items():
            bow_matrix[i, word_to_index[word]] = count

    return bow_matrix


def get_color(prediction):
    red = int(255 * (1 - prediction))
    green = int(255 * prediction)
    return f'\033[38;2;{red};{green};0m'

while True:
    sentence = input("Enter a sentence (or type 'exit' to quit): ")
    if sentence.lower() == 'exit':
        break

    pre_sentence = preprocess_sentences([sentence])
    sentence_bow = create_bow(pre_sentence, vocab)
    predictions_bow = model_bow.predict(sentence_bow)
    prediction = predictions_bow[0][0]
    color = get_color(prediction)
    print(f"{color}The sentence is {'positive' if prediction > 0.5 else 'negative'} ({prediction:.2f})\033[0m")
