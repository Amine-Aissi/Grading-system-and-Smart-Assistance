import tokenize
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import stanza


def predict(qId, answer):
    # Preprocess the input answer
    print(answer)
    model = load_model("C:\\Users\\amine\\OneDrive\\Bureau\\DeepLearning\\models\\savedModels\\q" + str(qId) + "_model.h5")
    ls = preprocess_text(answer)
    tn = []
    tn.append(ls)
    print(tn)
    # Preprocess the Arabic sentence
    tokenizert = tokenize()
    tokenizert.fit_on_texts(tn)
    sequencest = tokenizert.texts_to_sequences(tn)
    sequencest = pad_sequences(sequencest, 22)

    # Make predictions
    data = pad_sequences(sequencest, padding='post', truncating='post', maxlen = 22)

    prediction = model.predict(data)
    # Convert predictions to class indices
    rnn_class = int(np.argmax(prediction, axis=1)[0])

    return {"rnn_prediction": rnn_class}

# Load the Arabic NLP pipeline
stanza.download('ar')   
nlp = stanza.Pipeline('ar')

def preprocess_text(text):
    doc = nlp(text)
    tokens = [word.lemma for sent in doc.sentences for word in sent.words if word.upos != 'PUNCT']
    return tokens