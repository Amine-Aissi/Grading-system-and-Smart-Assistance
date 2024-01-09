import tokenize
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as keras
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from typing import List
import stanza
import strawberry
from strawberry.asgi import GraphQL
from fastapi.middleware.cors import CORSMiddleware


# Load the tokenizer and models
tokenizer = Tokenizer()
tokenizer.fit_on_texts([""])  # Empty initialization to avoid errors
max_sequence_length =  22 # Set the appropriate max_sequence_length value here



@strawberry.type
class Question:
    question_id: int
    question: str


@strawberry.type
class Answer:
    id: int
    qid: int
    answer: str



@strawberry.type
class Grade:
    question_id: int
    answer: str
    grade: int


@strawberry.type
class Mutation:
    @strawberry.mutation
    def recover_answer(self, question_id: int, answer: str) -> Grade:
        grade = predict(question_id, answer)
        new_answer = Grade(question_id=question_id, answer=answer, grade=grade)
        return new_answer


@strawberry.type
class Query:
    @strawberry.field
    def questions(self) -> List[Question]:
        return [
            Question(
                question_id=1,
                question="متى ولد رسول الله صلّى الله عليه وسلَّم؟",
            ),
            Question(
                question_id=2,
                question="من ظهر للنبي محمد في أول وحي له؟",
            ),
            Question(
                question_id=3, 
                question="من هي أم الرسول صلَّى الله عليه وسلَّم؟"
                ),
            Question(
                question_id=4, 
                question="أين ولد رسول الله صلَّى الله عليه وسلَّم؟"
            ),
            Question(
                question_id=5, 
                question="من كان أول مؤذن ؟"
                ),
            Question(
                question_id=6, question="من هي أول امرأة أسلمت ؟"
            ),
            Question(
                question_id=7,
                question="أين نزل الوحي على الرسول صلَّى الله عليه وسلَّم؟",
            ),
            Question(
                question_id=8, 
                question="من هو عم الرسول صلى الله عليه وسلم كفل تربية النبي ورعايته بعد وفاة جده؟"
            ),
            Question(
                question_id=9,
                question="ما أول سورة في الجزء الثلاثين من القرآن؟",
            ),
            Question(
                question_id=10,
                question="ﻣﺎ ﻫﻲ ﺍﻟﺴﻮﺭﺓ ﺍﻟﺘﻲ ﻧﺰﻟﺖ ﺑﻜﺎﻣﻠﻬﺎ؟"),
        ]

    @strawberry.field
    def answers(self) -> List[Answer]:
        return [
            Answer(
                id=1,
                qid=1,
                answer="ولد رسول الله صلَّى الله عليه وسلَّم في عام الفيل 571م",
            ),
            Answer(
                id=2,
                qid=2,
                answer="ظهر جبريل عليه السلام للنبي محمد في الوحي الأول",
            ),
            Answer(
                id=3,
                qid=3,
                answer="أم الرسول هي آمنة بنت وهب من بني زهرة",
            ),
            Answer(
                id=4,
                qid=4,
                answer="ولد رسول الله عليه الصَّلاة والسَّلام في مكة المكرمة في قبيلة قريش",
            ),
            Answer(
                id=5,
                qid=5,
                answer="أول مؤذنٍ في الإسلام بلال بن رباح رضي الله عنه",
            ),
            Answer(
                id=6,
                qid=6,
                answer="لأولى التي أسلمت إلى الإسلام كانت خديجة بنت خويلد (رضي الله عنها)",
            ),
            Answer(
                id=7,
                qid=7,
                answer="نزل الوحي على الرسول صلَّى الله عليه وسلَّم في غار حراء",
            ),
            Answer(
                id=8,
                qid=8,
                answer="عم الرسول صلى الله عليه وسلم الذي كفل تربية النبي ورعايته بعد وفاة جده هو أبي طالب، وهو عبد مناف بن عبد الطلب",
            ),
            Answer(
                id=9,
                qid=9,
                answer="أول سورة في الجزء الثلاثين هي سورة النبأ",
            ),
            Answer(
                id=10, 
                qid=10,
                answer="ﺍﻟﺴﻮﺭﺓ ﺍﻟﺘﻲ ﻧﺰﻟﺖ ﺑﻜﺎﻣﻠﻬﺎ ﻫﻲ ﺳﻮﺭﺓ ﺍﻟﻤﺪﺛﺮ"
                ),
        ]


schema = strawberry.Schema(query=Query, mutation=Mutation)

graphql_app = GraphQL(schema)

app = FastAPI()



rnn_model = load_model('C:\\Users\\amine\\OneDrive\\Bureau\\DeepLearning\\models\\savedModels\\q1_model.h5')

# Load the Arabic NLP pipeline
stanza.download('ar')   
nlp = stanza.Pipeline('ar')


class Answer(BaseModel): 
    answer: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # You can specify specific HTTP methods
    allow_headers=["*"],  # You can specify specific HTTP headers
)

@app.post("/predict/")
async def predict_grade(answer: Answer):
    return predict(1, answer.answer)


def preprocess_text(text):
    doc = nlp(text)
    tokens = [word.lemma for sent in doc.sentences for word in sent.words if word.upos != 'PUNCT']
    return tokens

app.add_route("/graphql", graphql_app)


def predict(qId, answer):
    # Preprocess the input answer
    print(answer)
    model = load_model("C:\\Users\\amine\\OneDrive\\Bureau\\DeepLearning\\models\\savedModels\\q" + str(qId) + "_model.h5")
    ls = preprocess_text(answer)
    tn = []
    tn.append(ls)
    print(tn)
    # Preprocess the Arabic sentence
    tokenizert = Tokenizer()
    tokenizert.fit_on_texts(tn)
    sequencest = tokenizert.texts_to_sequences(tn)
    sequencest = pad_sequences(sequencest, 22)

    # Make predictions
    data = pad_sequences(sequencest, padding='post', truncating='post', maxlen = 22)

    prediction = model.predict(data)
    # Convert predictions to class indices
    rnn_class = int(np.argmax(prediction, axis=1)[0])

    return {"rnn_prediction": rnn_class}