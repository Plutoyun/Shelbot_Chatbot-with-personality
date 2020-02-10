from flask import Flask, request
from flask import render_template
import numpy as np
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras import preprocessing, utils
import os
import yaml
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

# web deployment using flask
app = Flask(__name__)
app.config['DEBUG'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# model handler
rootPath = './model/SheldonTrainLargeData/'
print(os.getcwd() + '11111111')
q = open(rootPath + "context_sheldon.txt", 'r', encoding='UTF-8')
questions = q.read()
a = open(rootPath + "answers_sheldon.txt", 'r', encoding='UTF-8')
answers = a.read()
# print(type(questions),"\n",answers)
all = answers + questions
answers = [p for p in answers.split('\n')]
questions = [p for p in questions.split('\n')]
all_text = [p for p in all.split('\n')]
# print(questions[:2],"\n",answers[:2],"\n",all_text)

# questions = questions[:2000]
# answers = answers[:2000]

answers_with_tags = list()
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)

answers = list()
for i in range(len(answers_with_tags)):
    answers.append('<START> ' + answers_with_tags[i] + ' <END>')

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1
print('VOCAB SIZE : {}'.format(VOCAB_SIZE))

print(questions[:5], "\n", answers[:5], "\n", len(questions) == len(answers))

# encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])

tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])

tokenized_answers_output = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers_output)):
    tokenized_answers_output[i] = tokenized_answers_output[i][1:]


def str_to_tokens(sentence: str):
    print(sentence)
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    print(preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post'))
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


#
# global graph
# graph = tf.get_default_graph()
# g = tf.Graph()
# g2 = tf.Graph()
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

# with g.as_default():
enc_model = load_model(rootPath + 'modelLarge2enc.h5')
# with g2.as_default():
dec_model = load_model(rootPath + 'modelLarge2dec.h5')


# model predict handler
def handlePredict(input):
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        print(enc_model)
        # with g.as_default():
        states_values = enc_model.predict(str_to_tokens(input))
        print(states_values)
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            # with g2.as_default():
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]
    return decoded_translation[:-3]


@app.route('/')  # , methods=['GET', 'POST'])
def home():
    return render_template('hello.html')


@app.route('/get')  # ,methods=['GET', 'POST'])
def predict():
    userInput = request.args.get('msg')
    print("question: " + userInput)
    return handlePredict(userInput)

    # if request.method == "POST":
    #     userInput = request.form ['userInput']
    # answer = request.args.get('answer')
    # return render_template('hello.html', answer=userInput)


if __name__ == "__main__":
    app.run()
# activate environment: venv\Scripts\activate
# set FLASK_APP=test.py
# flask run
