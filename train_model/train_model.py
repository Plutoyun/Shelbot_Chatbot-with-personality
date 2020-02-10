import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers , activations , models , preprocessing
from tensorflow.keras import preprocessing , utils
import os
import yaml

rootPath = '../model/SheldonTrainLargeData/'

q = open(rootPath+"context_sheldon.txt", 'r', encoding='UTF-8')
questions = q.read()
a = open(rootPath+"answers_sheldon.txt", 'r', encoding='UTF-8')
answers = a.read()
# print(type(questions),"\n",answers)
all = answers + questions
answers = [p for p in answers.split('\n')]
questions = [p for p in questions.split('\n')]
all_text = [p for p in all.split('\n')]
print(questions[:2],"\n",answers[:2],"\n",all_text)

# questions = questions[:2000]
# answers = answers[:2000]

answers_with_tags = list()
for i in range( len( answers ) ):
    if type( answers[i] ) == str:
        answers_with_tags.append( answers[i] )
    else:
        questions.pop( i )

answers = list()
for i in range( len( answers_with_tags ) ) :
    answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( questions + answers )
VOCAB_SIZE = len( tokenizer.word_index )+1
print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))

print(questions[:5],"\n",answers[:5],"\n",len(questions)==len(answers))



# encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences( questions )
maxlen_questions = max( [ len(x) for x in tokenized_questions ] )


# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )


# decoder_output_data
tokenized_answers_output = tokenizer.texts_to_sequences( answers )
for i in range(len(tokenized_answers_output)) :
    tokenized_answers_output[i] = tokenized_answers_output[i][1:]


encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )

decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax )
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()


def gen(questions,answers):
    print('generator initiated')
    #Define a batch size
    batch_size = 50

    #Complete length of data
    data_size = len(questions)

    #Total number of batches will be created
    num_batches = int(data_size / batch_size)


    if (num_batches*batch_size) < data_size:
         num_batches += 1
    while True:
        cnt=0
        for i in range(num_batches):
            questions_p = []
            start_index = cnt * batch_size
            end_index = min((cnt + 1) * batch_size, data_size)
            cnt +=1

            padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions[start_index:end_index] , maxlen=maxlen_questions , padding='post' )
            encoder_input_data = np.array( padded_questions )

            padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers[start_index:end_index] , maxlen=maxlen_answers , padding='post' )
            decoder_input_data = np.array( padded_answers )

            padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers_output[start_index:end_index] , maxlen=maxlen_answers , padding='post' )
            onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE )
            decoder_output_data = np.array( onehot_answers )
            yield ([encoder_input_data , decoder_input_data], decoder_output_data)


model.fit_generator(gen(questions,answers),steps_per_epoch =int(len(questions)/50),epochs=300)
model.save(rootPath+ 'newModel.h5' )


# for first genarated model
def make_inference_models():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


enc_model , dec_model = make_inference_models()
# enc_model , dec_model = make_inference_models_read()
enc_model.save(rootPath+ 'modelLarge3_150_enc.h5' )
dec_model.save(rootPath+ 'modelLarge3_150_dec.h5' )






