import argparse
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dense  
from tensorflow.keras.optimizers import Adam

from data import load_data

TRAIN_PATH = "hi.translit.sampled.train.tsv"
VAL_PATH = "hi.translit.sampled.dev.tsv"
TEST_PATH = "hi.translit.sampled.test.tsv"


#loading training , testing and validation data
train_texts, train_target_texts = load_data(TRAIN_PATH)
val_texts, val_target_texts = load_data(VAL_PATH)
test_texts, test_target_texts = load_data(TEST_PATH)
print("Number of training samples: ", len(train_texts))
print("Number of validation samples: ", len(val_texts))
print("Number of testing samples: ", len(test_texts))

train_indices = np.arange(len(train_texts))
val_indices = np.arange(len(val_texts))
test_indices = np.arange(len(test_texts))

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

# Used to store vocabulary of source and target language
input_characters = set()
target_characters = set()

# Used to store texts after adding start and end token
train_target_texts_processed = []
val_target_texts_processed = []
test_target_texts_processed = []

# Adding starting and ending token in training data
for (input_text, target_text) in zip(train_texts, train_target_texts):
    # "S" -> start token, "E" -> end token, " " -> pad token
    target_text = "S" + target_text + "E"
    train_target_texts_processed.append(target_text)
    for char in input_text:
        input_characters.add(char)
    for char in target_text:
        target_characters.add(char)

# Adding starting and ending token in validation data
for (input_text, target_text) in zip(val_texts, val_target_texts):
    # "S" -> start token, "E" -> end token, " " -> pad token
    target_text = "S" + target_text + "E"
    val_target_texts_processed.append(target_text)
    for char in input_text:
        input_characters.add(char)
    for char in target_text:
        target_characters.add(char)

# Adding starting and ending token in testing data
for (input_text, target_text) in zip(test_texts, test_target_texts):
    # "S" -> start token, "E" -> end token, " " -> pad token
    target_text = "S" + target_text + "E"
    test_target_texts_processed.append(target_text)
    for char in input_text:
        input_characters.add(char)
    for char in target_text:
        target_characters.add(char)


input_texts = list(map(train_texts.__getitem__, train_indices))
target_texts = list(map(train_target_texts_processed.__getitem__, train_indices))

val_input_texts = list(map(val_texts.__getitem__, val_indices))
val_target_texts = list(map(val_target_texts_processed.__getitem__, val_indices))

test_input_texts = list(map(test_texts.__getitem__, test_indices))
test_target_texts = list(map(test_target_texts_processed.__getitem__, test_indices))

# Creating sorted vocabulary of source and target language
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

# Add pad tokens
input_characters.insert(0, " ")
target_characters.insert(0, " ")

# Creating essential parameters
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(t) for t in input_texts])
max_decoder_seq_length = max([len(t) for t in target_texts])
val_max_encoder_seq_length = max([len(t) for t in val_input_texts])
val_max_decoder_seq_length = max([len(t) for t in val_target_texts])

test_max_encoder_seq_length = max([len(t) for t in test_input_texts])
test_max_decoder_seq_length = max([len(t) for t in test_target_texts])

# Mapping each character of vocabulary to index
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# defining shapes of input sequence of encoder after padding for training data
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="float32")

# defining shapes of input and target sequence of decoder after padding for training data
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

# Adding training data
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t] = input_token_index[char]
    encoder_input_data[i, t+1 :] = input_token_index[" "]

    for t, char in enumerate(target_text):
        decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            decoder_target_data[i, t-1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1: ] = target_token_index[" "]
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# defining shapes of input sequence of encoder after padding for validation data
val_encoder_input_data = np.zeros((len(input_texts), val_max_encoder_seq_length), dtype="float32")

# defining shapes of input and target sequence of decoder after padding for validation data
val_decoder_input_data = np.zeros((len(input_texts), val_max_decoder_seq_length), dtype="float32")
val_decoder_target_data = np.zeros((len(input_texts), val_max_decoder_seq_length, num_decoder_tokens), dtype="float32")

# Adding validation data
for i, (input_text, target_text) in enumerate(zip(val_input_texts, val_target_texts)):
    for t, char in enumerate(input_text):
        val_encoder_input_data[i, t] = input_token_index[char]
    val_encoder_input_data[i, t + 1 :] = input_token_index[" "]

    for t, char in enumerate(target_text):
        val_decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            val_decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    val_decoder_input_data[i, t + 1: ] = target_token_index[" "]
    val_decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# defining shapes of input sequence of encoder after padding for testing data
test_encoder_input_data = np.zeros((len(input_texts), test_max_encoder_seq_length), dtype="float32")

# defining shapes of input and target sequence of decoder after padding for testing data
test_decoder_input_data = np.zeros((len(input_texts), test_max_decoder_seq_length), dtype="float32")
test_decoder_target_data = np.zeros((len(input_texts), test_max_decoder_seq_length, num_decoder_tokens), dtype="float32")

# Adding testing data
for i, (input_text, target_text) in enumerate(zip(test_input_texts, test_target_texts)):
    for t, char in enumerate(input_text):
        test_encoder_input_data[i, t] = input_token_index[char]
    test_encoder_input_data[i, t + 1 :] = input_token_index[" "]

    for t, char in enumerate(target_text):
        test_decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            test_decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    test_decoder_input_data[i, t + 1: ] = target_token_index[" "]
    test_decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# creating inverse map which maps integer to character
inverse_input_token_index = dict((i, char) for char, i in input_token_index.items())
inverse_target_token_index = dict((i, char) for char, i in target_token_index.items())


class TransliterationModel(object):
    
    def __init__(self, config):
        self.config = config

    def train_and_evaluate(self, encoder_input_data, decoder_input_data, decoder_target_data,
                         val_encoder_input_data, val_target_texts):
        # Encoder
        encoder_inputs = Input(shape=(None, ),name = 'Encoder_inputs')

        # Embedding layer: (num_encoder_tokens, input_embedding_size)
        encoder_embedded =  Embedding(num_encoder_tokens, self.config.input_embedding_size,
                            mask_zero=True, name='Encoder_embeddings')(encoder_inputs)
        encoder_outputs = encoder_embedded

        # Adding encoder layers and storing encoder states according to cell type
        if self.config.cell_type == 'RNN':
            encoder_layers = [SimpleRNN(self.config.hidden_units, 
                                  dropout=self.config.dropout, 
                                  return_sequences=True, 
                                  return_state=True, 
                                  name=f"Encoder_{layer_idx}")
                        for layer_idx in range(self.config.num_encoder_layers)]
            encoder_outputs, hidden = encoder_layers[0](encoder_outputs)
            encoder_states = [hidden]
            for layer_idx in range(1, self.config.num_encoder_layers):
                encoder_outputs, hidden = encoder_layers[layer_idx](encoder_outputs, initial_state=encoder_states)
                encoder_states = [hidden]  
        elif self.config.cell_type == 'LSTM':
            encoder_layers = [LSTM(self.config.hidden_units, 
                                    dropout=self.config.dropout, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    name=f"Encoder_{layer_idx}")
                                for layer_idx in range(self.config.num_encoder_layers)]
            encoder_outputs, hidden, context = encoder_layers[0](encoder_outputs)
            encoder_states = [hidden, context]
            for layer_idx in range(1, self.config.num_encoder_layers):
                encoder_outputs, hidden, context = encoder_layers[layer_idx](encoder_outputs, initial_state=encoder_states)
                encoder_states = [hidden, context]
        elif self.config.cell_type == 'GRU':
            encoder_layers = [GRU(self.config.hidden_units, 
                                    dropout=self.config.dropout, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    name=f"Encoder_{layer_idx}")
                                for layer_idx in range(self.config.num_encoder_layers)]
            encoder_outputs, hidden = encoder_layers[0](encoder_outputs)
            encoder_states = [hidden]
            for layer_idx in range(1, self.config.num_encoder_layers):
                encoder_outputs, hidden = encoder_layers[layer_idx](encoder_outputs, initial_state=encoder_states)
                encoder_states = [hidden]

        # Decoder
        decoder_inputs = Input(shape=(None,), name = 'Decoder_inputs')

        # Embedding layer: (num_decoder_tokens, hidden_units)
        decoder_embedded = Embedding(num_decoder_tokens, self.config.hidden_units,
                        mask_zero=True, name='Decoder_embeddings')(decoder_inputs)
        decoder_outputs = decoder_embedded

        # Adding decoder layers and storing decoder states according to cell type
        if self.config.cell_type == 'RNN':
            decoder_layers = [SimpleRNN(self.config.hidden_units, 
                                        dropout=self.config.dropout, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        name=f"Decoder_{layer_idx}")
                                for layer_idx in range(self.config.num_decoder_layers)]
            decoder_outputs, _ = decoder_layers[0](decoder_outputs, initial_state=encoder_states)
            for layer_idx in range(1, self.config.num_decoder_layers):
                decoder_outputs, _ = decoder_layers[layer_idx](decoder_outputs, initial_state = encoder_states)
        if self.config.cell_type == 'LSTM':
            decoder_layers = [LSTM(self.config.hidden_units, 
                                    dropout=self.config.dropout, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    name=f"Decoder_{layer_idx}")
                                for layer_idx in range(self.config.num_decoder_layers)]
            decoder_outputs, _, _ = decoder_layers[0](decoder_outputs, initial_state=encoder_states)
            for layer_idx in range(1, self.config.num_decoder_layers):
                decoder_outputs, _, _ = decoder_layers[layer_idx](decoder_outputs, initial_state = encoder_states)
        elif self.config.cell_type == 'GRU':
            decoder_layers = [GRU(self.config.hidden_units, 
                                    dropout=self.config.dropout, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    name=f"Decoder_{layer_idx}")
                                for layer_idx in range(self.config.num_decoder_layers)]
            decoder_outputs, _ = decoder_layers[0](decoder_outputs, initial_state=encoder_states)
            for layer_idx in range(1, self.config.num_decoder_layers):
                decoder_outputs, _ = decoder_layers[layer_idx](decoder_outputs, initial_state=encoder_states)
            decoder_outputs = Dense(num_decoder_tokens, activation='softmax', name='dense')(decoder_outputs)

        # Defining our Seq2seq model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        optimizer = Adam(learning_rate=self.config.learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=['accuracy'])
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data)
        )
    
        model.save("transliteration_model")
    
        # Wrap Encoder Decoder
        encoder_inputs = model.input[0]
        if self.config.cell_type in ['RNN', 'GRU']:
            encoder_outputs, hidden_state = model.get_layer(f'Encoder_{self.config.num_encoder_layers-1}').output
            encoder_states = [hidden_state]
            encoder = Model(encoder_inputs, encoder_states)
            decoder_inputs = model.input[1]
            decoder_outputs = model.get_layer('Decoder_embeddings')(decoder_inputs)
            decoder_states_inputs = []
            decoder_states = []
            for i in range(self.config.num_decoder_layers):
                decoder_hidden = keras.Input(shape=(self.config.hidden_units,))
                states = [decoder_hidden]
                decoder_outputs, hidden_state_decoder = model.get_layer(f'Decoder_{i}')(decoder_outputs, initial_state=states)
                decoder_states += [hidden_state_decoder]
                decoder_states_inputs += states
        elif self.config.cell_type == 'LSTM':
            encoder_outputs, hidden_state, context_state = model.get_layer(f'Encoder_{self.config.num_encoder_layers-1}').output
            encoder_states = [hidden_state, context_state]
            encoder = Model(encoder_inputs, encoder_states)
            decoder_inputs = model.input[1]  # input_1
            decoder_outputs = model.get_layer('Decoder_embeddings')(decoder_inputs)
            decoder_states_inputs = []
            decoder_states = []
            for i in range(self.config.num_decoder_layers):
                decoder_hidden = keras.Input(shape=(self.config.hidden_units,))
                decoder_context = keras.Input(shape=(self.config.hidden_units,))
                states = [decoder_hidden, decoder_context]
                decoder = model.get_layer(f'Decoder_{i}')
                decoder_outputs, hidden_state_decoder, context_state_decoder = decoder(decoder_outputs, initial_state=states)
                decoder_states += [hidden_state_decoder, context_state_decoder]
                decoder_states_inputs += states
        decoder_dense = model.get_layer('dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
 
        # finding validation accuracy
        total, correct = 0, 0
        output_list, target_list = [], []
        for i in range(len(val_texts)):
            output = self.decode_to_text(val_encoder_input_data[i:i+1], encoder, decoder)
            target = val_target_texts[i][1:len(val_target_texts[i])-1]
            output = output[0:len(output)-1]
            output = output.replace(' ', '').replace('S', '').replace('E', '')
            target = target.replace(' ', '').replace('S', '').replace('E', '')
            output_list.append(output)
            target_list.append(target)
            # print('O/P TGT: ', output, target)
            if output == target:
                correct += 1
            total += 1
        word_val_accuracy = correct / total
        print(word_val_accuracy)
      
    def decode_to_text(self, inputs, encoder, decoder):
        sentence, done = "", False
        beam_sizes = 1 if self.config.decoding_strategy == 'greedy' else self.config.beam_sizes
        sentence = self.beam_search_decoder(inputs, encoder, decoder, beam_sizes)
        return sentence

    def beam_search_decoder(self, inputs, encoder, decoder, beam_sizes):
        
        done, decoded_sentence = False, ""

        # Get encoder states
        encoder_states = [encoder.predict(inputs) for _ in range(self.config.num_decoder_layers)]

        # Decoder input begins with Start Token "S"
        target_sequence = np.array([[target_token_index["S"]]])

        # sum_of_log_probs (score), flag for end of current sequence, target_sequence, states , sequence_token, sequence_char
        sequences = [[0.0, 0,  target_sequence, encoder_states,  list(),list()]]
        while not done:
            candidates = list()
            for i in range(len(sequences)):
                output = decoder.predict([sequences[i][2]] + sequences[i][3])
                output_tokens, states = output[0], output[1:]
                prob = output_tokens[0,-1,:]
                score, flag, _, _, sequence_token, sequence_char = sequences[i]
                if flag == 0:
                    for j in range(len(inverse_target_token_index)):
                        char = inverse_target_token_index[j]
                        target_sequence = np.array([[j]])
                        candidate = [score - np.log(prob[j]), 0, target_sequence, states,  sequence_token + [j] , sequence_char + [char] ]
                        candidates.append(candidate)
            sorted_candidates = sorted(candidates, key=lambda x:x[0])
            k = min(beam_sizes, len(sorted_candidates))
            sequences = sorted_candidates[:k]
            done = True
            for sequence in range(len(sequences)):
                score, flag, tgt_seq, states, sequence_token, sequence_char = sequences[sequence]
                if (len(sequence_char) > max_decoder_seq_length) or (sequence_char[-1] == "E"): 
                    flag = 1
                sequences[sequence][1] = flag
                done = False if flag == 0 else done
            if sequences[0][-1][-1]=="E": 
                done = True
        top_decoded_sentence = ''.join(sequences[0][5])
        return top_decoded_sentence

def train(config):
    run_name = 'ep-'+str(config.epochs)+'-dr-'+str(config.dropout)+'-lr-'+str(config.learning_rate)+'-bs'+str(config.batch_size)+'-es-'+str(config.input_embedding_size)\
        +'-el-'+str(config.num_encoder_layers)+'-dl-'+str(config.num_decoder_layers)+'-hs-'+str(config.hidden_units)+'-cell-'+str(config.cell_type)+'-dec_str-'+str(config.decoding_strategy)\
        +'-bs-'+str(config.beam_sizes)
    print(run_name)
    model_transliteration = TransliterationModel(config)
    model_transliteration.train_and_evaluate(encoder_input_data, decoder_input_data, decoder_target_data,
                                             val_encoder_input_data, val_target_texts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train model.')
    parser.add_argument('-bs', '--batch_size', type=int, default=16,
                        help='Batch size to be used to train and evaluate model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-c', '--cell_type', type=str, default='GRU',
                        choices=['RNN', 'LSTM', 'GRU'])
    parser.add_argument('-d', '--dropout', type=float, default=0.3,
                        help='Dropout used in model')
    parser.add_argument('-hs', '--hidden_units', type=int, default=128,
                        help='Number of hidden units')
    parser.add_argument('-es', '--input_embedding_size', type=int, default=128,
                        help='Embedding dimensions.')
    parser.add_argument('-el', '--num_encoder_layers', type=int, default=1,
                        help='Number of encoder layers.')
    parser.add_argument('-dl', '--num_decoder_layers', type=int, default=1,
                        help='Number of decoder layers.')
    parser.add_argument('-beam', '--beam_sizes', type=int, default=5,
                        help='Beam size for decoding.')
    parser.add_argument('-dec_str', '--decoding_strategy', type=str, default='greedy',
                        choices=['beam_search', 'greedy'])
    args = parser.parse_args()
    train(args)