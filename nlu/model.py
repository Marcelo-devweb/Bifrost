import yaml
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import to_categorical

data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower())
    outputs.append('{}\{}'.format(command['entity'], command['action']))


chars = set()

for input in inputs + outputs:
    for ch in input:
        if ch not in chars:
            chars.add(ch)

chr2idx = {}
idx2chr = {}

for i, ch in enumerate(chars):
    chr2idx[ch] = i
    idx2chr[i] = ch



max_seq = max([len(x) for x in inputs])

print('Múmero de chars:', len(chars))
print('Maior seq:',max_seq)

input_data = np.zeros((len(inputs), max_seq, len(chars)), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k, chr2idx[ch]] = 1.0

input_data = np.zeros((len(inputs), max_seq), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k] = chr2idx[ch]

labels = set(outputs)
label2idx = {}
idx2label = {}

for k, label in enumerate (labels):
    label2idx[label] = k
    idx2chr[k] = label

output_data = []

for output in outputs:
    output_data.append(label2idx[output])

output_data = to_categorical(output_data, len(output_data))

print(output_data[0])

model = Sequential()
model.add(Embedding(len(chars), 64))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(len(output_data), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(input_data, output_data, epochs=16)

'''
print(inputs)
print(outputs)
'''

