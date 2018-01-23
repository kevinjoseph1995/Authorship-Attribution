import numpy as np

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
TRAIN_SPLIT = 0.8

# Load data
import cPickle as pickle
train_set = pickle.load(open('TrainData.p', 'rb'))
test_set = pickle.load(open('TestData.p', 'rb'))
texts = []
labels = []
labels_index = {}
for author in train_set:
    for text in train_set[author]:
        texts.append(text)
        if author in labels_index:
            label_id = labels_index[author]
        else:
            label_id = len(labels_index)
            labels_index[author] = label_id
        labels.append(label_id)

train_size = len(texts)

for author in test_set:
    for text in train_set[author]:
        texts.append(text)
        if author in labels_index:
            label_id = labels_index[author]
        else:
            label_id = len(labels_index)
            labels_index[author] = label_id
        labels.append(label_id)

print len(texts), len(labels)

# Prepare data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

from keras.utils import to_categorical
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
train_samples = int(TRAIN_SPLIT * train_size)

x_train = data[:train_samples]
y_train = labels[:train_samples]
x_val = data[train_samples:train_size]
y_val = labels[train_samples:train_size]
x_test = data[train_size:]
y_test = labels[train_size:]

print x_train.shape, y_train.shape
print x_val.shape, y_val.shape
print x_test.shape, y_test.shape

# Get word embeddings
import os
GLOVE_DIR = '/Users/sreekar/Downloads/glove.6B/'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Dropout
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = Dropout(0.3)(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
# x = Dropout(0.3)(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
# x = Dropout(0.3)(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

from keras.models import Model
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=128)

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=len(x_test))


