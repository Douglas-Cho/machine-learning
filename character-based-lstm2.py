# # Char-level Recurrent Neural Network
# 
# Let's try and build an LSTM-based neural network that uses chars instead of words as its input.
# from https://www.kaggle.com/mamamot/character-based-lstm


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, BatchNormalization, Dropout
from keras import optimizers
from keras.utils import to_categorical
import os

# some parameters
BATCH_SIZE = 1024  # batch size for the network
EPOCH_NUMBER = 1  # number of epochs to train
THRESHOLD = 5  # symbols appearing fewer times will be replaced by a placeholder

train = pd.read_csv('sanction_train.csv', encoding='iso-8859-1')
train.index = train['Index']
x_train = train['Name']
y_train = train['Label']


# An important statistic is the average length of the comment:

x_train.apply(lambda x: len(x)).describe()


# Get counts of unique symbols in the training set:

unique_symbols = Counter()

for _, message in x_train.iteritems():
    unique_symbols.update(message)
    
print("Unique symbols:", len(unique_symbols))


# Find symbols that appear fewer times than the threshold:

uncommon_symbols = list()

for symbol, count in unique_symbols.items():
    if count < THRESHOLD:
        uncommon_symbols.append(symbol)

print("Uncommon symbols:", len(uncommon_symbols))


# Replace them with a placeholder:

DUMMY = uncommon_symbols[0]
tr_table = str.maketrans("".join(uncommon_symbols), DUMMY * len(uncommon_symbols))

x_train = x_train.apply(lambda x: x.translate(tr_table))


# We will need the number of unique symbols further down when we will decide on the dimensionality of inputs.

num_unique_symbols = len(unique_symbols) - len(uncommon_symbols) + 1 

tokenizer = Tokenizer(
    char_level=True,
    filters=None,
    lower=False,
    num_words=num_unique_symbols
)

tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Pad the input: I use the 500 lenght, just a bit over the median length.

padded_sequences = pad_sequences(sequences, maxlen=500)

# I will take just a bit of the data as the validation set to see that the network converges:

#x_train, x_val, y_train, y_val = train_test_split(padded_sequences, y_train, stratify=y_train['Label'], test_size=0.25)
x_train = padded_sequences
x_val = padded_sequences
y_val = y_train

# So, let's define the model!
model = Sequential()
model.add(LSTM(150, input_shape=(500, num_unique_symbols), activation="tanh", return_sequences=True))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(LSTM(100, input_shape=(500, num_unique_symbols), activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(100, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(100, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(100, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(100, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(100, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(100, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(100, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(50, activation="tanh"))
model.add(BatchNormalization())
#model.add(Dropout(0.4))
model.add(Dense(1, activation="sigmoid"))

x_val = to_categorical(np.array(x_val), num_classes=num_unique_symbols)

# Load model weights
model.load_weights('lstm-char2_weight_ep1.h5')
checkpoint = ModelCheckpoint(filepath=os.path.join("my_weights_best.h5"), monitor='val_loss', save_best_only=True, mode='auto')

# Let's track the performance using the custom function that will be used for the leaderboard:

sgd = optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss="binary_crossentropy")


def kaggle_loss(y_true, y_pred):
    total_loss = 0
    for i in range(y_true.shape[0]):
        total_loss += log_loss(y_true, y_pred)
    return total_loss / y_true.shape[0]


with open("res_2.txt", "w") as f:
    for epoch in range(EPOCH_NUMBER):
        print("Epoch", epoch)
        for i in range(0, len(x_train), BATCH_SIZE):
            batch = x_train[i:i+BATCH_SIZE]
            batch = to_categorical(batch, num_classes=num_unique_symbols)
            y_batch = y_train.iloc[i:i+BATCH_SIZE]
            model.fit(batch, y_batch, batch_size=256, validation_data=(batch, y_batch), callbacks=[checkpoint])
        y_pred = np.round(model.predict_proba(x_val), 0)
        y_pred = y_pred.astype(np.int64)
        res = kaggle_loss(y_val, y_pred)
        print("Loss:", res)
        f.write("{}: {}\n".format(epoch, res))
        model.save("lstm-char2_ep{}.h5".format(epoch + 1))
        model.save_weights("lstm-char2_weight_ep{}.h5".format(epoch + 1))


print('################### with best weights #####################')
model.load_weights('my_weights_best.h5')
y_pred = np.round(model.predict_proba(x_val), 0)
y_pred = y_pred.astype(np.int64)
res = kaggle_loss(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)
print('\nConfusion Matrix : \n', cm)