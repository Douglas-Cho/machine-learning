from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sqlalchemy import create_engine
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os


# Define and connect SQL database
engine = create_engine('mssql+pyodbc://DESKTOP\SQLEXPRESS/ML2?driver=SQL+Server+Native+Client+11.0',
                       echo=True)
cnx = engine.raw_connection()

"""
# ==================================== Part 1. Preparing data from Audit Trail file =============================================
# Process audit trail file that has all actions in one column \
# to place each trade in a record with multiple columns of actions in chronological order.  

# List up all trade types
tradetypes = {'DEALP', 'BOND', 'FXFORWARD', 'FXSPOT', 'FXSWAP', 'MONEYMARKET', 'REPO', 'SWAP'}

for tradetype in tradetypes:
    # Create data tables per trade type
    sql_query0 = "SELECT M07_Flag, TID, AUDIT_DATETIME, AUDIT_ACTION FROM ML2.dbo.AUDIT_TRAIL_MASTER WHERE TRADETYPE = '" + tradetype + "';"
    df_temp = pd.read_sql_query(sql_query0, engine)
    sql_query1 = 'RNN_' + tradetype
    df_temp.to_sql(name=sql_query1, con=engine, if_exists='replace', index=False)

    # Load data from SQL server
    sql_query2 = 'SELECT * FROM RNN_' + tradetype
    df_train = pd.read_sql_query(sql_query2, cnx)
    count1 = len(df_train)
    print(count1)

    # Count the number of unique TIDs
    count2 = len(df_train.groupby('TID').nunique())
    print(count2)

    # Define max length of actions
    max_length = 40

    # Create a dataframe filled with zero
    df_data = pd.DataFrame(0, index=range(count2), columns=range(max_length))

    # Get the list of unique TIDs
    df_A = pd.DataFrame([[k, v.values]
                         for k, v in df_train.groupby('TID').groups.items()],
                        columns=['col', 'indices'])

    A = df_A['col']

    # Fillout the dataframe by transposing each TID's dataframe
    i = 0
    j = 0
    k = 0

    for TID in A:
        df_sub = df_train.groupby('TID').get_group(TID).sort_values(by=['AUDIT_DATETIME'])
        count3 = len(df_sub)
        df_data.iloc[i, 0] = df_sub.iat[0, 0]
        df_data.iloc[i, 1] = df_sub.iat[0, 1]
        del df_sub['M07_Flag']
        del df_sub['TID']
        del df_sub['AUDIT_DATETIME']
        df_sub_T = df_sub.T
        for k in range(count3):
            df_data.iloc[i, j + 2] = df_sub_T.iat[0, k]
            j += 1
            k += 1
        pass
        i += 1
        j = 0
        k = 0

    # Add headers and save in database for readable reference
    df_data.columns = ['Lable', 'TID', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI',
                       'AJ', 'AK', 'AL', 'AN']
    sql_query3 = 'RNN_' + tradetype + '_DATA'
    df_data.to_sql(name=sql_query3, con=engine, if_exists='replace', index=False)

# =================================== Part 2. Tokenizing the input =========================================
# Codify the actions into numeric representation form to pass to LSTM. 

# List up all trade types
tradetypes = {'DEALP', 'BOND', 'FXFORWARD', 'FXSPOT', 'FXSWAP', 'MONEYMARKET', 'REPO', 'SWAP'}
max_length = 40

for tradetype in tradetypes:
    # Create data tables per trade type
    sql_query4 = 'SELECT * FROM RNN_' + tradetype + '_DATA'
    df_data = pd.read_sql_query(sql_query4, cnx)
    count2 = len(df_data)
    print(count2)

    # tokenize the actions ===============================================
    print(df_data.head())

    # define action token dicktionary
    actions = {'SET': 0, 'CANCEL': 1, 'DO': 2, 'GOLIVE': 3, 'TEMPDO': 4, 'GOEDIT': 5, 'GOLIVE2': 6, 'LIVE': 7,
               'MONEYMARKETBK': 8, 'MONEYMARKETOV': 9, 'REJECT': 10, 'USERCREAT': 11, 'SAVE': 12, 'SSISET': 13,
               'PREP': 14, 'TEMPSAVE': 15, 'APPROVE': 16}

    # tokenize actions through looping

    for i in range(count2):
        for j in range(max_length):
            if j < 2:
                pass
            else:
                df_data.iloc[i, j] = actions.get(df_data.iat[i, j], 0)
                j += 1
        else:
            pass
        i += 1

    print(df_data.head())

    sql_query4 = 'RNN_' + tradetype + '_DATA_NUM'
    df_data.to_sql(name=sql_query4, con=engine, if_exists='replace', index=False)

"""
# ========================================== Part 3. LSTM learning =====================================================
# Used the LSTM model code for imdb sentiment analysis from here --> https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
# Added model callbacks and confusion matrix - Douglas Cho

max_features = 100
#max_features = 20000
maxlen = 40
#maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
NB_CLASSES = 2

#tradetypes = {'DEALP', 'BOND', 'FXFORWARD', 'FXSPOT', 'FXSWAP', 'MONEYMARKET', 'REPO', 'SWAP'}
tradetypes = {'DEALP', 'BOND', 'MONEYMARKET', 'REPO', 'SWAP'}

for tradetype in tradetypes:
    print('\n============================', tradetype, '===========================')
    # Define filenames to save weights
    best_filename = 'my_weights_best_lstm_' + tradetype + '.h5'
    ending_filename = 'my_weights_' + tradetype + '.h5'

    # Load training data from database
    sql_query5 = 'SELECT * FROM RNN_' + tradetype + '_DATA_NUM'
    df_data = pd.read_sql_query(sql_query5, cnx)
    count2 = len(df_data)
    print(count2)
    print('Loading training data...')
    del df_data['TID']
    X_train_orig = df_data.as_matrix()
    y_train = X_train_orig[:, 0]
    x_train = np.delete(X_train_orig, 0, axis=1)

    # Load test data from database
    sql_query6 = 'SELECT * FROM RNN_' + tradetype + '_TEST_NUM'
    df_test = pd.read_sql_query(sql_query6, cnx)
    print('Loading test data...')
    count3 = len(df_test)
    print(count3)
    del df_test['TID']
    X_test_orig = df_test.as_matrix()
    y_test = X_test_orig[:, 0]
    x_test = np.delete(X_test_orig, 0, axis=1)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print(x_train)

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    y_train1 = y_train
    y_test1 = y_test

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.load_weights(best_filename)
    checkpoint = ModelCheckpoint(filepath=os.path.join(best_filename), monitor='val_loss', save_best_only=True, mode='auto')
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    print('Train...')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              verbose=1,
              epochs=1000,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint, earlystopping])

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    # Predict with last weights ---------------------------------------------
    model.save_weights(ending_filename, overwrite=True)

    model.load_weights(ending_filename)
    pre_cls=model.predict_classes(x_test)
    cm = confusion_matrix(y_test1, pre_cls)
    print('\n============= ', tradetype, ' =============')
    print('\nConfusion Matrix : \n', cm)

    FalsePositiveRatio = round((sum(cm[0,:]) - cm[0,0]) / sum(cm[0,:]) * 100, 2)
    print('\nFalse Positive Ratio: ', FalsePositiveRatio, '%')

    i = 0
    Base = 0
    Correct = 0
    while i < NB_CLASSES - 1:
        Base += sum(cm[i,:])
        Correct += cm[i, i]
        i += 1

    FalseNegativeRatio = round((1 - (Correct / Base)) * 100, 2)
    print('False Negative Ratio: ', FalseNegativeRatio, '%')

    # Predict with best accuracy weights -------------------------------------
    model.load_weights(best_filename)
    pre_cls=model.predict_classes(x_test)
    cm = confusion_matrix(y_test1, pre_cls)
    print('\n============= ', tradetype, ' =============')
    print('\nConfusion Matrix with best case callback: \n', cm)

    FalsePositiveRatio = round((sum(cm[0,:]) - cm[0,0]) / sum(cm[0,:]) * 100, 2)
    print('\nFalse Positive Ratio: ', FalsePositiveRatio, '%')

    i = 0
    Base = 0
    Correct = 0
    while i < NB_CLASSES - 1:
        Base += sum(cm[i,:])
        Correct += cm[i, i]
        i += 1

    FalseNegativeRatio = round((1 - (Correct / Base)) * 100, 2)
    print('False Negative Ratio: ', FalseNegativeRatio, '%')

    # print exceptions on the screen
    incorrects = np.nonzero(pre_cls.reshape((-1,)) != y_test)
    print("\n", incorrects)


    # update database for further analysis
    model.load_weights(best_filename)
    pre_cls=model.predict_classes(x_test)
    df_pred = pd.DataFrame(pre_cls.reshape((-1,)), columns=['Pred'])
    outputdb = 'RNN_out_' + tradetype
    df_pred.to_sql(name=outputdb, con=engine, if_exists = 'replace', index=False)

cnx.close()
