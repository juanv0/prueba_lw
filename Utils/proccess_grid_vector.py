import pandas as pd
import os
from natsort import natsorted
import re
import numpy as np
from keras import Sequential
from keras.layers import Dense, Embedding, Flatten
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.backend import clear_session
import matplotlib.pyplot as plt
import nltk as nltk
plt.style.use('ggplot')
from keras.preprocessing.text import Tokenizer

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
max_size = 0
def get_fields_name(name):
    file_name = 'mls_data/%s'%name
    name_df = pd.read_csv(file_name)
    print('working on: %s' %name)
    name_array = name_df['Field']
    max_size = len(name_array)
    words = [nltk.word_tokenize(a) for a in name_array.to_list()]
    print(words)
    #note te to_list() func,Vectorizer and Tokenizer uses list of lisf of string
    return name_array.to_list()

def get_id(name):
    str = re.findall(r'\d+',name).pop() 
    return int(str)

cvs_files=natsorted(os.listdir('mls_data/'))
#Store every mls_file 
train_data = [ get_fields_name(name) for name in cvs_files]
#read the label file
listings = pd.read_csv('listing_mlses.csv')
id = [get_id(name) for name in cvs_files]
#we only use the mls_id that we could get the database columns name
filtered_listing = listings[listings['id'].isin(id)]
target_data = filtered_listing['listing_uid'].to_list()

#split the dataframe with sklearn
name_train, name_test, y_train, y_test = train_test_split(
    train_data, target_data, test_size=0.25, random_state=1000
)

""" vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(name_train)

x_train = vectorizer.transform(name_train)
x_test = vectorizer.transform(name_test)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)
print(x_train)
print(y_train)

score = classifier.score(x_test, y_test)
print ("Acuraccy: ", score)
print (len(y_train)) """

#will try this function first
def first_ai_approach(x_train, x_test, y_train, y_test):
    input_dim = x_train.shape[1]
    model = Sequential([
        Dense(5, input_dim=input_dim, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        verbose=False,
        validation_data=(x_test, y_test),
        batch_size=5
    )
    clear_session()

    loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    #plot_history(history)
#first_ai_approach(x_train, x_test, y_train, y_test)

tokenizer = Tokenizer(num_words=5000, lower=False)
tokenizer.fit_on_texts(name_train)
prueba_train = [np.asarray(tokenizer.texts_to_sequences(x))  for x in name_train]

x_prueba = np.asarray(prueba_train)

prueba_y = prueba_train = [np.asarray(tokenizer.texts_to_sequences(x)) for x in y_train]
y_prueba = np.asarray(prueba_y)
X_train = tokenizer.texts_to_sequences(name_train)
X_test = tokenizer.texts_to_sequences(name_test)

vocab_size = len(tokenizer.word_index) + 1
def tokenized_approach(X_train, X_test, y_train, y_test):
    embedding_dim = 50
    model = Sequential([
        Embedding(
        input_dim = vocab_size,
        output_dim=embedding_dim,
        input_length=1,
        ),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid'),

    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)

tokenized_approach(x_prueba, np.asarray(X_test), y_prueba, np.asarray(y_test))
"""
TEST_SIZE = 50287
VAL_SIZE = 1547
X=np.array(X)
Y=np.array(Y)
X_test = X[:-TEST_SIZE]
Y_test = Y[:-TEST_SIZE]

X_val = X[:-VAL_SIZE]
Y_val = Y[:-VAL_SIZE]
"""

""" target = pd.read_csv('listing_mlses.csv')
id = [get_id(name) for name in cvs_files]
corrected = target[target['id'].isin(id)]
corrected.to_csv('corrected.csv')

Y_train = corrected['listing_uid']
X_train = train_data

Y_test = Y_train.head(10).to_numpy()
X_test = X_train[:10]
Y_val = Y_train.tail(2).to_numpy()
X_val = X_train[:-2] """
#will try this method soon afther
def run_ai(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='sgd',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    hist = model.fit(X_train, Y_train,
            batch_size=32, epochs=100,
            validation_data=(X_val, Y_val))
    
    model.evaluate(X_test, Y_test)[1]

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

#run_ai(X, Y, X_val, Y_val, X_test, Y_test)

