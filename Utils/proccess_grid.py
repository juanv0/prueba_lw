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
plt.style.use('ggplot')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

#extracting the id for each mls
def get_id(name):
    str = re.findall(r'\d+',name).pop() 
    return int(str)

#todo: reimplement new way to get the data as ['row', 'in(listing_id)']
X = []
Y = []
#reading all files inside mls_data/
cvs_files=natsorted(os.listdir('mls_data/'))
#reading listing_mlses to get target(s) rows
listings_df = pd.read_csv('listing_mlses')
id = [get_id(name) for name in cvs_files]
filtered_listings = listings_df[listings_df['id'].isin(id)]

for file in cvs_files:

    file_name = 'mls_data/%s'%file
    print('reading %s' %file)
    #reading csv file as dataframe
    df = pd.read_csv(file_name)
    #getting fields row into one array with extend
    name_df = df['Field']
    X.extend(name_df.to_numpy())
    #get the id number
    id = get_id(file)
    #create an array with the same dimension of {name_df}
    #cointaning 1 or 0 if the value is in listing_uid row
    filter = (filtered_listings['id'] == id)
    target_value = filtered_listings[filter]['listing_uid'].to_numpy()
    #creating array of results
    #print(target_value)
    target_df = pd.DataFrame(name_df.isin(target_value).replace({True: target_value, False: ''}))
    print(target_df)
    Y.extend(target_df.to_numpy())

#make Y one-dimensional
Y = np.array(Y).ravel()
#split the dataframe with sklearn
X_train, X_cv, Y_train, Y_cv = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=None,
    stop_words=None,
    max_features=5000
)
vectorizer.fit(X_train)
X_train = vectorizer.fit_transform(X_train)
X_train = X_train.toarray()
print(X_train.shape)

X_cv = vectorizer.transform(X_cv)
X_cv = X_cv.toarray()
print(X_cv.shape)

X_test = vectorizer.transform(Y_train)
X_test = X_test.toarray()
print(X_test.shape)

""" x_train = vectorizer.transform(name_train)
x_test = vectorizer.transform(name_test) """

vocab = vectorizer.get_feature_names()
print(f"Printing first 100 vocabulary samples:\n{vocab[:100]}")

distribution = np.sum(X_train, axis=0)

print("Printing first 100 vocab-dist pairs:")

for tag, count in zip(vocab[:100], distribution[:100]):
    print(count, tag)

forest = RandomForestClassifier() 
forest = forest.fit( X_train, Y_train)

predictions = forest.predict(X_cv) 
print("Accuracy: ", accuracy_score(Y_cv, predictions))

""" classifier = LogisticRegression()
classifier.fit(x_train,y_train)
print(Y)
print(y_train)

score = classifier.score(x_test, y_test)
print ("Acuraccy: ", score)
print (len(Y)) """

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

""" tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(name_train)
X_train = tokenizer.texts_to_sequences(name_train)
X_test = tokenizer.texts_to_sequences(name_test)

vocab_size = len(tokenizer.word_index) + 1 """
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

#tokenized_approach(X_train, X_test, y_train, y_test)
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

