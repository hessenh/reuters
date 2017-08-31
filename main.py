from dataset import Dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM


'''
Model specifics
'''
def define_model(number_of_words, number_of_features, number_of_categories):
    model = Sequential()
    model.add(LSTM(int(number_of_words*1.3), input_shape=(number_of_words, number_of_features)))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_categories))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_model(model, dataset, batch_size, epochs):
    X_train, X_test, y_train, y_test = dataset.get_train_test()

    for i in range(0, epochs):
        model.fit(X_train,
                  y_train,
                  shuffle=True,
                  batch_size=batch_size,
                  epochs=1)
        print get_score(model, dataset)
    return model

def get_score(model, dataset):
    X_train, X_test, y_train, y_test = dataset.get_train_test()
    y_pred = model.predict(X_test)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    number_of_words = 100
    number_of_features = 400
    dataset = Dataset(
        path='./data/reuters21578/',
        should_load_word2vec=False,
        number_of_words=number_of_words,
        number_of_features=number_of_features)

    model = define_model(
        number_of_words=number_of_words,
        number_of_features=number_of_features,
        number_of_categories=135)

    trained_model = train_model(
        model=model,
        dataset=dataset,
        batch_size=64,
        epochs=5)
