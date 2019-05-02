from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation

class musicNN:

    def __init__(self, in_shape, out_size):
        self.model = Sequential()
        self.model.add(Dense(256, activation='relu', input_shape=(in_shape[0],)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(out_size))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train(self, input, output, filepath, it=200, batch=64):
        self.model.fit(input, output, epochs=it, batch_size=batch)
        self.model.save(filepath)

    def predict(self, input, filepath):
        self.model = load_model(filepath)
        return self.model.predict(input)
