from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation, Masking

class musicLSTM:

    def __init__(self, in_shape=None, out_size=None, filepath=None):
        if filepath is None:
            self.model = Sequential()
            self.model.add(LSTM(256, input_shape=in_shape, return_sequences=True))
            self.model.add(Dropout(0.3))
            self.model.add(LSTM(256))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(out_size))
            self.model.add(Activation('softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        else:
            self.model = load_model(filepath)

    def train(self, input, output, filepath, it=200, batch=64):
        self.model.fit(input, output, epochs=it, batch_size=batch)
        self.model.save(filepath)

    def predict(self, input):
        return self.model.predict(input)