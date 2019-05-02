from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Activation, Masking

class musicLSTM:

    def __init__(self, in_shape, out_size):
        self.model = Sequential()
        self.model.add(Masking(mask_value=-1, input_shape=(in_shape[0], in_shape[1])))
        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(out_size))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def train(self, input, output, filepath, it=200, batch=64):
        self.model.fit(input, output, epochs=it, batch_size=batch)
        self.model.save(filepath)

    def predict(self, input, filepath):
        self.model = load_model(filepath)
        return self.model.predict(input)