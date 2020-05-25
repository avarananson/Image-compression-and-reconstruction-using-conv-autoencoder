
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


class Model:
    """docstring for ClassName"""

    def __init__(self,input_shape):
        # self.train = tr
        # self.test = tst
        self.input_shape = input_shape

    def createmodel(self):
        # createing encoder

        self.encoder = Sequential()

        self.encoder.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                activation='relu', input_shape=self.input_shape))
        self.encoder.add(BatchNormalization())
        self.encoder.add(MaxPool2D(pool_size=(2, 2), padding='same'))

        self.encoder.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu', input_shape=self.input_shape))

        self.encoder.add(BatchNormalization())
        self.encoder.add(MaxPool2D(pool_size=(2, 2), padding='same'))

        self.encoder.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same',
                                activation='relu', input_shape=self.input_shape))
       # model.add(MaxPool2D(pool_size=(2, 2)), padding='same')

        # creatinf the decoder

        self.decoder = Sequential()
      #  print("valuesss", (self.encoder.layers[6].output.shape))
        self.decoder.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same',
                                activation='relu'))

        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(BatchNormalization())

        self.decoder.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                activation='relu', input_shape=self.input_shape))
        self.decoder.add(UpSampling2D((2, 2)))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                activation='relu', input_shape=self.input_shape))
        self.decoder.add(Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                                activation='sigmoid', input_shape=self.input_shape))

        model = Sequential([self.encoder, self.decoder])
        model.compile(optimizer='adam',
                      loss='mse', metrics=['accuracy'])
        print(model.summary())

        return model
