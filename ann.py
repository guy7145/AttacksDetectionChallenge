from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation


def generate_autoencoder(input_shape, hidden_layer_size):
    input_img = Input(shape=input_shape)
    encoded = Dense(units=hidden_layer_size, input_shape=input_shape, activation='relu')(input_img)
    decoded = Dense(units=765, activation='hard_sigmoid')(encoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return encoder, autoencoder


def generate_classifier(input_shape):
    print("input shape: " + str(input_shape))
    input_img = Input(shape=input_shape)
    hidden_layer = Dense(units=500, activation='relu')(input_img)
    hidden_layer = Dense(units=250, activation='relu')(hidden_layer)
    output_layer = Dense(units=1, activation='softmax')(hidden_layer)

    model = Model(input_img, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("model generated!")
    return model


def generate_user_autoencoder(num_of_keys, hidden_layer_size):
    input_img = Input(shape=(num_of_keys,))
    hidden_layer = Dense(shape=(hidden_layer_size,))(input_img)
    output_layer = Dense(units=1, activation='sigmoid')(hidden_layer)

    model = Model(input_img, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def ngram_simple_classifier(num_of_keys):
    input_img = Input(shape=(num_of_keys,))
    # act = Activation('relu')(input_img)
    # hidden_layer = Dense(units=int(num_of_keys/2), activation='relu')(input_img)
    output_layer = Dense(units=1, activation='sigmoid')(input_img)

    model = Model(input_img, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

