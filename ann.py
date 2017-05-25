from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout


def generate_autoencoder(input_shape, hidden_layer_size):
    input_img = Input(shape=input_shape)
    encoded = Dense(units=hidden_layer_size, input_shape=input_shape, activation='relu')(input_img)
    decoded = Dense(units=765, activation='hard_sigmoid')(encoded)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return encoder, autoencoder


def generate_classifier(num_of_keys, ratio):
    print("input size: {}; ratio: {}".format(num_of_keys, ratio))
    input_img = Input(shape=(num_of_keys, ))
    hidden_layer = Dense(units=int(num_of_keys * ratio), activation='linear')(input_img)
    # hidden_layer = Dense(units=int(num_of_keys * ratio / 2), activation='linear')(hidden_layer)
    # hidden_layer = Dense(units=int(num_of_keys * ratio / 3), activation='linear')(hidden_layer)
    output_layer = Dense(units=1, activation='sigmoid')(input_img)

    model = Model(input_img, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print("model generated!")
    return model