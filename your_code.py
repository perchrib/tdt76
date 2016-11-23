import pickle
import random
from PIL import Image
import os
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
import keras.callbacks as cb
from keras.callbacks import EarlyStopping
from keras.models import load_model



def get_dataset(dataset):
    print(dataset)
    img_dir = dataset + "/img_embeddings/"
    label_dir = dataset + '/label_embeddings/'
    img_embedding_files = get_files_in_dir(img_dir)
    label_embedding_files = get_files_in_dir(label_dir)
    if len(img_embedding_files) != len(label_embedding_files):
        print("Length size not match!!!")
        return
    X_train = []
    Y_train = []
    zeros = np.zeros(300, dtype='float32')
    for i in range(len(img_embedding_files)):
        img_emb = load_file(img_dir + img_embedding_files[i])
        img_label = load_file(label_dir + label_embedding_files[i])
        if len(img_emb) != len(img_label):
            print("pic miss word label")
            return
        else:
            for img_ID in img_emb.keys():
                img_vec = img_emb[img_ID]

                word_vec = img_label[img_ID]
                if np.array_equal(word_vec, img_vec):
                    print("Fuck")
                X_train.append(img_vec)
                Y_train.append(word_vec)
    X_train = np.asarray(X_train, dtype='float32')
    Y_train = np.asarray(Y_train, dtype='float32')
    return X_train, Y_train


def normalize_vectors(vectors):
    min = 0.0
    max = 0.0
    n_vectors = []
    for v in vectors:
        if v.min() < min:
            min = v.min()
        if v.max() > max:
            max = v.max()
    print("MAX: ", max, " MIN: ", min)
    total = abs(max) + abs(min)
    for v in vectors:
        n_vectors.append((v + min) / total)
    n_vectors = np.asarray(n_vectors, dtype='float32')
    return n_vectors





def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_files_in_dir(dir):
    files = os.listdir(dir)
    return files

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def train(location='./train/'):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that is required for testing the model
    must be saved to file (e.g., pickle) so that the test procedure can load, execute and report
    :param location: The location of the training data folder hierarchy
    :return: nothing

    """
    x_train, y_train = get_dataset('train')
    x_val, y_val = get_dataset('validate')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model = init_model()
    history = LossHistory()
    model.fit(x_train, y_train, nb_epoch=100, batch_size=128,
              callbacks=[early_stopping],
              validation_data=(x_val, y_val))
    model.save('models/sec_model.h5')



def test(queries=list(), location='./test'):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned
    :param queries: list of image-IDs. Each element is assumed to be an entry in the test set. Hence, the image
    with id <id> is located on my computer at './test/pics/<id>.jpg'. Make sure this is the file you work with...
    :param location: The location of the test data folder hierarchy
    :return: a dictionary with keys equal to the images in the queries - list, and values a list of image-IDs
    retrieved for that input
    """

    # ##### The following is an example implementation -- that would lead to 0 points  in the evaluation :-)
    my_return_dict = {}

    # Load the dictionary with all training files. This is just to get a hold of which
    # IDs are there; will choose randomly among them
    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
    training_labels = list(training_labels.keys())

    for query in queries:
        # This is the image. Just opening if here for the fun of it; not used later
        image = Image.open(location + '/pics/' + query + '.jpg')
        image.show()

        # Generate a random list of 50 entries
        cluster = [training_labels[random.randint(0, len(training_labels) - 1)] for idx in range(50)]
        my_return_dict[query] = cluster

    return my_return_dict

#first model
# def init_model():
#     print('Compiling Model ... ')
#     model = Sequential()
#     model.add(Dense(4096, input_dim=2048))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(1024))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(300))
#     model.add(Activation('softmax'))
#
#     rms = RMSprop()
#     model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
#     print('Model compiled')
#     return model

def init_model():
    print('Compiling Model ... ')
    model = Sequential()
    model.add(Dense(1024, input_dim=2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print('Model compiled')
    return model
if __name__ == "__main__":
    train()
