import pickle
import random
from PIL import Image
import os
import sys
import numpy as np
#cosimilarity
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
import keras.callbacks as cb
from keras.callbacks import EarlyStopping
from keras.models import load_model

#Inception
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from image_preprocessor import get_image_feature

#Directory for extracted features
root_dir = "./feature_extraction/"



#cluster functions
from cluster_imgs import predict_cluster

import operator

def get_dataset(dataset):
    print("Preprocessing ", dataset, "set....")
    img_dir = root_dir + dataset + "/img_embeddings/"
    label_dir = root_dir + dataset + '/label_embeddings/'
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
                if np.array_equal(word_vec, zeros):
                    #print("Word vector is Zero... Could'nt find in Glove")
                    continue
                else:
                    X_train.append(img_vec)
                    Y_train.append(word_vec)

    X_train = np.asarray(X_train, dtype='float32')
    Y_train = np.asarray(Y_train, dtype='float32')
    return X_train, Y_train

#Cluster
def find_most_similar_pics_in_cluster(img_vector, cluster_name):
    pics = load_file("db_cluster/" + cluster_name)
    treshold = 0.5
    img_ids = list(pics.keys())
    img_vectors = list(pics.values())
    #return img_ids
    results = cosine_similarity([img_vector], [img_vectors][0])[0]
    #print("RESULTS: ", results)
    indices = np.where(results >= treshold)[0]
    #print("inices: ", indices)
    img_retrieved = []
    for i in indices:
        img_retrieved.append(img_ids[i])
    #print("img retrievd: ", img_retrieved)
    return img_retrieved

def find_most_similar_pics(img_vector, img_ids, img_vectors):
    treshold = 0.7
    results = cosine_similarity([img_vector], [img_vectors][0])[0]
    indices = np.where(results >= treshold)[0]
    #while len(indices) == 0:
    #    treshold -= 0.05
    #    results = cosine_similarity([img_vector], [img_vectors][0])[0]
    #    indices = np.where(results >= treshold)[0]
    img_retrieved = []
    for i in indices:
        img_retrieved.append(img_ids[i])
    return img_retrieved


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_files_in_dir(dir):
    files = list(filter(lambda x: ".pickle" in x, os.listdir(dir)))
    return files


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
    history = model.fit(x_train, y_train, nb_epoch=100, batch_size=128,
              callbacks=[early_stopping],
              validation_data=(x_val, y_val), verbose=1)



    model.save('models/mse_three_hidden_layers_with_uniform.h5')
    plot_result(history)


def plot_result(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def test(queries=list(), location='./test'):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned
    :param queries: list of image-IDs. Each element is assumed to be an entry in the test set. Hence, the image
    with id <id> is located on my computer at './test/pics/<id>.jpg'. Make sure this is the file you work with...
    :param location: The location of the test data folder hierarchy
    :return: a dictionary with keys equal to the images in the queries - list, and values a list of image-IDs
    retrieved for that input
    """
    #create image that goes through first model #mine
    print("loading inception model...")
    base_model = InceptionV3(weights='imagenet')
    inception_model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
    print("loading my model...")
    my_model = load_my_model()


    # ##### The following is an example implementation -- that would lead to 0 points  in the evaluation :-)
    my_return_dict = {}

    # Load the dictionary with all training files. This is just to get a hold of which
    # IDs are there; will choose randomly among them
    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
    training_labels = list(training_labels.keys())

    all_extracted_training_labels = load_file("./feature_extraction/train/label_embeddings/combined.pick")
    img_ids = list(all_extracted_training_labels.keys())
    img_labels = list(all_extracted_training_labels.values())

    limit = len(queries)
    print('num of input pics: ', limit)
    counter = 0
    for query in queries:
        #mine
        subfolder = bruteforce_to_find_subfolder(query + ".jpg", location + "/pics/")
        #print(location + '/pics/' + subfolder + query + '.jpg')
        img_path = location + '/pics/' + subfolder + query + '.jpg'

        #find img feature from inception model
        img_inception_feature = get_image_feature(img_path, inception_model)
        img_feature = my_model.predict(np.asarray([img_inception_feature]))[0]

        #cluster_file = predict_cluster(img_feature)
        #similar_pics = find_most_similar_pics(img_feature, cluster_file)

        similar_pics = find_most_similar_pics(img_feature, img_ids, img_labels)

        #print("input img: ", query, " output img: ", similar_pics)
        my_return_dict[query] = similar_pics
        # This is the image. Just opening if here for the fun of it; not used later
        counter += 1
        print_status(counter, limit)
        #image = Image.open(img_path)
        #image.show()

        # Generate a random list of 50 entries

    return my_return_dict

def print_status(counter, limit):
    out = "Progress: " + str(counter) + " / " + str(limit)
    sys.stdout.write('\r')
    sys.stdout.write(out)
    sys.stdout.flush()

def bruteforce_to_find_subfolder(jpg_file, location):
    subfolders = list(filter(lambda x: "0000" in x, os.listdir(location)))
    for sub in subfolders:
        path = location + sub + "/"
        pics = list(filter(lambda x: ".jpg" in x, os.listdir(path)))
        if jpg_file in pics:
            return sub + "/"


def preprocess_image_in_inception(img_path, inception_model):
    feature = get_image_feature(img_path, inception_model)
    return feature

def load_my_model():
    model = load_model('models/mse_three_hidden_layers_with_uniform.h5')
    return model


def init_model():
    print('Init Model ... ')
    model = Sequential()
    model.add(Dense(1024, input_dim=2048, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(750, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(512, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(300, init='uniform'))
    model.add(Activation('tanh'))

    rms = RMSprop(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=rms, metrics=['accuracy'])
    print('Model compiled')
    return model

if __name__ == "__main__":
    start = time.time()
    train()
    end = time.time()
    seconds = end - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("Time spend training ==> %d h ,%02d min %02d sec" % (h, m, s))

    """Sample run for retrieved images"""
    input_label = load_file('./validate/pickle/combined.pickle')
    training_labels = load_file("./train/pickle/combined.pickle")
    res = test(['61a820bad2154a4d'], "validate")
    imgs = list(res.values())
    print("\n")
    print(' Input: ', '169aee7b9ebde79d', "Labels: ", input_label['61a820bad2154a4d'])
    for i in imgs[0]:
        print('Output: ', i, "Labels: ", training_labels[i])