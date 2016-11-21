from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
import os
import pickle
import sys
import time
from error_finder import get_lost_images, get_lost_labels
dataset = "train/"

pickle_dir = "pickle/"
image_dir = "pics/"
txt_dir = "txt/"
root_dir = dataset + image_dir
save_dir = "img_embeddings/"


def run_image_preprocessor(root_dir, save_dir, model):
    sub_dir = get_sub_dir(root_dir)
    dir_counter = 1
    print("Started preprocessor ")
    for dir in sub_dir:
        if os.path.exists(save_dir + dir + ".pickle"):
            print("\nFile already exists:", save_dir + dir + ".pickle")
        else:
            img_path = root_dir + dir + "/"
            save_path = save_dir + dir + ".pickle"
            embeddings = get_embeddings(img_path, model, dir_counter)
            save_embeddings(save_path, embeddings)
            dir_counter += 1


def get_image_feature(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)[0]
    return feature


def print_status(dir_counter, img_counter, failed_img_counter, list):
    out = "Extracting Images: {0} / 1000 dir ==>  {1} / {2} images  ==> FAILED IMAGES: {3}".format(str(dir_counter),
                                                                                                   str(img_counter),
                                                                                                   str(len(list)), str(
            failed_img_counter))
    sys.stdout.write('\r')
    sys.stdout.write(out)
    sys.stdout.flush()

failed_image_counter = 0
def get_embeddings(path, model, dir_counter):
    images = get_pics_in_subdir(path)
    embed_images = dict()
    img_counter = 0

    for img in images:
        img_ID = img[:-4]
        global failed_image_counter
        if img_ID not in lost_images and img_ID not in lost_image_labels:
            embed_images[img_ID] = get_image_feature(path + img, model)
            print_status(dir_counter, img_counter + 1, failed_image_counter, images)
        else:
            failed_image_counter += 1
        img_counter += 1
    return embed_images


def save_embeddings(filename, dict):
    with open(filename, "wb") as img_embeddings:
        pickle.dump(dict, img_embeddings)


def load_embeddings(filename):
    with open(filename, "rb") as img_embeddings:
        embeddings = pickle.load(filename, img_embeddings)
    return embeddings


def get_pics_in_subdir(path):
    pics = list(filter(lambda x: ".DS_Store" not in x, os.listdir(path)))
    return pics


def get_sub_dir(path):
    directories = list(filter(lambda x: ".DS_Store" not in x, os.listdir(path)))
    return directories

if __name__ == "__main__":
    lost_images = get_lost_images(image_dir, dataset)
    lost_image_labels = get_lost_labels(txt_dir, dataset)
    print("Import model....")
    base_model = InceptionV3(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)

    start = time.time()
    run_image_preprocessor(root_dir, save_dir, model)
    end = time.time()
    seconds = end - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("\n%d h ,%02d min %02d sec" % (h, m, s))

