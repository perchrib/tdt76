import pickle
import numpy as np
from error_finder import get_lost_images, get_lost_labels
import os
import sys

dataset = "train/"
glove_dir = "glove/"
glove_file = "glove.6B.300d.txt"
pickle_dir = "pickle/"
image_dir = "pics/"
txt_dir = "txt/"

save_dir = "label_embeddings/"

root_dir = dataset + pickle_dir

def get_word_embeddings_from_glove_file(dir, filename):
    word_embeddings = dict()
    with open(dir + filename) as glove:
        for line in glove:
            values = line.split()
            word = values[0]
            word_embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = word_embedding
    return word_embeddings

def run_img_labels_preprocessor(root_dir, save_dir, word_embeddings):
    descriptions = get_all_descriptions_labels(root_dir)
    for descript in descriptions:
        if os.path.exists(save_dir + descript):
            print("\nFile already exists:", save_dir + descript)
        else:
            img_label_word_embeddings = get_img_label_word_embeddings(save_dir) #TODO, make this function

def get_img_label_word_embeddings(save_dir):
    pass




#takes the 1000 description "0000xxxxx.pickle" files
def get_all_descriptions_labels(dir):
    files = filter(lambda x: "descriptions" in x, os.listdir(dir))
    return files

def load_pickle_description(filename, dir):
    with open(dir + filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_img_label_embeddings(filename, dir, data):
    with open(dir + filename, "wb") as f:
        pickle.dump(data, f)

def print_status(counter, limit):
    out = "Progress: " + str(counter) + " / " + str(limit)
    sys.stdout.write('\r')
    sys.stdout.write(out)
    sys.stdout.flush()
if __name__ == "__main__":
    #lost_images = get_lost_images(image_dir, dataset)
    #lost_image_labels = get_lost_labels(txt_dir, dataset)
    #word_embeddings = get_word_embeddings_from_glove_file(glove_dir, glove_file)
    #print(len(word_embeddings))
    files = get_all_descriptions_labels(root_dir)
    for file in files:
        print(file)
