import pickle
import numpy as np
from error_finder import get_lost_images, get_lost_labels
import os
import sys

#Change only dataset to get and store things, when running validate the last pickle file must be deleted manually
dataset = "train/"

glove_dir = "glove/"
glove_file = "glove.6B.300d.txt"
pickle_dir = "pickle/"
image_dir = "pics/"
txt_dir = "txt/"

save_dir = "feature_extraction_1/" + dataset + "label_embeddings/"

root_dir = dataset + pickle_dir

def get_word_embeddings_from_glove_file(dir, filename):
    word_embeddings = dict()
    if os.path.exists(dir + filename[:-4] + ".pickle"):
        print("File already exists:", dir + filename[:-4] + ".pickle")
        with open(dir + filename[:-4] + ".pickle", "rb") as f:
            word_embeddings = pickle.load(f)

    else:
        with open(dir + filename) as glove:
            for line in glove:
                values = line.split()
                word = values[0]
                word_embedding = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = word_embedding
        print(filename)
        with open(dir + filename[:-4] + ".pickle", "wb") as file:
            pickle.dump(word_embeddings, file)
    return word_embeddings

def run_img_labels_preprocessor(root_dir, save_dir, glove_word_vectors):
    descriptions = get_all_descriptions_labels(root_dir)
    counter = 0
    for descript in descriptions:
        if os.path.exists(save_dir + descript):
            print("\nFile already exists:", save_dir + descript)
        else:
            img_description = load_img_description(descript, root_dir)

            word_embedded_img_labels = get_word_embedded_img_label(img_description, glove_word_vectors)
            save_img_label_embeddings(descript, save_dir, word_embedded_img_labels)
            counter += 1
            print_status(counter, len(descriptions))

def get_word_embedded_img_label(img_description, glove_word_vectors):
    word_embedded_img_labels = dict()
    for img_ID in img_description:
        if img_ID not in lost_images and img_ID not in lost_image_labels:

            labels = img_description[img_ID]
            word_vectors = get_word_avg_vector(labels, glove_word_vectors)
            word_embedded_img_labels[img_ID] = word_vectors
    return word_embedded_img_labels

missed_words = dict()
def get_word_avg_vector(labels, glove_word_vectors):
    avg_word_vector = np.zeros(300, dtype='float32')
    for word, value in labels:
        #Finding if a label has two words
        words = word.split()
        for w in words:
            if w in glove_word_vectors:
                label_word_vector = glove_word_vectors[w]
                avg_word_vector += label_word_vector * value
            else:
                if w not in missed_words:
                    missed_words[w] = 1

    return avg_word_vector

#takes the 1000 description "0000xxxxx.pickle" files
def get_all_descriptions_labels(dir):
    files = list(filter(lambda x: "descriptions" in x, os.listdir(dir)))
    return files

def load_img_description(filename, dir):
    with open(dir + filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_img_label_embeddings(filename, dir, data):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename, "wb") as f:
        pickle.dump(data, f)

def print_status(counter, limit):
    out = "Progress: " + str(counter) + " / " + str(limit)
    sys.stdout.write('\r')
    sys.stdout.write(out)
    sys.stdout.flush()

def combine_files():
    combined = dict()
    files = list(filter(lambda x: "0000" in x, os.listdir("feature_extraction/train/label_embeddings/")))
    path = "feature_extraction/train/label_embeddings/"
    for file in files:
        data = load_img_description(file, path)
        for img_id in data:
            combined[img_id] = data[img_id]

    save_img_label_embeddings("combined.pick", path, combined)


if __name__ == "__main__":
    if not os.path.exists("feature_extraction/"):
        os.makedirs("feature_extraction_1/")
    lost_images = get_lost_images(image_dir, dataset)
    lost_image_labels = get_lost_labels(txt_dir, dataset)
    glove_word_embeddings = get_word_embeddings_from_glove_file(glove_dir, glove_file)
    run_img_labels_preprocessor(root_dir, save_dir, glove_word_embeddings)
    combine_files()
    print("\n Words missed: ", len(missed_words))
