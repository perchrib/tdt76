import pickle
import numpy as np
glove_dir = "glove/"
glove_file = "glove.6B.300d.txt"

def get_word_embeddings_from_glove_file(dir, filename):
    word_embeddings = dict()
    counter = 0
    with open(dir + filename) as glove:
        for line in glove:
            values = line.split()
            word = values[0]
            word_embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = word_embedding
    return word_embeddings

def preprocess_


if __name__ == "__main__":
    word_embeddings = get_word_embeddings_from_glove_file(glove_dir, glove_file)
    print(len(word_embeddings))