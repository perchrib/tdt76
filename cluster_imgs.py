from sklearn.cluster import KMeans
import numpy as np
import pickle
import os
import time

root = 'feature_extraction/'
def predict_cluster(img_vector):
    kmeans = load_file('kmeans.pickle')
    cluster = kmeans.predict([img_vector])
    return get_file_string_of_cluster_type(cluster[0])


def find_clusters(dataset='train'):
    img_dir = root + dataset + "/label_embeddings/"
    img_embedding_files = get_files_in_dir(img_dir)
    img_vectors, img_IDs = get_img_vectors_img_ids(img_embedding_files, img_dir)

    if os.path.exists('kmeans.pickle'):
        print("Loading clusters")
        kmeans = load_file('kmeans.pickle')
    else:
        print("Calculate clusters...")
        start = time.time()
        kmeans = KMeans(n_clusters=100, random_state=0).fit(img_vectors)
        end = time.time()

        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print("\n%d h ,%02d min %02d sec" % (h, m, s))

    create_cluster_structure_in_file_system(img_vectors, img_IDs, kmeans)
    save_kmeans(kmeans)
    #print(kmeans.labels_)
    #zero = np.zeros(300, dtype="float32")
    #print("Predict...")
    #print(kmeans.predict([zero]))

def create_cluster_structure_in_file_system(img_vectors, img_Ids, kmeans):
    clusterIds = kmeans.labels_
    distinct_clusters = set(clusterIds)
    if len(img_Ids) != len(img_vectors):
        print("img Ids not match with img vectors")
    else:
        for c in distinct_clusters:
            img_clusters = dict()
            for index in range(len(img_vectors)):
                img_vector = img_vectors[index]
                img_ID = img_Ids[index]
                if clusterIds[index] == c:
                    img_clusters[img_ID] = img_vector
            save_cluster_dict("db_cluster/", c, img_clusters)


def get_img_vectors_img_ids(files, path):
    img_vector = []
    img_ids = []
    for f in files:
        img_dict = load_file(path + f)
        for id in img_dict:
            img_vec = img_dict[id]
            img_vector.append(img_vec)
            img_ids.append(id)
    return np.asarray(img_vector), np.asarray(img_ids)

def get_file_string_of_cluster_type(cluster_type):
    if cluster_type < 10:
        return "00" + str(cluster_type) + ".pickle"
    elif cluster_type < 100:
        return "0" + str(cluster_type) + ".pickle"
    else:
        return str(cluster_type) + ".pickle"

def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_cluster_dict(dir, cluster_type, dict):
    if not os.path.exists(dir):
        os.makedirs(dir)
    f_name = get_file_string_of_cluster_type(cluster_type)
    if not os.path.exists(dir + f_name):
        with open(dir + f_name, "wb") as f:
            pickle.dump(dict, f)

def save_kmeans(kmeans):
    if not os.path.exists("kmeans.pickle"):
        with open("kmeans.pickle", "wb") as km:
            pickle.dump(kmeans, km)


def get_files_in_dir(dir):
    files = os.listdir(dir)
    return files

if __name__ == "__main__":
    find_clusters()