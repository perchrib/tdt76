import pickle
from PIL import Image
import os
import sys

pickle_dir = "pickle/"
image_dir = "pics/"
txt_dir = "txt/"


# noinspection PyPep8

def find_lost_images(path, dataset="train/"):
    all_directories = os.listdir(dataset + path)
    destroyed_images = dict()
    counter = 0
    print("Finding Lost Images:")
    for dir in all_directories:
        file_path = dataset + path + dir + "/"
        if "." in dir:
            pass
        else:
            images = os.listdir(file_path)
            for image in images:
                if image.endswith(".jpg"):
                    # noinspection PyBroadException
                    try:
                        with Image.open(file_path + image) as img:
                            pass
                    except:
                        destroyed_images[image[:-4]] = file_path
                counter += 1
                if counter % (len(all_directories) * len(images) // 10) == 0:
                    out = "\tProgress: " + str(int(counter / (len(all_directories) * len(images)) * 100)) + " / 100 %"
                    sys.stdout.write('\r')
                    sys.stdout.write(out)
                    sys.stdout.flush()
                    #print("\t", int(counter / (len(all_directories) * len(images)) * 100), " / 100")
    sys.stdout.write('\r\tProgress: 100 / 100 %')
    print("\n\t\tLost Images: ", len(destroyed_images))
    return destroyed_images


def find_lost_labels(path, dataset="train/"):
    file_path = dataset + path
    all_files = os.listdir(file_path)
    empty_labels = dict()
    print("Finding Lost Image Labels:")
    for filename in all_files:
        if filename.endswith(".txt"):
            file = open(file_path + filename, "r")
            img_labels = [line.strip().split(";") for line in file]
            suspected_empty_labels = list(filter(lambda img_label: len(img_label) == 2, img_labels))
            for img_label in suspected_empty_labels:
                img_ID = img_label[0]
                if not img_label[1]:
                    empty_labels[img_ID] = file_path + filename
            file.close()

    print("\tLost Labels: ", len(empty_labels))
    return empty_labels


def get_lost_labels(path, dataset="train/"):
    filename = "lost_labels.pickle"
    if os.path.exists(filename):
        return get_pickle_file(filename)
    else:
        data = find_lost_labels(path, dataset)
        save_pickle_file(filename, data)
        return data


def get_lost_images(path, dataset="train/"):
    filename = "lost_images.pickle"
    if os.path.exists(filename):
        return get_pickle_file(filename)
    else:
        data = find_lost_images(path, dataset)
        save_pickle_file(filename, data)
        return data


def save_pickle_file(filename, dict):
    with open(filename, "wb") as f:
        pickle.dump(dict, f)


def get_pickle_file(filename):
    with open(filename, "rb") as file:
        data = pickle.load(file)
        return data
