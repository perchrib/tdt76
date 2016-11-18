import pickle
import random
from PIL import Image


def train(location='./train/'):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that is required for testing the model
    must be saved to file (e.g., pickle) so that the test procedure can load, execute and report
    :param location: The location of the training data folder hierarchy
    :return: nothing
    """
    pass


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
