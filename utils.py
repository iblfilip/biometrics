import os
import cv2
import matplotlib.pyplot as plt


def get_path(rel_path):
    """
    Get absolute path based on relative path
    :param rel_path: relative path
    :return: absolute path
    """
    path = os.path.join(os.path.dirname(__file__), rel_path)
    return path


def load_images(path, dataset_name, to_gray=False):
    """
    Load images from dataset
    :param path: Path to image directory
    :param dataset_name: nameof  dataset - training, test
    :param to_gray: mark whether loadimages in grayscale
    :return: imgs: list of images, labels: list of  labels
    """
    file_names = os.listdir(get_path('datasets/' + dataset_name))
    file_names.remove('.DS_Store')

    imgs = []
    labels = []
    for file_name in file_names:
        if to_gray:
            img = cv2.imread(
                get_path(path + '/' + dataset_name + '/' + file_name),
                cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(
                get_path(path + '/' + dataset_name + '/' + file_name))
        imgs.append(img)
        labels.append(file_name.lower().replace('.jpg', ''))
    return imgs, labels


def create_label(test_label, train_label):
    """
    Helper function to return string label if labels match
    :param test_label
    :param train_label
    :return: string label
    """

    if test_label == train_label:
        return 'match_' + test_label
    else:
        return 'non-match_' + test_label


def show_image(img):
    """
    Print image
    :param img: image
    """
    plt.imshow(img), plt.show()


def show_contour(img, cnt):
    """
    Print contours
    :param img: image
    :param cnt: contours
    """
    img_cnt = cv2.drawContours(img, cnt, -1, (100, 100, 0), 3)
    show_image(img_cnt)


def get_shape_type(label):
    """
    Return type of image, face or profile based on image label
    :param label: image label
    :return: string type
    """
    # return type of shape, face or profile, from image name
    if label.endswith('_a'):
        return 'a'
    else:
        return 'b'


def print_accuracy(rank_list, n_items, name):
    """
    Print accuracy for first 3 ranks
    :param rank_list:
    :param n_items:
    :param name:
    """

    print('\n--- {} accuracy ---'.format(name))
    overall_acc = 0
    for i, sum in rank_list.items():
        if i <= 2:
            overall_acc += sum / n_items * 100
            print('{}. rank accuracy: {}'.format(i + 1, overall_acc))


def print_compare_results(rank, scores):
    """
    Print accuracy
    :param rank:
    :param scores:
    """
    print('--------')
    if rank == 0:
        print('>>> same is 1. best, best is {}, same is {}'.format(scores[0][0], scores[0][0]))
    elif rank == 1:
        print('>>> same is 2. best, best is {}, same is {}'.format(scores[0][0], scores[1][0]))
    else:
        print('best {}, same {}'.format(scores[0][0], scores[rank][0]))
