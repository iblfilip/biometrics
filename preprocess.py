import cv2
import numpy as np

import utils


PREPROCESS_PATH = 'preprocessed'
TEMPLATE_PATH = 'templates'


def resize(img, gray_scale=False):
    """
    Resize image to smaller size
    :param img: image to process
    :param gray_scale: mark if image is grayscaled
    :return: processed image
    """

    if gray_scale:
        (h, w) = img.shape
    else:
        (h, w, d) = img.shape

    r = 500.0 / w
    dim = (500, int(h * r))
    return cv2.resize(img, dim)


def crop(img):
    """
    Crop img to leave only parts with green screen background
    :param img: image to process
    :return: processed image
    """
    return img[350:2000, 900:1500]


def make_mask(img):
    """
    Mask image, remove green channel
    :param img: image to process
    :return: processed image
    """

    lower_green = np.array([0, 120, 0])  # [R value, G value, B value]
    upper_green = np.array([120, 255, 120])
    return cv2.inRange(img, lower_green, upper_green)


def cut_by_template(img, template):
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img_cut = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return img_cut


def save_cut_masks_small_images(imgs, labels, dataset_name):
    """
    Pre-process and re-save body masked images. Crop, resize, mask image and conduct template matching.
    :param imgs: list of images
    :param labels: list of labels
    :param dataset_name
    """

    print('Pre-processing and re-saving masked body shape preprocessed, dataset {}'.format(dataset_name))
    template = cv2.imread(utils.get_path(TEMPLATE_PATH + '/template_masked_body.jpg'), cv2.IMREAD_GRAYSCALE)

    for (img, name) in zip(imgs, labels):
        img_res = crop(img)
        img_res = resize(img_res)
        img_res = make_mask(img_res)
        img_res = cut_by_template(img_res, template)
        cv2.imwrite(utils.get_path(PREPROCESS_PATH + '/masked_body/' + dataset_name + '/' + name + '.jpg'),
                    img_res)


def save_cut_masks_head_images(imgs, labels, dataset_name):
    """
    Pre-process nad re-save head masked preprocessed images. Crop, resize image and apply template matching
    use template a) for face, b)  for profile images
    :param imgs: list of images
    :param labels: list of labels
    :param dataset_name
    """

    print('Pre-processing and re-saving masked head shapes, dataset {}'.format(dataset_name))
    templates = {
        'a': cv2.imread(utils.get_path(TEMPLATE_PATH + '/template_masked_head_a.jpg'), cv2.IMREAD_GRAYSCALE),
        'b': cv2.imread(utils.get_path(TEMPLATE_PATH + '/template_masked_head_b.jpg'), cv2.IMREAD_GRAYSCALE)}

    for (img, name) in zip(imgs, labels):
        if '_a' in name:
            profile = 'a'
        else:
            profile = 'b'

        img_res = crop(img)
        img_res = resize(img_res)
        img_res = make_mask(img_res)
        img_res = cut_by_template(img_res, templates[profile])
        cv2.imwrite(utils.get_path(PREPROCESS_PATH + '/masked_head/' + dataset_name + '/' + name + '.jpg'),
                    img_res)


def save_cut_small_images(imgs, labels, dataset_name):
    """
    Pre-process nad re-save body preprocessed images. Crop, resize and apply template matching.
    :param imgs: list of images
    :param labels: list of labels
    :param dataset_name
    """

    print('Pre-processing and re-saving small body preprocessed, dataset {}'.format(dataset_name))
    template = cv2.imread(utils.get_path(TEMPLATE_PATH + '/template_body.jpg'), cv2.IMREAD_GRAYSCALE)

    for (img, name) in zip(imgs, labels):
        img_res = crop(img)
        img_res = resize(img_res, gray_scale=True)
        img_res = cut_by_template(img_res, template)
        cv2.imwrite(utils.get_path(PREPROCESS_PATH + '/body/' + dataset_name + '/' + name + '.jpg'),
                    img_res)


def main():
    print('Pre-processing of image files')
    print('OpenCV version {}'.format(cv2.__version__))

    imgs_train_gray, labels_train = utils.load_images('datasets', 'training', to_gray=True)
    imgs_test_gray, labels_test = utils.load_images('datasets', 'test', to_gray=True)
    imgs_train, _ = utils.load_images('datasets', 'training')
    imgs_test, _ = utils.load_images('datasets', 'test')

    save_cut_masks_small_images(imgs_train, labels_train, 'training')
    save_cut_masks_small_images(imgs_test, labels_test, 'test')

    save_cut_small_images(imgs_train_gray, labels_train, 'training')
    save_cut_small_images(imgs_test_gray, labels_test, 'test')

    save_cut_masks_head_images(imgs_train, labels_train, 'training')
    save_cut_masks_head_images(imgs_test, labels_test, 'test')


if __name__ == "__main__":
    main()
