import os
import re

import cv2
import numpy as np
import torch.utils.data as utils
import torchvision.datasets as dset
from PIL import Image, ImageFilter
from resizeimage import resizeimage
from collections import Counter
import itertools

from torchvision.transforms import transforms


def make_video(imgs, name='video'):
    """
       Make a video from given images. Outputs the video into videos folder.

       @param imgs: a numpy array of images join to a video.
       @param name: the name of the output video.
       """
    dims = imgs[0].ndim
    if dims > 2:
        height, width, layers = imgs[0].shape
    else:
        height, width = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    if not os.path.exists('videos'):
        os.makedirs('videos')
    video = cv2.VideoWriter('./videos/' + name + '.avi', fourcc, 35.0, (width, height))

    for j in range(0, len(imgs)):
        if dims > 2:
            new_image = np.zeros((imgs[j].shape[0], imgs[j].shape[1], 3), dtype=np.uint8)
            new_image[:, :, 0] = imgs[j][:, :, 2]
            new_image[:, :, 1] = imgs[j][:, :, 1]
            new_image[:, :, 2] = imgs[j][:, :, 0]
            video.write(new_image)
        else:
            new_image = np.stack((imgs[j],) * 3, axis=-1)
            video.write(new_image)

    video.release()
    cv2.destroyAllWindows()


def make_image_strip(imgs, name='image_strip', category=0):
    """
    Make a single image strip from given images. Outputs the images the folder.

    @param imgs: a numpy array of images join to a strip.
    @param name: the name of the output image strip.
    @param category: the category of the video
    """
    dims = imgs[0].ndim
    if dims > 2:
        height, width, layers = imgs[0].shape
        image_strip = np.zeros((height, width * len(imgs), layers), dtype=np.uint8)
    else:
        height, width = imgs[0].shape
        image_strip = np.zeros((height, width * len(imgs)), dtype=imgs.dtype)
    for j in range(0, len(imgs)):
        if dims > 2:
            image_strip[0:height, j * width: (j + 1) * width, :] = imgs[j]
        else:
            image_strip[0:height, j * width: (j + 1) * width] = imgs[j]
    if dims > 2:
        image = Image.fromarray(image_strip, 'RGB')
    else:
        image = Image.fromarray(image_strip)

    if not os.path.exists('image_strips'):
        os.makedirs('image_strips')
        os.makedirs('image_strips/0')
        os.makedirs('image_strips/1')
        os.makedirs('image_strips/2')
    image.save('./image_strips/' + str(category) + '/' + name + '.png')


def chunks_torch_dataset(cap, n):
    """
    :param cap: data base to split in to chunks(videos)
    :param n: size of one chunk
    :return: generator of list of spited data base
    """
    for i in range(0, len(cap), n):
        yield utils.Subset(cap, range(i - n, i))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_path(image_id, root, img_type='objects'):
    """
    :param image_id: extract object or depth image base on original image
    :param root: base folder of data sets images
    :param img_type: chose type of image to return if objects return detection object image if depth return depth image
            img_type = 'objects' or 'depth'
    :return: path to the image by image_id
    """
    run_id = int(image_id // 10e8)
    map_id = int((image_id - run_id * 10e8) // 10e6)
    map_id_string = str(map_id)
    if map_id < 10:
        map_id_string = '0' + map_id_string
    tic = int(image_id - run_id * 10e8 - map_id * 10e6)
    string_tic = str(tic)
    zeros = ''
    for i in range(6 - len(string_tic)):
        zeros = '0' + zeros
    return root + '/run' + str(run_id) + '/map' + map_id_string + '/' + img_type + '/' + zeros + str(tic) + '.png'


def chunk_to_objects_images(chunk, root, img_type='objects'):
    """
    :param chunk: a sequence of images (c)
    :param root:base folder of data sets images
    :param img_type: chose type of image to return if objects return detection object image if depth return depth image
            img_type = 'objects' or 'depth'
    :return: list of numpy array of the images
    """
    object_image_paths = []
    prev_id = 0
    for img, target in chunk:
        image_id = target[0]['image_id'] if len(target) > 0 else prev_id + 1
        prev_id = image_id
        if image_id is not None:
            object_image_paths.append(extract_path(image_id, root, img_type=img_type))
        else:
            object_image_paths.append(None)
    images = map(lambda path: np.zeros((200, 320, 3), dtype=np.uint8) if path is None else
    np.array(Image.open(path) if img_type != 'depth' else cv2.imread(path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8),
                 object_image_paths)
    return list(images)


def resize(imgs, shape, sharpen=False):
    reshaped = []

    for img in imgs:
        img = Image.fromarray(img)
        img = resizeimage.resize_cover(img, shape)
        if sharpen:
            img = img.filter(ImageFilter.SHARPEN)
        img = np.array(img)
        reshaped.append(img)

    return np.array(reshaped)


def load_cocodoom_images_paths(path, pic_type='rgb'):
    data = []
    for dirname in sorted(os.listdir(path)):
        if re.search('map', dirname) is not None:
            dir_path = '/'.join((path, dirname, pic_type))
            for filename in sorted(os.listdir(dir_path)):
                image_path = '/'.join((dir_path, filename))
                data.append(image_path)
    return list(data)


def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    return np.array(img, dtype=np.uint8)


def load_image_grey(image_path):
    img = Image.open(image_path)
    return np.array(img)


def load_images(image_paths):
    return np.array(list(map(load_image, image_paths)))


def load_images_grey(image_paths):
    return np.array(list(map(load_image_grey, image_paths)))


def load_images_category(image_paths, data_set):
    category = get_categories(paths=image_paths, dataset=data_set)
    return np.array(list(map(load_image, image_paths))), category


def get_category(target):
    categories = []
    for category in target:
        categories.append(category['category_id'])
    return categories


def path_to_image_id(path):
    run_index = re.search('run', path).end()
    run_id = int(path[run_index])
    map_index = re.search('map', path).end()
    map_id = int(path[map_index: map_index + 2])
    tic_index = re.search("/\d\d\d\d\d\d\.", path).span()
    tic = int(path[tic_index[0] + 1: tic_index[1] - 1])
    return int(10e8 * run_id + 10e6 * map_id + tic), run_id, map_id, tic


def get_target(path, dataset, offset):
    image_id, run_id, map_id, tic = path_to_image_id(path)
    if tic - offset < len(dataset):
        _, target = dataset[tic - offset]
        return target
    else:
        return []


def get_categories(paths, dataset):
    categories = []
    for path in paths:
        offset = 0
        target = get_target(path, dataset, offset)
        if target:
            image_id, *rest = path_to_image_id(path)
            offset = np.abs((int(target[0]['image_id']) % int(1e6)) - (image_id % int(1e6)))
            target = get_target(path, dataset, offset)
            categories.append(get_category(target))
    if categories:  # if no object in the video return 2
        flat_categories = []
        for cat in categories:
            for c in cat:
                flat_categories.append(c)
        count = Counter(flat_categories)
        # if less then object id of monster return 0 else 1
        if count.most_common(1) and count.most_common(1)[0][0] < 23:
            return 0
        else:
            return 1
    else:
        return 2


def main(path, chunk_size=2 * 35, amount=20, chunks_to_take=None, strip=False, video=False, shape=None, sharpen=False):
    dataset = dset.CocoDetection(root='../cocodoom/',
                                 annFile='../cocodoom/run-full-train.json',
                                 transform=transforms.ToTensor())

    data_path_rgb = load_cocodoom_images_paths(path)
    data_path_objects = load_cocodoom_images_paths(path, pic_type='objects')
    data_path_depth = load_cocodoom_images_paths(path, pic_type='depth')
    subsets_paths_rgb = list(chunks(data_path_rgb, chunk_size))
    subsets_paths_objects = list(chunks(data_path_objects, chunk_size))
    subsets_paths_depth = list(chunks(data_path_depth, chunk_size))
    if chunks_to_take:
        choices = chunks_to_take
    else:
        choices = np.random.choice(len(subsets_paths_rgb), amount, replace=True)

    for i in choices:
        vid_rgb, category = load_images_category(subsets_paths_rgb[i], dataset)
        vid_objects = load_images(subsets_paths_objects[i])
        vid_rgb_objects = vid_rgb.copy()
        #  mask the object and the background
        vid_rgb_objects[:, :, :, 0][vid_objects[:, :, :, 2] == 128] = 0
        vid_rgb_objects[:, :, :, 1][vid_objects[:, :, :, 2] == 128] = 0
        vid_rgb_objects[:, :, :, 2][vid_objects[:, :, :, 2] == 128] = 0
        vid_rgb_background = vid_rgb - vid_rgb_objects

        vid_depth = load_images_grey(subsets_paths_depth[i])
        if shape:
            vid_rgb_objects = np.array(
                list(map(lambda x: np.array(resizeimage.resize_cover(Image.fromarray(x, "RGB"), shape)),
                         vid_rgb_objects)))
            vid_rgb_background = np.array(
                list(map(lambda x: np.array(resizeimage.resize_cover(Image.fromarray(x, "RGB"), shape)),
                         vid_rgb_background)))
            vid_depth = np.array(
                list(map(lambda x: np.array(resizeimage.resize_cover(Image.fromarray(x), shape)),
                         vid_depth)))
        if strip:
            make_image_strip(vid_rgb_objects, str(i - 1).zfill(8), category)
            make_image_strip(vid_rgb_background, str(i - 1).zfill(8) + 'back', category)
            make_image_strip(vid_depth, str(i - 1).zfill(8) + 'd', category)
            multi_channel_array = np.concatenate(
                (vid_rgb_objects, vid_rgb_background, vid_depth.reshape((*vid_depth.shape, 1))), axis=-1)
            np.save('./image_strips/' + str(category) + '/' + str(i - 1).zfill(8) + '.npy', multi_channel_array)

        if video:
            make_video(vid_rgb, str(i - 1).zfill(8))
        if not (strip or video):
            # TODO: implement a case for neither a video nor a strip
            pass

        print("Strip no." + str(i) + " is done " + "category " + str(category))


if __name__ == '__main__':
    main('../cocodoom/run1', 4 * 35, amount=100, chunks_to_take=[50], strip=True, video=False, shape=(96, 96))
