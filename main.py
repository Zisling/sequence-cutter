import os

import cv2
import numpy as np
import torch.utils.data as utils
import torchvision.datasets as dset
from PIL import Image, ImageFilter
from resizeimage import resizeimage


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
    video = cv2.VideoWriter('./videos/' + name + '.avi', fourcc, 20.0, (width, height))

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


def make_image_strip(imgs, name='image_strip'):
    """
    Make a single image strip from given images. Outputs the images the folder.

    @param imgs: a numpy array of images join to a strip.
    @param name: the name of the output image strip.
    """
    dims = imgs[0].ndim
    if dims > 2:
        height, width, layers = imgs[0].shape
        image_strip = np.zeros((height, width * len(imgs), layers), dtype=np.uint8)
    else:
        height, width = imgs[0].shape
        image_strip = None

    for j in range(0, len(imgs)):
        if dims > 2:
            image_strip[0:height, j * width: (j + 1) * width, :] = imgs[j]
        else:
            new_image = np.stack((imgs[j],) * 3, axis=-1)

    image = Image.fromarray(image_strip, 'RGB')

    if not os.path.exists('image_strips'):
        os.makedirs('image_strips')
    image.save('./image_strips/' + name + '.png')


def chunks(cap, n):
    """
    :param cap: data base to split in to chunks(videos)
    :param n: size of one chunk
    :return: generator of list of spited data base
    """
    for i in range(0, len(cap), n):
        yield utils.Subset(cap, range(i - n, i))


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


# def resize(imgs, shape, sharpen=False):
#     reshaped = []
#     # for img in imgs:
#
#         # img = Image.fromarray(img)
#         # img = resizeimage.resize_cover(img, shape)
#         # if sharpen:
#         #     img = img.filter(ImageFilter.SHARPEN)
#         # img = np.array(img)
#         # reshaped.append(img)
#     return np.array(reshaped)


def main(time_tic=2 * 35, amount=20, strip=False, video=False, shape=None, sharpen=False):
    cap = dset.CocoDetection(root='./cocodoom',
                             annFile='./cocodoom/run-full-test.json')

    print('Number of samples: ', len(cap))

    subsets = list(chunks(cap, time_tic))

    # This code is needed for processing none rgb images, such as segmented images
    # depth_vid = chunk_to_objects_images(list(subsets[set_num]), cap.root, img_type='depth')
    # object_vid = chunk_to_objects_images(list(subsets[set_num]), cap.root)

    choices = np.random.choice(len(subsets), amount, replace=True)

    for i in choices:
        vid = np.array(list(map(lambda x: np.array(x[0]), list(subsets[i]))))
        if shape:
            vid = np.array(list(map(lambda x: np.array(resizeimage.resize_cover(x[0], shape)), list(subsets[i]))))
        if strip:
            make_image_strip(vid, str(i - 1).zfill(8))
        if video:
            make_video(vid, str(i - 1).zfill(8))
        if not (strip or video):
            # TODO: implement a case for neither a video nor a strip
            pass

        print("Strip no." + str(i) + " is done")

    # This code is needed for processing none rgb images, such as segmented images
    # make_video(np.array(object_vid), 'object')
    # make_video(np.array(depth_vid), 'depth')


if __name__ == '__main__':
    main(5 * 35, amount=1, strip=True, video=True,)
