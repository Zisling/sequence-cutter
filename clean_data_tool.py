import os

import cv2 as cv
import numpy as np
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from multiprocessing import Pool


def calc_optical_flow(video):
    """k
    :param video: video
    :type video:np.array
    :return: optical flow as an array of len(video)-1
    :rtype: np.array
    """
    img_array_new = np.zeros_like(video, dtype=np.uint8)
    size = (video.shape[1], video.shape[2])
    for i in range(len(video)):
        new_image = np.zeros((video[i].shape[0], video[i].shape[1], 3), dtype=np.uint8)
        new_image[:, :, 0] = video[i][:, :, 2]
        new_image[:, :, 1] = video[i][:, :, 1]
        new_image[:, :, 2] = video[i][:, :, 0]
        img_array_new[i] = new_image

    first_frame = img_array_new[0]

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255
    out_optical = np.zeros((len(img_array_new) - 1, *size, 3), dtype=np.uint8)
    for i in range(1, len(img_array_new)):
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        frame = img_array_new[i]
        # Opens a new window and displays the input
        # frame
        # cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # from shape (channel , width , height) to (width , height , channel)
        # rgb = rgb.transpose((2, 1, 0))
        # Flip image to back to rgb
        flip_rgb = np.zeros_like(rgb, dtype=np.uint8)
        flip_rgb[:, :, 0] = rgb[:, :, 2]
        flip_rgb[:, :, 1] = rgb[:, :, 1]
        flip_rgb[:, :, 2] = rgb[:, :, 0]

        out_optical[i - 1] = flip_rgb

        # Updates previous frame
        prev_gray = gray
    return out_optical


def find_box_cords(a):
    r = a.any(1)
    if r.any():
        m, n = a.shape
        c = a.any(0)
        out = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
    else:
        out = (0, 0, 0, 0)
    return out


def strip_prosses(filename):
    dirr = 'image_strips/TROOP/'
    im_count = 0
    total = 0
    if filename.endswith('.npy'):
        img, flow, bbox, data = np.load(dirr + filename, allow_pickle=True)
        total += len(img)
        strip_num = 0
        strip_len = 0
        jumped_frame = False
        strip_img = []
        strip_bbox = []
        for i in range(len(img) - 1):
            if strip_len == 11:
                strip_num += 1
                strip_img_op = calc_optical_flow(np.array(strip_img))
                strip_data = np.array([strip_img, strip_img_op, strip_bbox], dtype=np.object)
                np.save('./image_strips/CLEAN/' + str(strip_num).zfill(4) + filename, strip_data)
                # for im in strip_img:
                #     plt.imshow(im)
                #     plt.show()
                jumped_frame = False
                strip_img.clear()
                strip_bbox.clear()
                strip_len = 0

            idxA = i
            idxB = i + 1
            if jumped_frame:
                idxA = i - 1
            xA, x_wA, yA, y_hA = find_box_cords(bbox[idxA][:, :, 0])
            xB, x_wB, yB, y_hB = find_box_cords(bbox[idxB][:, :, 0])
            imageA = img[idxA][xA:x_wA, yA:y_hA]
            imageB = img[idxB][xB:x_wB, yB:y_hB]

            ratioA = imageA.shape[1] / imageA.shape[0]
            ratioB = imageB.shape[1] / imageB.shape[0]
            if ratioA < 0.5 or ratioA > 3 or ratioB < 0.5 or ratioB > 3 or imageB.shape[1] <= 35 or \
                    imageB.shape[0] <= 35:
                strip_len = 0
                jumped_frame = False
                continue

            # interpolation = cv.INTER_CUBIC if imageA.shape[0] > imageB.shape[0] else cv.INTER_AREA
            # imageB = cv.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation=interpolation)
            histA0 = cv.calcHist([imageA], [0], None, [256], [0, 256])[0]
            histB0 = cv.calcHist([imageB], [0], None, [256], [0, 256])[0]
            histA1 = cv.calcHist([imageA], [1], None, [256], [0, 256])[0]
            histB1 = cv.calcHist([imageB], [1], None, [256], [0, 256])[0]
            histA2 = cv.calcHist([imageA], [2], None, [256], [0, 256])[0]
            histB2 = cv.calcHist([imageB], [2], None, [256], [0, 256])[0]
            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned

            # try:
            dist0 = wasserstein_distance(histA0, histB0)
            dist1 = wasserstein_distance(histA1, histB1)
            dist2 = wasserstein_distance(histA2, histB2)
            dist = (dist0 + dist1 + dist2) / 3
            # print(dist)
            # except ValueError:
            #     print(f'Failed on EMD')
            #     strip_len = 0
            #     jumped_frame = False
            #     continue

            if dist < 300:
                strip_img.append(img[idxA])
                strip_bbox.append(bbox[idxA])
                strip_len += 1
                im_count += 1
                jumped_frame = False
                if strip_len == 10:
                    strip_img.append(img[idxB])
                    strip_bbox.append(bbox[idxB])
                    strip_len += 1
                    im_count += 1
                # f, axarr = plt.subplots(1, 2)
                # axarr[0].imshow(imageA)
                # axarr[0].set_title('A for ass')
                # axarr[1].imshow(imageB)
                # axarr[1].set_title('B for butt')
                # f.text(0.5, 0.04,
                #        f'-EMD:{dist:.4f}\nratio A:{ratioA} ratio B:{ratioB}',
                #        ha='center', va='center', size='medium')
                # plt.show()
            else:
                if jumped_frame:
                    strip_len = 0
                    jumped_frame = False
                else:
                    jumped_frame = True
    return im_count, total


if __name__ == '__main__':
    dirr = 'image_strips/TROOP/'
    # for filename in os.listdir(dirr):
    filenames = os.listdir(dirr)
    with Pool(6) as p:
        count_total = p.map(strip_prosses, filenames)
    im_count = sum([ct[0] for ct in count_total])
    total = sum([ct[1] for ct in count_total])
    strip_num = len(count_total)
    print(f'total: {total} im_count:{im_count} strip_num:{strip_num}')
