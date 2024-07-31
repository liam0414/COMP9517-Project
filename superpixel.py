# SUPERPIXEL ALGORITHM
# from An Improved Semantic Segmentation Method Based on Superpixels and Conditional Random Fields
# (Zhao et al., 2019)
# https://www.mdpi.com/2076-3417/8/5/837

from skimage.segmentation import slic, mark_boundaries
import numpy as np

# Define METAINFO
METAINFO = {
    "classes": (
        "unlabelled",
        "asphalt/concrete",
        "dirt",
        "mud",
        "water",
        "gravel",
        "other-terrain",
        "tree-trunk",
        "tree-foliage",
        "bush/shrub",
        "fence",
        "other-structure",
        "pole",
        "vehicle",
        "rock",
        "log",
        "other-object",
        "sky",
        "grass",
    ),
    "palette": [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ],
    "cidx": list(range(19))
}


def superpixel_run(image, prediction, n_segments=5000):
    # apply SLIC to segment the whole image into K superpixels: segments
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=1)

    # prediction_rgb = label_to_rgb(prediction, METAINFO['palette'])
    prediction_new = prediction.copy()

    # for each superpixel
    for segment in np.unique(segments):
        N = np.count_nonzero(segment) # number of pixels in superpixel
        
        # count number of classes in segment
        segment_rgb = prediction[segments == segment]
        pixels = [tuple(colour) for colour in segment_rgb.reshape(-1, 3)]
        classes = list(set(pixels))
        num_classes = len(classes)

        # initialise all class weights with 0
        weights = {}
        for tup in classes:
            weights[tup] = 0

        # for each pixel
        for p in segment_rgb.reshape(-1, 3):
            pixel = tuple(p)

            # Wcj = oldWcj + 1/N
            weights[pixel] += (1 / num_classes)

            # if Wcj > 0.8, exit this inner loop
            if (weights[pixel] > 0.8):
                break

        # search maximum Wmax and sub-maximum Wsub
        curr_max = 0
        max_label = classes[0]
        for label in weights:
            if weights[label] > curr_max:
                curr_max = weights[label]
                max_label = label

        curr_sub = 0
        sub_max_label = classes[0]
        for label in weights:
            if weights[label] > curr_sub:
                if label != max_label:
                    curr_sub = weights[label]
                    sub_max_label = label

        # if Wmax - Wsub > 0.2, move onto next step
        if (sub_max_label != max_label and  curr_max - curr_sub > 0.2):
            # reassign classification of current superpixel with LCmax
            prediction_new[segments == segment] = max_label
    return prediction_new