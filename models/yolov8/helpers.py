import numpy as np
import cv2
from skimage import morphology, measure
from .. import NUM_CLASSES, label_to_rgb, ORG_HEIGHT, ORG_WIDTH

BIN_PATH = 'bin/'

WIDTH         = 640
HEIGHT        = int(WIDTH * ORG_HEIGHT / ORG_WIDTH)
SHAPE         = (HEIGHT, WIDTH)
CV2_SHAPE     = (WIDTH, HEIGHT)
ORG_SHAPE     = (ORG_HEIGHT, ORG_WIDTH)
CV2_ORG_SHAPE = (ORG_WIDTH, ORG_HEIGHT)


# Prediction and IoU scoring helpers
# -----------------------------------------------------------------

def get_img_and_labels(df):
    images = np.array(df['im_path']).astype(str)
    labels = np.array(df['label_path']).astype(str)
    return images, labels

def resize_image(path, shape=CV2_SHAPE):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, shape)
    return img

def get_corresponding_prediction(path, prefix):
    prediction_path = prefix + path.split('/')[-1]
    prediction = cv2.imread(prediction_path, cv2.IMREAD_UNCHANGED)
    return prediction

def read_img(path, resize=True, shape=CV2_SHAPE):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if resize:
        img = cv2.resize(img, shape)
    return img

def read_batch(batch, resize=True):
    ret = [None] * len(batch)
    for i in range(len(batch)):
        ret[i] = read_img(batch[i], resize=resize)
    return ret


# YOLO folder structure helper
# -----------------------------------------------------------------

def create_yolo_labels(images_dest, labels_dest, df, n_class, create_path=False):
    if not os.path.exists(labels_dest):
        os.makedirs(labels_dest)
    if not os.path.exists(images_dest):
        os.makedirs(images_dest)

    for image_path, label_path in zip(df['im_path'], df['label_path']):
        image = os.path.basename(image_path)

        # Construct the full path to the destination label_txt
        if create_path:
            image_dest = os.path.join(images_dest, image)
            if not os.path.exists(image_dest):
                shutil.copy(image_path, image_dest)

        # create yolo labels
        label_dest = image.replace('.png', '.txt')
        convert_label_to_yolo(
            label_path, f'{labels_dest}\\{label_dest}', n_class
        )


def convert_label_to_yolo(label_path, out_path, n_class):
    label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    height, width = label_img.shape

    with open(out_path, 'w') as label_txt:
        for class_id in range(n_class):
            # Create a binary mask for the current class
            class_mask = cv2.inRange(label_img, class_id, class_id)

            # Find contours for the current class
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if len(cnt) < 3:
                    continue
                normalized_points = [
                    (point[0][0] / width, point[0][1] / height) for point in cnt
                ]
                label_txt.write(
                    f'{class_id} ' + " ".join(
                        f'{x:.6f} {y:.6f}' for x, y in normalized_points
                    ) + "\n"
                )


def run_create_yolo_folder_structure(path):
    datasets = ['train' , 'val' ]
    dfs      = [train_df, val_df]
    for i, ds in enumerate(datasets):
        images_destination = path +  + f'{ds}/images'
        labels_destination = path +  + f'{ds}/labels'
        create_yolo_labels(
            images_destination, labels_destination, dfs[i], NUM_CLASSES,
            create_path=True
        )


# Plotting helper
# -----------------------------------------------------------------

def plot_yolo_predictions(
    model, test_df, indices, title, font_factor=1, save=True
):
    df = test_df.iloc[indices]
    test_inputs, ground_truths = get_img_and_labels(df)

    # row, column, and factoring
    cols = 3
    rows = len(indices)
    y_axis_factor = 1*(rows-1)
    x_axis_factor = 1

    # figure and axis
    fig, ax = plt.subplots(
        rows, cols, figsize=(40*x_axis_factor, 15*y_axis_factor)
    )
    fig.suptitle(title, fontsize=45*font_factor)
    subtitles = ["Original Image", "Ground Truth", "Prediction"]
    images = np.array(df['im_path']).astype(str)
    predictions = model.predict(test_inputs, ret=True)
    imread_rgb = lambda img: cv2.cvtColor(cv2.imread(img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    # for each prediction
    for i in range(rows):
        disp_images = [
            imread_rgb(images[i]),
            imread_rgb(ground_truths[i].replace('indexLabel', 'label')),
            label_to_rgb(predictions[i])
        ]
        for j in range(cols):
            ax[i,j].imshow(disp_images[j])
            ax[i,j].set_title(subtitles[j], fontsize=30*font_factor)
            ax[i,j].axis('off')

    if save:
        fig.savefig(BIN_PATH + title)
    
    del predictions
    del images
    del ground_truths
    gc.collect()


# Post-processing helpers
# -----------------------------------------------------------------

def refine_segmentation(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Smooth the mask using morphological operations
    refined_mask = morphology.opening(mask, morphology.disk(3))
    refined_mask = morphology.closing(refined_mask, morphology.disk(3))
    refined_mask = measure.label(refined_mask)
    return refined_mask


def reassign_unlabeled(prediction, unlabeled_class=0):
    prediction = prediction.astype('int64')
    unlabeled_indices = np.where(prediction == unlabeled_class)
    for y, x in zip(*unlabeled_indices):
        # Get the 8-connected neighbors, remove unlabeled from neighbors
        neighbors = prediction[max(0, y-1):y+2, max(0, x-1):x+2].flatten()
        neighbors = neighbors[neighbors != unlabeled_class]
        if len(neighbors) == 0:
            continue
        # Assign the most frequent class among the neighbors to the unlabeled pixel
        prediction[y, x] = np.bincount(neighbors).argmax()
    return prediction


def convert_instance_to_semantic(results, num_classes):
    _, height, width = results[0].masks.data.cpu().numpy().shape  # Get the size of the input image
    semantic_map = np.zeros((height, width), dtype=np.uint8)
    
    # Iterate through each detected instance
    for result in results:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        labels = boxes.cls.cpu().numpy()

        for label, mask, box in zip(labels, masks, boxes.xyxy):
            [ymin, xmin, ymax, xmax] = box
            ymin = int(max(0, ymin))
            xmin = int(max(0, xmin))
            ymax = int(min(height, ymax))
            xmax = int(min(width, xmax))
            semantic_map[mask != 0] = label
            semantic_map[ymin:ymax, xmin:xmax] = label # remove?

    segment_color = label_to_rgb(semantic_map)
    return segment_color


def most_frequent(arr, excl=0):
    bincounts = np.bincount(arr[arr != excl])
    bincounts[excl] = 0
    return np.argmax(bincounts)


def replace_border_black_regions(mask):
    # # mask = np.copy(input_mask)
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[ 0, :] = True
    border_mask[-1, :] = True
    border_mask[ :, 0] = True
    border_mask[ :,-1] = True

    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1],
    ])

    border_black = np.logical_and((mask == 0) & border_mask, border_mask)
    labeled, num_labels = measure.label(mask == 0, connectivity=2, return_num=True)
    # print(labeled, num_labels)
    in_bound = lambda y, x, mat: y < mat.shape[0] and y >= 0 and x < mat.shape[0] and x >= 0

    for region_id in range(1, num_labels+1):
        region_mask = labeled == region_id
        bincount = np.zeros(mask.shape, dtype='int64')
        counted = False
        for y, x in zip(*np.where(region_mask)):            
            neighbors = np.array([
                mask[y+r][x+c] for r,c in offsets if in_bound(y+r, x+c, mask)
            ])
            neighbors = neighbors[neighbors != 0]
            if len(neighbors) == 0:
                continue
            counted = True
            bincount[y][x] = np.bincount(neighbors).argmax()
        if counted:
            mask[region_mask] = most_frequent(bincount)