import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


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
        (0  , 0  , 0  ),
        (230, 25 , 75 ),
        (60 , 180, 75 ),
        (255, 225, 25 ),
        (0  , 130, 200),
        (145, 30 , 180),
        (70 , 240, 240),
        (240, 50 , 230),
        (210, 245, 60 ),
        (250, 190, 190),
        (0  , 128, 128),
        (170, 110, 40 ),
        (255, 250, 200),
        (128, 0  , 0  ),
        (170, 255, 195),
        (128, 128, 0  ),
        (255, 215, 180),
        (0  , 0  , 128),
        (128, 128, 128),
    ],
    "cidx": list(range(19))
}

# Parameters
IMG_SIZE = 256  # Resized image size
BATCH_SIZE = 8  # Adjust batch size according to available memory
NUM_CLASSES = 19


def load_and_sample_data(file_path, sample_fraction=1):
    df = pd.read_csv(file_path)
    sampled_df = df.sample(frac=sample_fraction, random_state=42)
    return sampled_df

PATH = ''
def get_dfs(sample_fraction=1):
    train_df = load_and_sample_data(PATH + 'train.csv', sample_fraction)
    val_df   = load_and_sample_data(PATH + 'val.csv'  , sample_fraction)
    test_df  = load_and_sample_data(PATH + 'test.csv' , sample_fraction)
    return (train_df, val_df, test_df)

# Data generators for training and validation
def data_generator(df, batch_size, img_size, num_classes):
    while True:
        for start in range(0, len(df), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(df))
            batch_df = df[start:end]
            for _, row in batch_df.iterrows():
                img = cv2.imread(row['im_path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255
                
                label = cv2.imread(row['label_path'], cv2.IMREAD_GRAYSCALE)
                label = cv2.resize(label, (img_size, img_size))
                label = to_categorical(label, num_classes=num_classes)
                
                x_batch.append(img)
                y_batch.append(label)

            yield np.array(x_batch), np.array(y_batch)


# Data EDA, check unique values in label images, we want to make sure that all classes
# are in the set of unique values
# def check_unique_values(df):
#     unique_values = set()
#     for _, row in df.iterrows():
#         label = cv2.imread(row['label_path'], cv2.IMREAD_GRAYSCALE)
#         unique_values.update(np.unique(label))
#     return unique_values

# train_unique_values = check_unique_values(train_df)
# val_unique_values = check_unique_values(val_df)
# test_unique_values = check_unique_values(test_df)

# print("Unique values in train labels:", train_unique_values)
# print("Unique values in val labels:", val_unique_values)
# print("Unique values in test labels:", test_unique_values)

# Jaccard coefficient and loss
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.cast(y_true_f, dtype='float32')
    y_pred_f = K.cast(y_pred_f, dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def label_to_rgb(label, palette):
    rgb_image = np.zeros((*label.shape, 3), dtype=np.uint8)
    for label_idx, color in enumerate(palette):
        rgb_image[label == label_idx] = color
    return rgb_image


def calculate_class_iou(y_true, y_pred, num_classes):
    ious = []
    for cls in range(num_classes):
        true_class = (y_true == cls)
        pred_class = (y_pred == cls)
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        iou = np.nan if union == 0 else intersection / union
        ious.append(iou)
    return ious


# Function to calculate IoUs for each class
def calculate_class_ious(model, test_df, img_size, num_classes):
    test_gen = data_generator(test_df, 1, img_size, num_classes)
    y_preds = []
    y_trues = []
    
    for _ in range(len(test_df)):
        test_images, test_labels = next(test_gen)
        y_pred = model.predict(test_images, verbose=False)
        y_preds.append(y_pred)
        y_trues.append(test_labels)
        
    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    
    y_pred_argmax = np.argmax(y_preds, axis=3)
    y_true_argmax = np.argmax(y_trues, axis=3)
    
    class_ious = calculate_class_iou(y_true_argmax, y_pred_argmax, num_classes)
    return class_ious


def plot_pred_sample(model):
    # Predict on a sample image
    palette = METAINFO['palette']
    sample_image, sample_mask = next(data_generator(test_df, 1, IMG_SIZE, NUM_CLASSES))
    prediction = model.predict(sample_image)[0]

    # visualize the sample image, ground truth and prediction
    sample_image = sample_image[0]
    sample_mask  = sample_mask[0]

    plt.figure(figsize=(15, 15))
    plt.subplot(131)
    plt.imshow(sample_image)
    plt.title("Image")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(label_to_rgb(sample_mask.argmax(axis=2), palette))
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(label_to_rgb(prediction.argmax(axis=2), palette))
    plt.title("Prediction")
    plt.axis('off')
    plt.show()


def display_iou_score(class_ious):
    classes = palette = METAINFO['classes']
    merged_classes = []
    merged_ious = []

    for idx, class_name in enumerate(classes):
        if class_name == "pole":
            # Merge pole IoU into other-object
            other_object_idx = classes.index("other-object")
            class_ious[other_object_idx] += class_ious[idx]
        elif class_name == "asphalt":
            # Merge asphalt IoU into other-terrain
            other_terrain_idx = classes.index("other-terrain")
            class_ious[other_terrain_idx] += class_ious[idx]
        elif class_name not in ["vehicle", "pole", "asphalt", "unlabelled"]:
            merged_classes.append(class_name)
            merged_ious.append(class_ious[idx])

    class_iou_pairs = list(zip(merged_classes, merged_ious))
    sorted_class_iou_pairs = sorted(class_iou_pairs, key=lambda x: x[0])

    for class_name, iou in sorted_class_iou_pairs:
        print(f"Class {class_name} IoU: {iou * 100: .2f}")

    MIou = np.nanmean(merged_ious)
    print(f"Mean IoU: {MIou * 100: .2f}")