import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

from ..helpers import NUM_CLASSES, METAINFO, label_to_rgb

# Parameters
IMG_SIZE = 512
BATCH_SIZE = 8

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class WildScene(Dataset):
    def __init__(self, df, img_size, num_classes, transform=None):
        self.df = df
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['im_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        
        label = cv2.imread(row['label_path'], cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (self.img_size, self.img_size))
        
        if self.transform:
            img = self.transform(img)
        
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        label = torch.from_numpy(label).long()
        
        return img, label


# a helper function to 
def shuffle_load_df(path, random_state=42):
    df = pd.read_csv(path)
    sampled_df = df.sample(frac=1, random_state=random_state)
    return sampled_df


PATH = 'dataset/'
def get_unet_df(path=PATH, batch_size=BATCH_SIZE):
    train_df = shuffle_load_df(path + 'train.csv')
    val_df   = shuffle_load_df(path + 'val.csv'  )
    test_df  = shuffle_load_df(path + 'test.csv' )

    train_dataset = WildScene(train_df, IMG_SIZE, NUM_CLASSES)
    val_dataset   = WildScene(val_df  , IMG_SIZE, NUM_CLASSES)
    test_dataset  = WildScene(test_df , IMG_SIZE, NUM_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset  , batch_size=batch_size)
    test_loader  = DataLoader(test_dataset , batch_size=batch_size)

    return train_loader, val_loader, test_loader


def calculate_class_iou(y_true, y_pred, num_classes):
    ious = []
    for cls in range(num_classes):
        true_class = (y_true == cls)
        pred_class = (y_pred == cls)
        intersection = (true_class & pred_class).sum().item()
        union = (true_class | pred_class).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious


# Define function to evaluate the model
def evaluate_model(model, test_loader, classes):
    model.eval()
    class_ious = {cls: 0 for cls in classes}
    merged_classes = []
    merged_ious = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            
            # If model outputs a list, take the last element
            if isinstance(outputs, list):
                outputs = outputs[-1]
                
            preds = torch.argmax(outputs, dim=1)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy().flatten()
                label = label.cpu().numpy().flatten()

                for class_idx in np.unique(label):
                    if class_idx in np.unique(pred):
                        iou = calculate_class_iou(label, pred, len(classes))[class_idx]
                        class_ious[classes[class_idx]] += iou
            # break 

    for class_name in classes:
        if class_name == "pole":
            other_object_idx = classes.index("other-object")
            class_ious[classes[other_object_idx]] += class_ious[class_name]
        elif class_name == "asphalt/concrete":
            other_terrain_idx = classes.index("other-terrain")
            class_ious[classes[other_terrain_idx]] += class_ious[class_name]
        elif class_name not in ["vehicle", "pole", "asphalt/concrete", "unlabelled"]:
            merged_classes.append(class_name)
            merged_ious.append(class_ious[class_name])

    merged_class_ious = dict(zip(merged_classes, merged_ious))
    return merged_class_ious


def print_unetpp_ious(model, test_loader):
    # Evaluate the model
    merged_class_ious = evaluate_model(model, test_loader, METAINFO['classes'])

    # Print IoU results
    print("\nClass-wise IoU:")
    total_iou = 0
    for class_name, iou in sorted(merged_class_ious.items()):
        total_iou += iou / len(test_loader.dataset) * 100
        print(f"Class {class_name:<20} IoU: {iou / len(test_loader.dataset) * 100:.4f}")

    # Calculate Mean IoU
    mean_iou = total_iou / 15
    print(f"\nMean IoU: {mean_iou:.4f}")


def plot_unet_prediction(model, test_loader):
    def visualize_prediction(image, label, prediction, palette):
        image = image.cpu().numpy().transpose((1, 2, 0))
        label = label.cpu().numpy()
        prediction = prediction.cpu().numpy()

        plt.figure(figsize=(15, 15))
        plt.subplot(131)
        plt.imshow(image)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(label_to_rgb(label, palette))
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(label_to_rgb(prediction, palette))
        plt.title("Prediction")
        plt.axis('off')
        plt.show()
    
    with torch.no_grad():
        for data in test_loader:
            if data is None:
                continue

            images, labels = data
            images = images.cuda()

            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]

            outputs = outputs.argmax(1).cpu()

            for i in range(len(images)):
                image = images[i]
                label = labels[i]
                prediction = outputs[i]
                visualize_prediction(image, label, prediction, METAINFO['palette'])
            break
