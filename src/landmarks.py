import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils

ROOT_PATH = 'Data/'
OUTPUTS = 'Outputs/'

# learning parameters
BATCH_SIZE = 256
LR = 0.0001
EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train/test split
TEST_SPLIT = 0.2

# show dataset keypoint plot
SHOW_DATASET_PLOT = True

resize = 96


class FacialLandmarks(Dataset):
    def __init__(self, data):
        self.data = data
        # get the image pixel column only
        self.pixel_col = self.data.Image
        self.image_pixels = []
        for i in tqdm(range(len(self.data))):
            img = self.pixel_col.iloc[i].split(' ')
            self.image_pixels.append(img)
        self.images = np.array(self.image_pixels, dtype='float32')

    def __getitem__(self, index):
        # reshape the images into their original 96x96 dimensions
        image = self.images[index].reshape(96, 96)
        orig_w, orig_h = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (resize, resize))
        # again reshape to add grayscale channel format
        image = image.reshape(resize, resize, 1)
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))
        # get the keypoints
        keypoints = self.data.iloc[index][:30]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        keypoints = keypoints * [resize / orig_w, resize / orig_h]
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }

    def __len__(self):
        return len(self.images)


def train_test_split(csv_path, split):
    df_data = pd.read_csv(csv_path)
    # drop all the rows with missing values
    df_data = df_data.dropna()
    len_data = len(df_data)
    # calculate the validation data sample length
    valid_split = int(len_data * split)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    training_samples = df_data.iloc[:train_split][:]
    valid_samples = df_data.iloc[-valid_split:][:]
    print(f"Training sample instances: {len(training_samples)}")
    print(f"Validation sample instances: {len(valid_samples)}")
    return training_samples, valid_samples


# get the training and validation data samples
training_samples, valid_samples = train_test_split(f"{ROOT_PATH}FacialKeypoint/training.csv",
                                                   TEST_SPLIT)
# initialize the dataset - `FaceKeypointDataset()`
print('\n-------------- PREPARING DATA --------------\n')
train_data = FacialLandmarks(training_samples)
valid_data = FacialLandmarks(valid_samples)
print('\n-------------- DATA PREPARATION DONE --------------\n')
# prepare data loaders
train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
valid_loader = DataLoader(valid_data,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

# whether to show dataset keypoint plots
if SHOW_DATASET_PLOT:
    utils.plot_dataset(valid_data)


