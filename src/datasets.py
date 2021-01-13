import os, cv2, glob, ast, torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Getting a key-value pair to separate training images into
# training data and validation data. Used for dataloading
# purpose.
train_img = {}
val_img = {}
img = glob.glob(train_dir + '/*.jpg')
n = len(img)

# 95% data for training
train_keys = img[:int(0.95*n)]
val_keys = img[int(0.95*n):]

split_dict = {}
for key in train_keys:
    split_dict[key] = 'train'
for key in val_keys:
    split_dict[key] = 'val'
    
for i in img:
    temp = i.split('/')[-1]
    img_name = temp.split('.')[0]
    orig_path = train_dir + img_name.split('_')[0] + '.jpg'
    if (split_dict[orig_path] == 'train'):
        train_img[img_name] = i
    else:
        val_img[img_name] = i

# Appending training and validation image paths to 
# corresponding lists to create separate dataframes.
train_img_list = []
val_img_list = []

for i in img:
    temp = i.split('/')[-1]
    img_name = temp.split('.')[0]
    orig_path = train_dir + img_name.split('_')[0] + '.jpg'
    if (split_dict[orig_path] == 'train'):
        train_img_list.append(img_name)
    else:
        val_img_list.append(img_name)

# Creating a separate dataframe for training images
image_id = []
bbox = []
for i in range(len(train_img_list)):
    
    for img_id, box in zip(df['image_id'].values, df['bbox'].values):
        
        if train_img_list[i] == img_id:
            image_id.append(img_id)
            bbox.append(box)

df_train = pd.DataFrame()
df_train['image_id'] = image_id
df_train['bbox'] = bbox
# df_train

# Creating a separate dataframe for validation images
image_id = []
bbox = []
for i in range(len(val_img_list)):
    
    for img_id, box in zip(df['image_id'].values, df['bbox'].values):
        
        if val_img_list[i] == img_id:
            image_id.append(img_id)
            bbox.append(box)

df_val = pd.DataFrame()
df_val['image_id'] = image_id
df_val['bbox'] = bbox
# df_val

class dataset(Dataset):
    def __init__(self, df, train=True, transforms=None):
        
        self.df = df
        self.train = train
        self.image_ids = self.df['image_id'].unique()
        self.transforms = transforms

    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]
        if self.train:
            img_path = train_img.get(image_id)
        else:
            img_path = val_img.get(image_id)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        img = img / 255.0

        boxes = np.int64(np.array([ast.literal_eval(box) for box in self.df[self.df['image_id'] == image_id]['bbox'].values]))
        boxes[:,2] += boxes[:,0]
        boxes[:,3] += boxes[:,1]

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype = torch.int64)
        target['labels'] = torch.ones((len(boxes,)), dtype = torch.int64)
        target['iscrowd'] = torch.zeros((len(boxes,)), dtype = torch.int64)
        target['area'] = torch.as_tensor(((boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])), dtype = torch.float32)
        target['image_id'] = torch.tensor([idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target, image_id
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]
