import cv2, torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from visualize import *
from model import *

test_df = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

# Dataset class for evaluation.
class eval_dataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        img = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0

        if self.transforms is not None:
            img = self.transforms(img)

        return img, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

# Custom dataloader for test data.
test_dataset = eval_dataset(test_df, test_dir, transforms=T.Compose([T.ToTensor()]))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False, collate_fn=collate_fn)

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

def evaluate(device, model):
    
    model.eval()
    detection_threshold = 0.5
    results = []

    for images, image_ids in test_loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):

            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            img = get_image(test_dir, test_df, image_id=image_id)

            for box in boxes:
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                rect = cv2.rectangle(img, (x,y), (w,h), (220, 0, 0), 3)

            plt.figure(figsize=(8,8))
            plt.axis('off')
            plt.title('Image ID: '+image_id)
            plt.imshow(rect)
            plt.show()

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }

            results.append(result)
            
    return results
