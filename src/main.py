import pandas as pd
import torch
from engine import *
from test import *
from model import *

train_csv = '../input/global-wheat-detection/train.csv'
train_dir = '../input/global-wheat-detection/train/'
test_dir = '../input/global-wheat-detection/test/'

df = pd.read_csv(train_csv)
# df

BATCH_SIZE = 8
ITER_STEP = 100
EPOCHS = 10
model_path = None
INIT_EPOCH = None
DEVICE = torch.device('cuda')

cv_score = engine_fn(DEVICE)

pprint(cv_score)

# Get the epoch number for weights with best score.
max_score = -1
for score in cv_score:
    if score[0] > max_score:
        max_score = score[0]
        epoch = score[1]
print('Best model found in Epoch {}'.format(epoch))

DEVICE = torch.device('cpu')
weight_path = f'./frcnn_best_{epoch}.pth'

model = get_model(2)
model.load_state_dict(torch.load(weight_path))

results = evaluate(DEVICE, model = model)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()

test_df.to_csv('submission.csv', index=False)
