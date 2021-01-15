import pandas as pd
import torch
from engine import *
from test import *
from model import *
from datasets import *

train_csv = '../input/global-wheat-detection/train.csv'
train_dir = '../input/global-wheat-detection/train/'
test_dir = '../input/global-wheat-detection/test/'
test_df = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

df = pd.read_csv(train_csv)

train_img_dict, val_img_dict, train_img_list, val_img_list = get_dict_list()

df_train = create_df(train_img_list)
df_val = create_df(val_img_list)

BATCH_SIZE = 8
ITER_STEP = 100
EPOCHS = 10
model_path = None
INIT_EPOCH = None
DEVICE = torch.device('cuda')

final_score = engine_fn(DEVICE)

def get_best_epoch():
    pprint(final_score)

    # Get the epoch number for weights with best score.
    max_score = -1
    for score in final_score:
        if score[0] > max_score:
            max_score = score[0]
            epoch = score[1]
    print('Best model found in Epoch {}'.format(epoch))
    
    return epoch

epoch = get_best_epoch()

DEVICE = torch.device('cpu')
weight_path = f'./frcnn_best_{epoch}.pth'

model = get_model(2)
model.load_state_dict(torch.load(weight_path))

results = evaluate(DEVICE, model = model)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()

test_df.to_csv('submission.csv', index=False)
