import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, cv2, ast
from evaluation_metrics import *


def get_image(img_dir, dataframe, idx=None, image_id=None):
    '''
    Read and output the image in the form of numpy arrays.
    Args:
        img_dir: image directory.
        dataframe: dataframe.
        idx: index to get the image with ground truth boxes before training.
        image_id: image ID, to output the image while validating.
    Returns:
        image in the form of numpy arrays.
    '''
    
    if image_id is None:
        img = os.path.join(img_dir, dataframe['image_id'][idx]) + '.jpg'
    else:
        img = os.path.join(img_dir, image_id) + '.jpg'
                           
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_image(img_dir, dataframe, boxes_pred=None, show=False, idx=None, image_id=None):
    '''
    Function to display the image corresponding to passed arguments.
    Args:
        img_dir: image directory.
        dataframe: dataframe.
        boxes_pred: predicted box coordinates while validation.
        show: displays the image if True.
        idx: index to get the image with ground truth boxes before training.
        image_id: image ID, to output the image while validating.
    Returns:
        if show is True then displays image with ground truth boxes before training
        if boxes_pred is None or if boxes_pred is given then displays image with ground
        truth and predicted boxes.
        if show is False then returns ground truth coordinates and predicted coordinates.
    '''

    if image_id is not None:
        image_id = image_id
    else:
        image_id = df['image_id'][idx]
    img = get_image(img_dir, dataframe, idx, image_id)
    boxes_gt = df[df['image_id'] == image_id]['bbox'].values
    for box in boxes_gt:
        box = ast.literal_eval(box) # https://stackoverflow.com/questions/29552950/when-to-use-ast-literal-eval/29556591
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x + w  #x_max = w
        h = y + h  #y_max = h
        gt_rect = cv2.rectangle(img, (x,y), (w, h), (0,255,0), 3)

    if boxes_pred is not None:
        for box in boxes_pred:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # w = x + w  #x_max = w
            # h = y + h  #y_max = h
            pred_rect = cv2.rectangle(img, (x,y), (w, h), (255,0,0), 2)

    if show:
        if boxes_pred is None:
            plt.figure(figsize=(8,8))
            plt.axis('off')
            plt.title('Image ID: '+image_id)
            plt.imshow(gt_rect)
            plt.show()

        else:
            plt.figure(figsize=(8,8))
            plt.axis('off')
            plt.title('Image ID: '+image_id+'       Green: Ground Truth, Box Count: '+str(len(boxes_gt))
            +'     Red: Predicted, Box Count: '+str(len(boxes_pred)))
            plt.imshow(pred_rect)
            plt.show()

    else:
        if boxes_pred is None:
            return gt_pred
        else:
            return gt_rect, pred_rect

def val_show(gts, dataframe, boxes, image_id):
    '''
    Function to select predicted boxes above threshold and passing that
    as an argument to show_image function.
    Args:
        gts: ground truth box coordinates.
        dataframe: dataframe
        boxes: predicted boxes
        image_id: corresponding image ID.
    Returns:
        arguments for show_image function.
    '''

    ious = np.ones((len(gts), len(boxes))) * -1
    boxes_pred_itr = [] # for all boxes(repetition of boxes)
    boxes_pred = [] # for unique boxes
    for pred_idx in range(len(boxes)):
        best_match_gt_idx = find_best_match(gts, boxes[pred_idx], pred_idx, threshold=0.5, ious=ious)
        boxes_pred_itr.append(boxes[best_match_gt_idx])
    
    # for removing duplicate boxes
    boxes_pred = list(set(tuple(sub) for sub in boxes_pred_itr))
    show_image(train_dir, dataframe, boxes_pred, show=True, image_id=image_id)
