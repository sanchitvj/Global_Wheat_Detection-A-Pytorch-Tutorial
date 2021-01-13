import numpy as np
import torch
from tqdm.autonotebook import tqdm
from evaluation_metrics import *
from visualize import *
from model import *

def val_fn(dataloader, model, device, display_random=False, show_img_num=None):
    '''
    Validation function with epoch wise visualization.
    Args:
        dataloader: to load the data batch-wise.
        model: trained model for validation.
        device: device used for computation.
        display_random: for visualiztion of random images in every epoch.
        show_img_num: to visualize a particular image, number between 0 and
                      batch size.
    Returns:
        Visualizations and a list of dictionary consisting of predicted box
        coordinates, corresponding scores, ground truth box coordinates and
        image ID.
    '''
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        
        loader = tqdm(dataloader, total=len(dataloader))
        for step, (images, targets, image_id) in enumerate(loader):
            
            images = [image.to(device, dtype=torch.float32) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            output = model(images)

            for i in range(len(images)):
                # tensor.detach() creates a tensor that shares storage with tensor
                # that does not require grad. It detaches the output from the computational
                # graph. So no gradient will be backpropagated along this variable.
                boxes = output[i]['boxes'].detach().cpu().numpy()
                scores = output[i]['scores'].detach().cpu().numpy()

                # boxes_itr = boxes
                all_predictions.append({
                    'pred_boxes': (boxes).astype(int),
                    'scores': scores,
                    'gt_boxes': (targets[i]['boxes'].cpu().numpy()).astype(int),
                    'image_id': image_id[i],
                })

                if display_random:
                    itr = np.random.randint(low=0, high=BATCH_SIZE-1, size=1)
                else:
                    itr = show_img_num

                if step%15==0 and i==itr:

                    gts = (targets[i]['boxes'].cpu().numpy()).astype(int)
                    val_show(gts, df_val, boxes, image_id[i])
                
    return all_predictions
