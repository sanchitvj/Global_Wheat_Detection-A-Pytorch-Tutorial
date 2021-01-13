import torch, time
from tqdm.autonotebook import tqdm
from model import *
from evaluation_metrics import *

def train_fn(dataloader, epoch, model, optimizer, device):
    '''
    Training function.
    Args:
        dataloader: for loading training data batch-wise.
        model: network architecture for training.
        optimizer: optimizer used for gradient descent.
        device: computation device for training.
    Returns:
        loss after every epoch.
    '''
    model.train()  # training mode enables dropout
    
    loss = AverageMeter()  # loss update/reset
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    
    start = time.time()
    
    loader = tqdm(dataloader, total = len(dataloader))
    for step, (images, targets, image_id) in enumerate(loader):
        
        # take the list of images and targets to feed the network
        images = [image.to(device, dtype=torch.float32) for image in images]
        targets = [{k: v.to(device) for k,v in target.items()} for target in targets]
        data_time.update(time.time() - start)

        # forward + backward + optimize
        loss_dict = model(images, targets)
        # loss_dict: {'loss_classifier': tensor(0.6591, device='cuda:0', grad_fn=<NllLossBackward>),
                    # 'loss_box_reg': tensor(0.7574, device='cuda:0', grad_fn=<DivBackward0>),
                    # 'loss_objectness': tensor(0.6313, device='cuda:0',
                    #                           grad_fn=<BinaryCrossEntropyWithLogitsBackward>),
                    #  'loss_rpn_box_reg': tensor(0.1344, device='cuda:0', grad_fn=<DivBackward0>)}
        losses = sum(loss_ind for loss_ind in loss_dict.values())
        
        optimizer.zero_grad()  # zero the parameter gradients
        losses.backward()
        optimizer.step()
        
        batch_time.update(time.time() - start)
        # Update loss of after every batch.
        loss.update(losses.item(), BATCH_SIZE)
        
        start = time.time()
        
        if step % ITER_STEP == 0:
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, step, len(dataloader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=loss))
        # To check the loss real-time while iterating.
        loader.set_postfix(loss=loss.avg)

    return loss
