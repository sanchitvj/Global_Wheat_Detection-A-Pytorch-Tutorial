import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from train import *
from val import *
from model import *
from evaluation_metrics import *
from datasets import *

# collate_fn is called with a list of data samples at each time.
# It is expected to collate the input samples into a batch for
# yielding from the data loader iterator.
# https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
def collate_fn(batch):
    return tuple(zip(*batch))

def engine(device, model_path=None, init_epoch=None, resume=False):
    '''
    Main funtion to train and validate.
    Args:
        device: device for computation.
        model_path: path of saved model.
        init_epoch: initial epoch to resume training from.
        resume: to resume training from last epoch.
    Return:
        final_score
    '''
    
    final_score = []
    best_score = 0
    
    # Custom DataLoaders
    train_dataset = dataset(df_train, transforms=T.Compose([T.ToTensor()]))
    valid_dataset = dataset(df_val, train=False, transforms=T.Compose([T.ToTensor()]))

    train_loader = DataLoader(train_dataset,
                              BATCH_SIZE,
                              shuffle=False,
                              num_workers=8,
                              collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset,
                            BATCH_SIZE,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=collate_fn, )
    
    if resume:
        model = torch.load(model_path)
        init_epoch = init_epoch
    else:
        model = get_model(2)
        init_epoch = 0
    model.to(device)  # loading model on GPU

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0007)

    for epoch in range(init_epoch, EPOCHS):
        '''
        Call the train function then validation function to take a look on how
        model is performed in that epoch. Output of val_fn, prediction will be
        given to evaluation metrics for getting score.
        '''
        train_loss = train_fn(train_loader, epoch, model, optimizer, device)
        prediction = val_fn(val_loader, model, device, display_random=True)
        valid_score = calculate_final_score(prediction, 0.5, 'pascal_voc')

        if valid_score > best_score:
                best_score = valid_score
                torch.save(model.state_dict(), f'frcnn_best_{epoch}.pth')
#                 torch.save(model, f'frcnn_best_model_epoch_{epoch}') 
        final_score.append([best_score, epoch])
        
    return final_score
