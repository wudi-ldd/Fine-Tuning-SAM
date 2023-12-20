import torch
import torch.optim as optim
from segment_anything import sam_model_registry
from utils.utils_fit import fit
from utils.image_process import ImagePreprocessor

if __name__ == "__main__":
    #---------------------------------#
    # Cuda Whether to use CUDA
    # Set to False if you don't have a GPU.
    #---------------------------------#
    device='cuda:0'
    #---------------------------------------------------------------------#
    model_type = 'vit_h'
    device = 'cuda:0'
    model_path  = "checkpoint/sam_vit_h_4b8939.pth"
    #----------------------------------------------------------------------------------------------------------------------------#
    epoch   = 200
    #------------------------------------------------------------------#
    # Predicted classes.
    #------------------------------------------------------------------#
    label_filter='pore'
    #------------------------------------------------------------------#
    # Other training parameters: learning rate, optimizer, and learning rate decay.
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Init_lr Maximum learning rate for the model
    # When using the Adam optimizer, it is recommended to set Init_lr=1e-5
    # Min_lr Minimum learning rate for the model, default is 0.01 times the maximum learning rate.
    #------------------------------------------------------------------#
    Init_lr             =1e-5

    Min_lr              = Init_lr
    #------------------------------------------------------------------#
    # optimizer_type Type of optimizer to be used, options are adam and sgd
    # When using the Adam optimizer, it is recommended to set Init_lr=1e-4
    # When using the SGD optimizer, it is recommended to set Init_lr=1e-2
    # momentum Momentum parameter used internally by the optimizer
    # weight_decay Weight decay to prevent overfitting
    # Weight decay should be set to 0 when using Adam, as it can lead to errors.
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    # save_period How often to save weights, in terms of epochs.
    #------------------------------------------------------------------# 
    save_period         = 10
    #------------------------------------------------------------------#
    # save_dir Folder for saving weights and log files.
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    # Dataset path.
    #------------------------------------------------------------------#
    train_dataset_path  = 'VOCdevkit/VOC2007'
    #------------------------------------------------------------------#
    # Loss-related parameters:
    # Loss function used, options include 'Dice', 'BCE' (Binary Cross-Entropy), 'Focal', 'MSE' (Mean Squared Error), 'DiceFocal', ...
    #------------------------------------------------------------------#
    loss       = 'Dice'
    #------------------------------------------------------------------#
    # class_weight Weight for each class, used for Focal loss.
    #------------------------------------------------------------------#
    class_weight = [0.2,0.8]

    # Start training.
    model = sam_model_registry[model_type](checkpoint=model_path)
    model.to(device)
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.mask_decoder.parameters(), lr=Init_lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.mask_decoder.parameters(), lr=Init_lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.mask_decoder.parameters(), lr=Init_lr , alpha=0.99, eps=1e-8)
    image_data = ImagePreprocessor(train_dataset_path, model.image_encoder.img_size,model, device)

    fit(model,optimizer,loss,epoch,train_dataset_path,image_data,save_dir,device,save_period,Min_lr,class_weight=class_weight,label_filter=label_filter)


    #------------------------------------------------------#

    

