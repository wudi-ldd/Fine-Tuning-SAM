import torch
from statistics import mean
from torch.nn.functional import threshold, normalize
import numpy as np
import os
from utils.load_image_names import load_image_names
from utils.json_bbox_extractor import json_bbox_extractor, random_scale_bboxes
from utils.mask_extractor import mask_extractor
from utils.lr import create_cosine_annealing_scheduler
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('logs/runs/loss')
scaler = GradScaler()

# # Define Focal Loss.
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Ensure that alpha is a tensor with weights for each class.
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
            else:
                self.alpha = torch.tensor(alpha). cuda()
        else:
            self.alpha = None

    def forward(self, y_pred, y_true):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(y_pred, y_true)

        if self.alpha is not None:
            # Apply weights for each class.
            alpha = self.alpha[y_true.data.view(-1).long()].view_as(y_true)
            BCE_loss = alpha * BCE_loss

        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()


# Define Dice Loss.
def dice_loss(y_pred, y_true):
    smooth = 1.0
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

# Combine Dice Loss and Focal Loss.
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5,alpha=None):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = dice_loss
        self.focal_loss = FocalLoss(alpha=alpha)

    def forward(self, y_pred, y_true):
        dice_term = self.dice_weight * self.dice_loss(y_pred, y_true)
        focal_term = self.focal_weight * self.focal_loss(y_pred, y_true)
        combined_loss = dice_term + focal_term
        return combined_loss


# Load data and preprocess.
def load_and_preprocess_data(train_dataset_path, bbox_coords,image_data):
    train_image_names = load_image_names(train_dataset_path+'/ImageSets/Segmentation/train.txt')
    val_image_names = load_image_names(train_dataset_path+'/ImageSets/Segmentation/val.txt')
    ground_truth_masks = mask_extractor(bbox_coords, train_dataset_path+'/SegmentationClass')
    # Preprocess the images.
    transformed_data = image_data.preprocess_dataset(bbox_coords)
    return train_image_names, val_image_names, ground_truth_masks, transformed_data

# Training process for a single epoch.
def train_epoch(model, optimizer, loss_fn, train_image_names, rescaled_bbox_coords, transformed_data, ground_truth_masks, device, transform,scheduler,current_epoch):
    model.train()  # Set the model to training mode.
    epoch_losses = []
    # Wrap the training data name list with tqdm.
    with tqdm(enumerate(train_image_names), total=len(train_image_names), desc='Training', leave=True) as train_progress_bar:
        for filename in train_image_names:
            input_image = transformed_data[filename]['image'].to(device)
            input_size = transformed_data[filename]['input_size']
            original_image_size = transformed_data[filename]['original_image_size']
            # Convert filename to a string representation of a key.
            filename=str(filename)
            # Obtain the real mask.
            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[filename], (1, 1, ground_truth_masks[filename].shape[0], ground_truth_masks[filename].shape[1]))).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                image_embedding = model.image_encoder(input_image)
                # Create two empty lists for storing sparse_embeddings and dense_embeddings.
                sparse_embeddings_list = []
                dense_embeddings_list = []
                # Iterate through all the bounding boxes.
                for i in rescaled_bbox_coords.keys():
                    if i.rsplit('_', 1)[0] ==filename:
                        prompt_box = rescaled_bbox_coords[i]
                        box = transform.apply_boxes(prompt_box, original_image_size)
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                        box_torch = box_torch[None, :]
            
                        sparse_embeddings, dense_embeddings = model.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                        # Add the obtained sparse_embeddings and dense_embeddings to their respective lists.
                        sparse_embeddings_list.append(sparse_embeddings)
                        dense_embeddings_list.append(dense_embeddings)
            with autocast():
                # Concatenate all elements in sparse_embeddings_list and dense_embeddings_list.
                sparse_embeddings_all = torch.cat(sparse_embeddings_list, dim=0)
                dense_embeddings_all = torch.cat(dense_embeddings_list, dim=0)
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings_all,
                    dense_prompt_embeddings=dense_embeddings_all,
                    multimask_output=False,
                )
                upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                binary_mask = torch.sum(normalize(threshold(upscaled_masks, 0.0, 0)),dim=0, keepdim=True)
                # Handle overlapping issues in the merged data by setting values greater than 1 to 1.
                binary_mask[binary_mask>1]=1
                # Calculate the loss and perform backpropagation.
                loss = loss_fn(binary_mask, gt_binary_mask)
                # Backpropagation.
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
                # Update the description of the progress bar.
                train_progress_bar.set_description(f"Training (loss: {loss.item():.4f})")
                train_progress_bar.update(1)
        # Update the learning rate.
        scheduler.step()
        # Optional: Print the current learning rate.
        print("Epoch {}: Learning Rate: {:.6f}".format(current_epoch, optimizer.param_groups[0]['lr']))
        avg_loss = mean(epoch_losses)
        print(f'Train Loss: {avg_loss}\n')
        writer.add_scalar('Loss/train', avg_loss, current_epoch)
        return avg_loss

# Validation process for a single epoch.
def validate_epoch(model, loss_fn, val_image_names,rescaled_bbox_coords, transformed_data, ground_truth_masks, device, transform,current_epoch):
    model.eval()  # Set the model to evaluation mode.
    epoch_losses = []
    # Wrap the validation data name list with tqdm.
    with tqdm(enumerate(val_image_names), total=len(val_image_names), desc='Validation', leave=True) as val_progress_bar:
        for filename in val_image_names:
            input_image = transformed_data[filename]['image'].to(device)
            input_size = transformed_data[filename]['input_size']
            original_image_size = transformed_data[filename]['original_image_size']
            # Obtain the real mask for the entire image.
            filename = str(filename)  # Convert filename to a string representation of a key.
            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[filename], (1, 1, ground_truth_masks[filename].shape[0], ground_truth_masks[filename].shape[1]))).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            with torch.no_grad():
                image_embedding = model.image_encoder(input_image)
                # Create two empty lists for storing sparse_embeddings and dense_embeddings.
                sparse_embeddings_list = []
                dense_embeddings_list = []
                # Iterate through all the bounding boxes.
                for i in rescaled_bbox_coords.keys():
                    if i.rsplit('_', 1)[0] ==filename:
                        prompt_box = rescaled_bbox_coords[i]
                        box = transform.apply_boxes(prompt_box, original_image_size)
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                        box_torch = box_torch[None, :]
            
                        sparse_embeddings, dense_embeddings = model.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                        # Add the obtained sparse_embeddings and dense_embeddings to their respective lists.
                        sparse_embeddings_list.append(sparse_embeddings)
                        dense_embeddings_list.append(dense_embeddings)
                # Concatenate all elements in sparse_embeddings_list and dense_embeddings_list.
                sparse_embeddings_all = torch.cat(sparse_embeddings_list, dim=0)
                dense_embeddings_all = torch.cat(dense_embeddings_list, dim=0)
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings_all,
                    dense_prompt_embeddings=dense_embeddings_all,
                    multimask_output=False,
                )
                upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                binary_mask = torch.sum(normalize(threshold(upscaled_masks, 0.0, 0)),dim=0, keepdim=True)
                # Handle overlapping issues in the merged data by setting values greater than 1 to 1.
                binary_mask[binary_mask>1]=1
                # Calculate the loss.
                loss = loss_fn(binary_mask, gt_binary_mask)
                epoch_losses.append(loss.item())
            # Update the description of the progress bar.
            val_progress_bar.set_description(f"Validation (loss: {loss.item():.4f})")
            val_progress_bar.update(1)
        avg_loss = mean(epoch_losses)
        print('\n')
        print(f'Validation Loss: {avg_loss}')
        writer.add_scalar('Loss/val', avg_loss, current_epoch)
        return avg_loss

# Main function for model training and validation.
def fit(model, optimizer, loss, epoch, train_dataset_path, image_data, save_dir, device, save_period,Min_lr,class_weight,label_filter):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    bbox_coords = json_bbox_extractor(train_dataset_path+'/json',label_filter)
    train_image_names, val_image_names, ground_truth_masks, transformed_data = load_and_preprocess_data(train_dataset_path, bbox_coords,image_data)
    transform = ResizeLongestSide(model.image_encoder.img_size)

    if loss=='Dice':
        loss_fn=dice_loss
    elif loss=='MSE':
        loss_fn = torch.nn.MSELoss()
    elif loss=='BCE':
        loss_fn=torch.nn.BCELoss()
    elif loss=='DiceFocal':
        loss_fn=CombinedLoss(dice_weight=0.3, focal_weight=0.7,alpha=class_weight)
    elif loss=='Focal':
        loss_fn=FocalLoss(alpha=class_weight)
    scheduler = create_cosine_annealing_scheduler(optimizer, epoch,Min_lr)
    loss_record_file = os.path.join(save_dir, "epoch_losses.txt")
    for epoch_num in range(epoch):
        print(f'Epoch {epoch_num}')
        original_bbox_coords = random_scale_bboxes(bbox_coords,min_scale=1, max_scale=1)
        rescaled_bbox_coords =random_scale_bboxes(bbox_coords,min_scale=1.1, max_scale=1.1)
        train_loss = train_epoch(model, optimizer, loss_fn, train_image_names, rescaled_bbox_coords, transformed_data, ground_truth_masks, device, transform,scheduler,epoch_num)
        val_loss = validate_epoch(model, loss_fn, val_image_names, original_bbox_coords, transformed_data, ground_truth_masks, device, transform,epoch_num)
        # Record the loss for each epoch.
        with open(loss_record_file, 'w' if epoch_num == 0 else 'a') as file:
            file.write(f'Epoch {epoch_num}: Train Loss: {train_loss}, Validation Loss: {val_loss}\n')

        # Save the model.
        if epoch_num % save_period == 0 :
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch_num}.pth'))
        if val_loss < best_loss:
            best_loss = min(best_loss, val_loss)
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_epoch_{epoch_num}.pth'))