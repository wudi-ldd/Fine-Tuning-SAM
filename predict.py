import torch
from torch.nn.functional import threshold, normalize
import numpy as np
from utils.load_image_names import load_image_names
from utils.json_bbox_extractor import json_bbox_extractor, random_scale_bboxes
from utils.image_process import ImagePreprocessor
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from matplotlib import pyplot as plt


# Load data and preprocess.
def load_and_preprocess_data(test_dataset_path, bbox_coords,image_data):
    test_image_names = load_image_names(test_dataset_path+'/test.txt')
    # Preprocess the image.
    transformed_data = image_data.preprocess_dataset(bbox_coords)
    return test_image_names,  transformed_data


if __name__ == '__main__':
    #------------------------------------------------------------------#
    # Input the path to the test folder.
    #------------------------------------------------------------------#
    test_dataset_path = 'img'
    #------------------------------------------------------------------#
    # Model type.
    #--------------------------------------------------------------------#
    model_type = 'vit_h'
    #------------------------------------------------------------------#
    model_path = 'logs/best_model_epoch_6.pth'
    #-----------------------------------------------------------------#
    # Device.
    #------------------------------------------------------------------#
    device = 'cuda:0'
    #------------------------------------------------------------------#
    # Predicted classes.
    #------------------------------------------------------------------#
    label_filter='pore'
    #------------------------------------------------------------------#
    # Randomly scale the bounding boxes by a factor.
    #-----------------------------------------------------------------#
    min_scale = 1.4
    max_scale = 1.4
    #------------------------------------------------------------------#
    model = sam_model_registry[model_type](checkpoint=model_path)
    model.to(device)
    model.eval()
    bbox_coords = json_bbox_extractor(test_dataset_path+'/json',label_filter)
    rescaled_bbox_coords =random_scale_bboxes(bbox_coords,min_scale,max_scale)
    image_data = ImagePreprocessor(test_dataset_path, model.image_encoder.img_size,model, device)
    test_image_names, transformed_data = load_and_preprocess_data(test_dataset_path, bbox_coords,image_data)
    transform = ResizeLongestSide(model.image_encoder.img_size)
    

    for filename in test_image_names:
        input_image = transformed_data[filename]['image'].to(device)
        input_size = transformed_data[filename]['input_size']
        original_image_size = transformed_data[filename]['original_image_size']
        # Obtain the real mask for the entire image.
        filename = str(filename)  # Convert filename to a string representation of a key.

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
            # Save the predicted mask as a binary image.
            binary_mask = binary_mask.to('cpu').numpy()
            binary_mask = np.squeeze(binary_mask)
            plt.imsave('img_out/'+filename+'.png',binary_mask,cmap='gray')

