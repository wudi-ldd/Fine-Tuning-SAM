# Fine-tuning SAM with Custom Dataset

This guide explains how to fine-tune the SAM model using a custom dataset with bounding boxes, ideal for complex scenarios. This approach allows for fine-tuning on specific targets within images without considering non-target areas.

## Preparing Your Dataset

1. **Dataset Preparation**: Place annotated images and corresponding JSON files in the `datasets\before` folder. This should include the original images and their JSON files.

2. **Data Conversion**:
   - Run `json_to_dataset.py` or `json_to_dataset_only.py` to generate PNG format annotated masks and JPEG format original images.
     - `json_to_dataset.py` creates semantic segmentation-style annotated masks and original images.
     - `json_to_dataset_only.py` separates the annotated masks within the images. For example, if an image contains 20 cavities, it generates 20 original images and 20 separate cavity PNG masks, along with a JSON file with separated annotation information.
   - Use `datasets\JPEGImages` for original images, `datasets\json` for JSON files, and `datasets\SegmentationClass` for annotated masks.

3. **Preparing VOCdevkit/VOC2007 Folder**:
   - Run `resize.py` to place resized original images and corresponding masks into `VOCdevkit\VOC2007\JPEGImages` and `VOCdevkit\VOC2007\SegmentationClass`.
   - Run `json_scaling.py` to resize JSON files and place them in `VOCdevkit\VOC2007\json`.
   - **Note**: If the width and height of individual training images are inconsistent, the data normalization process of the SAM model can cause the generated masks to misalign with the actual masks, leading to incorrect loss
 calculations. Therefore, it is necessary to normalize each image, mask, and JSON file annotation in `VOCdevkit/VOC2007` to have consistent dimensions before training. You can adjust the dimensions by modifying the `size` 
 parameter in the `resize.py` and `json_scaling.py` files.

4. **Dataset Splitting**: Run `voc_annotation.py` to divide the dataset into training and validation sets. The results will be saved in `VOCdevkit\VOC2007\ImageSets\Segmentation`.

5. **Modifying Training File (train.py)**:
   - `optimizer_type = "adam"`: Choose an optimizer. Adam is currently the most effective.
   - `model_type = 'vit_h'`: Type of pre-trained SAM model to fine-tune. Options include vit_b, vit_l, vit_h.
   - `model_path = "checkpoint/sam_vit_h_4b8939.pth"`: Location of the pre-trained SAM model. Download from [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything).
   - `epoch = 200`: Number of training iterations.
   - `Init_lr = 1e-5`: Initial learning rate. 1e-5 is optimal for small datasets; increase for larger datasets.
   - `save_period = 10`: Frequency of saving model weights.
   - `save_dir = 'logs'`: Path to save training results.
   - `train_dataset_path = 'VOCdevkit/VOC2007'`: Training dataset path. Includes a `json` folder for annotation information.
   - `class_weight`: Set weights for each class if using FocalLoss or DiceFocal as the loss function, useful for balancing data.

6. **Starting Training**: Run `train.py` to begin training.

## Making Predictions

After training the model, use `predict.py` for predictions.

1. Set `test_dataset_path = 'img'`: Path for prediction. `img\JPEGImages` for images, `img\json` for JSON annotation files (either bounding box or semantic segmentation annotations), and `img\test.txt` for image names to predict.
2. `label_filter='pore'`: Specify the target class for prediction.
3. Set `min_scale = 1` and `max_scale = 1` to test model predictions on various bounding box sizes.
4. Run `predict.py` for predictions. Results will be saved in the `img_out` folder.

## Evaluating the Model

1. The `img\test` folder contains the real masks of the test set images, and `img_out` contains the prediction results.
2. Run `Metric Calculation.py` to calculate various evaluation metrics.

## Additional Tools

- `extract_matching_images.py`: Extracts images from a target folder based on image names in a txt file. Useful for data segmentation.
- `convert_labels_color.py`: Converts training masks to the same color as prediction masks.
