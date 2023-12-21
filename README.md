# Fine-tuning Segment Anything Model (SAM) with Custom Dataset

This guide outlines the steps for fine-tuning the SAM model using a custom dataset. This process is particularly useful for complex scenarios involving target-specific training within images. It focuses on excluding non-target areas, allowing for more precise model behavior.

## Table of Contents
- [Preparing Your Dataset](#preparing-your-dataset)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Evaluating the Model](#evaluating-the-model)
- [Additional Tools](#additional-tools)
- [Contact Information](#contact-information)

## Preparing Your Dataset

1. **Dataset Preparation**: Store your annotated images and corresponding JSON files in the `datasets\before` folder. This includes the original images and their JSON annotation files.

2. **Data Conversion**:
   - Execute `json_to_dataset.py` or `json_to_dataset_only.py` for PNG format annotated masks and JPEG format original images.
     - `json_to_dataset.py` generates semantic segmentation-style masks and original images.
     - `json_to_dataset_only.py` produces separate annotated masks for individual entities within an image, alongside corresponding JSON annotations.
   - Original images go to `datasets\JPEGImages`, JSON files to `datasets\json`, and annotated masks to `datasets\SegmentationClass`.

3. **Preparing VOCdevkit/VOC2007 Folder**:
   - Use `resize.py` to resize and place original images and masks in `VOCdevkit\VOC2007\JPEGImages` and `VOCdevkit\VOC2007\SegmentationClass`.
   - Utilize `json_scaling.py` to resize JSON files for `VOCdevkit\VOC2007\json`.
   - **Note**: Inconsistent image dimensions can misalign masks during training. Normalize the size of each image, mask, and JSON annotation in `VOCdevkit/VOC2007` using `resize.py` and `json_scaling.py`.

4. **Dataset Splitting**: Employ `voc_annotation.py` for dividing the dataset into training and validation sets, with results in `VOCdevkit\VOC2007\ImageSets\Segmentation`.

5. **Modifying Training File (train.py)**:
   - Choose optimizer: `optimizer_type = "adam"`.
   - Select SAM model type: `model_type = 'vit_h'`.
   - Specify the model path: `model_path = "checkpoint/sam_vit_h_4b8939.pth"`. Download from [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything).
   - Set training iterations: `epoch = 200`.
   - Initial learning rate: `Init_lr = 1e-5`.
   - Model saving frequency: `save_period = 10`.
   - Training results directory: `save_dir = 'logs'`.
   - Training dataset path: `train_dataset_path = 'VOCdevkit/VOC2007'`.
   - If applicable, set `class_weight` for loss function balancing.

6. **Starting Training**: Launch `train.py` to commence training.

## Making Predictions

Post-training, employ `predict.py` for model predictions.

1. Prediction path setup: `test_dataset_path = 'img'`.
2. Specify target class: `label_filter='pore'`.
3. Test various bounding box sizes: Set `min_scale = 1` and `max_scale = 1`.
4. Execute `predict.py` for predictions, with results in `img_out`.

## Evaluating the Model

1. Real masks of test images are located in `img\test`, while predictions reside in `img_out`.
2. Run `Metric Calculation.py` for various evaluation metrics.

## Additional Tools

- `extract_matching_images.py`: For extracting images from a folder based on txt file listings.
- `convert_labels_color.py`: Aligns training mask colors with prediction mask colors for consistency.

## Contact Information

For any issues or queries regarding the code or dataset, feel free to reach out.

- Email: [ldslidongshen@163.com](mailto:ldslidongshen@163.com)
