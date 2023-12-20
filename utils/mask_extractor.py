import cv2

def mask_extractor(bbox_coords, image_dir='train_data/SegmentationClass'):
    """
    Extracts masks from the provided bounding box coordinates.

    :param unique_bbox_coords: A dictionary of unique bounding box coordinates.
    :param image_dir: Directory containing the corresponding images.
    :return: A dictionary of masks.
    """
    ground_truth_masks = {}
    for file_name in set(k.rsplit('_', 1)[0] for k in bbox_coords):
        image_path = f'{image_dir}/{file_name}.png'
        gt_grayscale = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gt_grayscale is not None:
            ground_truth_masks[file_name] = (gt_grayscale != 0)
        else:
            print(f"Warning: Unable to read image for {file_name}")

    return ground_truth_masks











