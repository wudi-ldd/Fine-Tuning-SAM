from PIL import Image
import numpy as np
import os

# Define a function to calculate metrics for a single image
def calculate_metrics(gt_image, pred_image):
    ## Resize images to match the smaller one's dimensions
    if gt_image.size != pred_image.size:
        new_size = (min(gt_image.size[0], pred_image.size[0]), min(gt_image.size[1], pred_image.size[1]))
        gt_image = gt_image.resize(new_size, Image.Resampling.LANCZOS)
        pred_image = pred_image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert images to grayscale if they are not
    if gt_image.mode != 'L':
        gt_image = gt_image.convert('L')
    if pred_image.mode != 'L':
        pred_image = pred_image.convert('L')

    # Load images and convert them to NumPy arrays
    gt = np.array(gt_image)
    pred = np.array(pred_image)

    # Ensure the dimensions of the images are the same
    if gt.shape != pred.shape:
        raise ValueError("Image dimensions do not match")

    # Calculate True Positives
    true_positives = np.sum(np.logical_and(gt == 255, pred == 255))

    # Calculate False Positives
    false_positives = np.sum(np.logical_and(gt == 0, pred == 255))

    # Calculate False Negatives
    false_negatives = np.sum(np.logical_and(gt == 255, pred == 0))

    # Calculate True Negatives
    true_negatives = np.sum(np.logical_and(gt == 0, pred == 0))

    # Calculate IoU (Intersection over Union)
    iou = true_positives / (true_positives + false_positives + false_negatives)

    # Calculate Dice coefficient
    dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

    # Calculate Precision
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision =1

    # Calculate FPR (False Positive Rate)
    fpr = false_positives / (false_positives + true_negatives)

    # Calculate PA
    pa = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)

    return iou, dice, precision, fpr,pa

# Define folder paths
gt_folder = "img/test"
pred_folder = "img_out"

# Get all image filenames in the folder
gt_files = os.listdir(gt_folder)

# Initialize lists to store metrics
ious = []
dices = []
precisions = []
fprs = []
pas=[]

# Iterate over each image file and calculate metrics
for filename in gt_files:
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, filename)

    # Open image files
    # Skip if unable to open
    try:
        gt_image = Image.open(gt_path)
        pred_image = Image.open(pred_path)
    except:
        continue

    iou, dice, precision, fpr, pa = calculate_metrics(gt_image, pred_image)

    ious.append(iou)
    dices.append(dice)
    precisions.append(precision)
    fprs.append(fpr)
    pas.append(pa)

# Calculate averages
mean_iou = np.mean(ious)
mean_dice = np.mean(dices)
mean_precision = np.mean(precisions)
mean_fpr = np.mean(fprs)
mean_pa=np.mean(pa)


# Print the results
print(f"IoU: {mean_iou:.8f}")
print(f"Dice Coefficient: {mean_dice:.8f}")
print(f"Precision: {mean_precision:.8f}")
print(f"False Positive Rate: {mean_fpr:.8f}")
print(f"PA: {mean_pa:.8f}")