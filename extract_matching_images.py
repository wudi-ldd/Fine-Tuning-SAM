import os
import shutil

def extract_matching_images(txt_file, source_folder, target_folder):
    # Make sure the target folder exists.
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Read the image names from the text file.
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Extract the image names and remove the file extensions.
        image_name = line.split(':')[0].strip()
        image_name_without_ext = os.path.splitext(image_name)[0]

        # Search for images with the same names in the source folder.
        for filename in os.listdir(source_folder):
            if filename.startswith(image_name_without_ext):
                # Copy the matched images to the target folder.
                shutil.copy2(os.path.join(source_folder, filename), 
                             os.path.join(target_folder, filename))
                print(f"Copied {filename} to {target_folder}")

txt_file = 'VOCdevkit\VOC2007\ImageSets\Segmentation/test.txt'  # TXT file path
source_folder = 'VOCdevkit\VOC2007\SegmentationClass'  # Source folder path
target_folder = 'img/test'  # Target folder path
extract_matching_images(txt_file, source_folder, target_folder)
