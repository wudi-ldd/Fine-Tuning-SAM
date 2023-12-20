import json
import os
import cv2
from PIL import Image, ImageDraw

def create_mask_image(points, image_size):
    # Create an 8-bit depth black image (pixel values are 0).
    mask = Image.new('L', image_size, 0)
    # Fill the "pore" region with white (pixel values of 255).
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=255)
    return mask


def process_image_and_json(json_file, input_folder, output_image_folder, mask_folder, json_folder):
    with open(json_file, 'r') as file:
        data = json.load(file)

    image_base_name = os.path.splitext(data["imagePath"])[0]
    original_image_path = os.path.join(input_folder, data["imagePath"])
    counter = 1

    for shape in data["shapes"]:
        if shape["label"] == "pore":
            # Generate and save the mask image.
            mask_image = create_mask_image([tuple(point) for point in shape["points"]], (data["imageWidth"], data["imageHeight"]))
            mask_image_path = os.path.join(mask_folder, f"{image_base_name}_{counter}.png")
            mask_image.save(mask_image_path)

            # Copy and save the original image in JPG format.
            original_image = cv2.imread(original_image_path)
            jpg_image_path = os.path.join(output_image_folder, f"{image_base_name}_{counter}.jpg")
            cv2.imwrite(jpg_image_path, original_image)

            # Create a new JSON file containing only the current "pore" annotation.
            new_shape_data = {
                "version": data["version"],
                "imagePath": f"{image_base_name}_{counter}.jpg",
                "imageData": data["imageData"],
                "imageHeight": data["imageHeight"],
                "imageWidth": data["imageWidth"],
                "flags": data["flags"],
                "shapes": [shape],  # Include only the current "pore."
                # You can add any additional necessary fields as needed.
            }
            new_json_file = os.path.join(json_folder, f"{image_base_name}_{counter}.json")
            with open(new_json_file, 'w') as out_file:
                json.dump(new_shape_data, out_file, indent=4)

            counter += 1

def process_all_images(input_folder, output_image_folder, mask_folder, json_folder):
    for file in os.listdir(input_folder):
        if file.endswith('.json'):
            json_file = os.path.join(input_folder, file)
            process_image_and_json(json_file, input_folder, output_image_folder, mask_folder, json_folder)
            print(f"Processed {file}")
    print("Done!")
# Call the function.
process_all_images(
    'datasets/before',  # Input folder.
    'datasets/JPEGImages',       # Output image folder.
    'datasets/SegmentationClass',       # Mask folder.
    'datasets/json'        # JSON folder.
)
