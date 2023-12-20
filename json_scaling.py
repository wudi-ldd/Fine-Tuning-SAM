import os
import json

input_folder = 'datasets/json'  # Input folder path, containing original JSON files and corresponding image files.
output_folder = 'VOCdevkit\VOC2007/json'  # Output folder path for saving processed JSON files and resized images.

# Set the width and height of the target images.
TARGET_WIDTH = 1024
TARGET_HEIGHT = 1024

# Create the output folder (if it doesn't exist).
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Retrieve all JSON files from the input folder.
json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

for json_file in json_files:
    # Read the JSON file.
    with open(os.path.join(input_folder, json_file), 'r') as f:
        data = json.load(f)

    # Retrieve the size of the original image.
    image_width = data['imageWidth']
    image_height = data['imageHeight']

    # Calculate the scaling ratio.
    scale_x = TARGET_WIDTH / image_width
    scale_y = TARGET_HEIGHT / image_height

    # Scale the coordinates of each point in the annotation information.
    for shape in data['shapes']:
        for i in range(len(shape['points'])):
            shape['points'][i][0] = int(shape['points'][i][0] * scale_x)
            shape['points'][i][1] = int(shape['points'][i][1] * scale_y)

    # Update the image size information.
    data['imageWidth'] = TARGET_WIDTH
    data['imageHeight'] = TARGET_HEIGHT

    # Save the updated JSON file to the output folder.
    output_json_path = os.path.join(output_folder, json_file)
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)  # Add the indent parameter to specify an indentation of 4 spaces.


print("Processing complete!")
