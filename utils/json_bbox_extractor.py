from pathlib import Path
import json
import numpy as np

def json_bbox_extractor(json_dir,label_filter):
    bbox_coords = {}
    # Iterate through all the files in the JSON folder.
    for json_file in sorted(Path(json_dir).iterdir()):
        # Get the file name (without the file extension).
        file_name = Path(json_file).stem
        
        # Read the JSON file.
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Parse JSON data and calculate bounding boxes.
        shapes = json_data['shapes']
        for i, shape in enumerate(shapes):
                if shape['label'] == label_filter:
                    points = shape['points']

                    # Extract x and y coordinates.
                    x_coords = [point[0] for point in points]
                    y_coords = [point[1] for point in points]

                    # Calculate the coordinates of the bounding box.
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    # Calculate the area of the mask box.
                    box_area = (x_max - x_min) * (y_max - y_min)
                    if box_area != 0:
                        # Construct the key name (maintaining the same format as the previous code).
                        key = f'{file_name}_{i}'

                        # Store the coordinate information as tuples for easier comparison.
                        bbox_coords[key] = (x_min, y_min, x_max, y_max)

    
    return bbox_coords

def random_scale_bboxes(bbox_coords, min_scale=1, max_scale=1):
    """
    Randomly scale or enlarge bounding box coordinates.

    Parameters:
    bbox_coords (dict): A dictionary containing bounding box coordinates, where the keys are box identifiers and the values are numpy arrays in the form [x_min, y_min, x_max, y_max].
    min_scale (float): The minimum scaling factor.
    max_scale (float): The maximum scaling factor.

    Returns:
    dict: A dictionary containing bounding box coordinates after random scaling or enlargement, with keys matching those in bbox_coords.
    """
    # Generate random scaling or enlargement factors.
    scale_factor = np.random.uniform(min_scale, max_scale)
    scaled_bbox_coords = {}

    for key, value in bbox_coords.items():
        # Calculate the new bounding box coordinates.
        x_min, y_min, x_max, y_max = value
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        new_width = (x_max - x_min) * scale_factor
        new_height = (y_max - y_min) * scale_factor
        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2

        # Update the bounding box coordinates in the dictionary.
        scaled_bbox_coords[key] = np.array([new_x_min, new_y_min, new_x_max, new_y_max])

    return scaled_bbox_coords
