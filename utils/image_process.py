import cv2
import torch
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide

class ImagePreprocessor:
    def __init__(self, train_dataset_path, encoder_img_size, model, device):
        self.image_dir = train_dataset_path+'/JPEGImages'
        self.encoder_img_size = encoder_img_size
        self.device = device
        self.transform = ResizeLongestSide(self.encoder_img_size)
        self.model=model

    def preprocess_image(self,image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.model.preprocess(transformed_image)

        return input_image, image.shape[:2]

    def preprocess_dataset(self, bbox_coords):
        transformed_data = defaultdict(dict)
        processed_files = set()

        for full_file_name in bbox_coords.keys():
            file_name=full_file_name.rsplit('_', 1)[0]  # Remove the file extension.
            if file_name in processed_files:
                continue

            processed_files.add(file_name)
            image_path = f'{self.image_dir}/{file_name}.jpg'
            input_image, original_image_size = self.preprocess_image(image_path)
            input_size = tuple(input_image.shape[-2:])

            transformed_data[file_name]['image'] = input_image
            transformed_data[file_name]['input_size'] = input_size
            transformed_data[file_name]['original_image_size'] = original_image_size

        return transformed_data
    
