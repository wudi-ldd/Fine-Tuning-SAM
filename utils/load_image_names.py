def load_image_names(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()
