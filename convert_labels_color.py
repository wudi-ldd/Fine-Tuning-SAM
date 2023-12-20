import os
from PIL import Image

def convert_labels(input_dir, output_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中的所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # 遍历每个图像文件
    for file_name in image_files:
        # 构建输入和输出文件的完整路径
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # 打开图像
        image = Image.open(input_path)

        # 转换图像为 RGB 或者 L 模式
        image = image.convert('RGB')

        # 获取图像的像素数组
        pixels = image.load()

        # 遍历图像的每个像素
        for i in range(image.width):
            for j in range(image.height):
                # 检查像素的颜色是否为 (255, 255, 255) 或者 (255, 255, 255, 255)
                if pixels[i, j] != (0,0,0): #or pixels[i, j] == (255, 255, 255, 255):
                    # 将目标像素的颜色改为 (1, 1, 1)
                    pixels[i, j] = (255, 255, 255)
        # 保存修改后的图像
        image.save(output_path)

    print("转换完成，输出图像保存在:", output_dir)

# 调用函数进行转换
input_folder = "img/test"  # 替换为你的输入目录路径
output_folder = "img/test"  # 替换为你的输出目录路径
convert_labels(input_folder, output_folder)