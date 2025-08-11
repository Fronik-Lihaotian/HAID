import os
import random
from PIL import Image
from svg import Parser, Rasterizer
from torch.utils.data import Dataset


# SVG 2 RGB
class SvgDataset(Dataset):
    def __init__(self, root_paths, num_shapes=100, mode=0, transform=None, scaling=1.):
        """
        Args:
            root_paths (str): path of SVG dataset。
            num_shapes (int): the shapes number of svg files, default 100 (100 shapes)
            mode (int): mode type of shapes, default 0 (all shapes)
            transform (callable, optional): transform of images。
            scaling (float): (0-1], any number not within this range will be changed to 1, default 1
        """
        # TODO scaling training [10%-100%]
        self.scaling = scaling if 1. >= scaling > 1e-5 else 1.
        self.root_paths = root_paths
        self.transform = transform
        self.data = []
        if num_shapes not in [10, 20, 30, 50, 100, 500, 1000]:
            raise ValueError(
                "The number of shapes is only valid at 10, 20, 30, 50, and 100, not " + str(num_shapes)
            )
        if mode not in [0, 1, 2, 5, 7, 8]:
            raise ValueError("The mode number is only valid at 0, 1, 2, 5, 7 and 8, not " + str(mode))

        complexity = f"{num_shapes}_shapes_mode{mode}"  # dir name for specific complexity

        for label, class_dir in enumerate(sorted(os.listdir(self.root_paths))):
            class_path = os.path.join(self.root_paths, class_dir)
            if not os.path.isdir(class_path):
                continue
            for complex_level in os.listdir(class_path):
                if not complex_level == complexity:  # find correct complexity
                    continue
                svg_paths = os.path.join(class_path, complexity)
                files = sorted(os.listdir(svg_paths))
                random.shuffle(files)
                # use different volume of input data to train
                train_end = int(len(files)*self.scaling) if self.scaling < 1. else len(files)
                for file_name in files[:train_end]:
                    if file_name.endswith('.svg'):
                        file_path = os.path.join(svg_paths, file_name)
                        self.data.append((label, file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read SVG files
        label, svg_path = self.data[idx]
        with open(svg_path, "r", encoding="utf-8") as f:
            svg = f.read()

        svg_data = Parser.parse(svg)

        rast = Rasterizer()
        buff = rast.rasterize(svg_data, svg_data.width, svg_data.height)
        image = Image.frombytes('RGBA', (svg_data.width, svg_data.height), buff).convert('RGB')
        # image.show()
        # transform apply
        if self.transform:
            image = self.transform(image)

        return label, image


# # 数据预处理和增强
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # 调整到固定大小
#     transforms.ToTensor(),  # 转为张量并归一化到 [0, 1]
# ])
#
# # 获取所有 SVG 文件路径
# svg_dir = "path_to_svg_files"
# svg_files = [os.path.join(svg_dir, f) for f in os.listdir(svg_dir) if f.endswith('.svg')]
#
# # 创建数据集和数据加载器
# dataset = SvgDataset(svg_paths=svg_files, transform=transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#
# # 使用 DataLoader 进行批量处理
# for batch_idx, images in enumerate(dataloader):
#     print(f"Batch {batch_idx}, Image Tensor Shape: {images.shape}")
#     # 假设每个 batch 的图像形状为 [batch_size, 3, 128, 128]
