import os
import random
import shutil


def split_dataset_with_complexities(dataset_path, output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a raster dataset with complexities into train, validation, and test sets. 

    Parameters:
        dataset_path (str): Path to the dataset. Assumes structure dataset_path/class_name/complexity_level/image.svg.
        output_path (str): Path to save the split datasets. Structure will be output_path/split/class_name/complexity_level/image.svg.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)

    # List all class directories
    class_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)

        # List all complexity directories within the class
        complexity_dirs = sorted([d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))])

        for complexity_level in complexity_dirs:
            complexity_path = os.path.join(class_path, complexity_level)
            images = sorted(os.listdir(complexity_path))  # List all images in the complexity directory
            random.shuffle(images)  # Shuffle images for randomness

            # Split indices
            total_images = len(images)
            train_end = int(total_images * train_ratio)
            val_end = train_end + int(total_images * val_ratio)

            # Assign images to splits
            splits = {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:]
            }

            # Copy files to their respective directories
            for split, split_images in splits.items():
                split_complexity_dir = os.path.join(output_path, split, class_name, complexity_level)
                os.makedirs(split_complexity_dir, exist_ok=True)
                for img in split_images:
                    src = os.path.join(complexity_path, img)
                    dst = os.path.join(split_complexity_dir, img)
                    shutil.copy(src, dst)
            print(f"class: {class_name}, complexity: {complexity_level}, finished!")

    print("Dataset split completed!")
    print(f"Train ratio: {train_ratio}, Validation ratio: {val_ratio}, Test ratio: {test_ratio}")


def split_dataset_by_class(dataset_path, output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a dataset into train, validation, and test sets for each class.
    We split the MiniImageNet using this method.

    Parameters:
        dataset_path (str): Path to the dataset. Assumes structure dataset_path/class_name/image.jpg.
        output_path (str): Path to save the split datasets. Structure will be output_path/split/class_name/image.jpg.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)

    # List all class directories
    class_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        images = sorted(os.listdir(class_path))  # List all images in the class directory
        random.shuffle(images)  # Shuffle images for randomness

        # Split indices
        total_images = len(images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        # Assign images to splits
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        # Copy files to their respective directories
        for split, split_images in splits.items():
            split_class_dir = os.path.join(output_path, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy(src, dst)

    print("Dataset split completed!")
    print(f"Train ratio: {train_ratio}, Validation ratio: {val_ratio}, Test ratio: {test_ratio}")


def map_svg_dataset(dataset_path, output_path, svg_dataset_path):
    """
    map the train/val/test set from raster images dataset to svg datasets with different complexities
    :param dataset_path: raster datasets path
    :param output_path: target path
    :param svg_dataset_path: svg dataset path.
    """
    class_dirs = sorted([d for d in os.listdir(svg_dataset_path) if os.path.isdir(os.path.join(svg_dataset_path, d))])
    for set_name in ['train', 'val']:
        src_path = os.path.join(dataset_path, set_name)
        tar_path = os.path.join(output_path, set_name)
        for class_name in class_dirs:
            src_path_c = os.path.join(src_path, class_name)
            images = os.listdir(src_path_c)
            for mode in [0, 1]:
                for shape in [10, 20, 30, 50, 100]:
                    svg_src_path = os.path.join(svg_dataset_path, class_name, f'{shape}_shapes_mode{mode}')
                    tar_path_c = os.path.join(tar_path, class_name, f'{shape}_shapes_mode{mode}')
                    os.makedirs(tar_path_c, exist_ok=True)
                    for image in images:
                        names = image.split('.')
                        if names[-1] == 'jpg':
                            svg_name = names[0] + '.svg'
                            src_img = os.path.join(svg_src_path, svg_name)
                            tar_img = os.path.join(tar_path_c, svg_name)
                            shutil.copy(src_img, tar_img)
            print(f'{class_name} finished!')
        print(f'{set_name} set finished!')
    print('all finished!')

def map_vtracer_dataset(dataset_path, output_path, svg_dataset_path):
    """
    map the train/val/test set from raster images dataset to vtracer datasets with different complexities
    :param dataset_path: raster datasets path
    :param output_path: target path
    :param svg_dataset_path: svg dataset path.
    """
    class_dirs = sorted([d for d in os.listdir(svg_dataset_path) if os.path.isdir(os.path.join(svg_dataset_path, d))])
    for set_name in ['train', 'val','test']:
        src_path = os.path.join(dataset_path, set_name)
        tar_path = os.path.join(output_path, set_name)
        for class_name in class_dirs:
            src_path_c = os.path.join(src_path, class_name)
            images = os.listdir(src_path_c)
            # for mode in [0, 1]:
            #     for shape in [10, 20, 30, 50, 100]:
            svg_src_path = os.path.join(svg_dataset_path, class_name)
            tar_path_c = os.path.join(tar_path, class_name)
            os.makedirs(tar_path_c, exist_ok=True)
            for image in images:
                names = image.split('.')
                if names[-1] == 'JPEG':
                    svg_name = names[0] + '.svg'
                    src_img = os.path.join(svg_src_path, svg_name)
                    tar_img = os.path.join(tar_path_c, svg_name)
                    shutil.copy(src_img, tar_img)
            print(f'{class_name} finished!')
        print(f'{set_name} set finished!')
    print('all finished!')

# # Example usage
# dataset_path = "E:/学习/去雾/data/dataset/miniImageNet"
# output_path = "E:/学习/去雾/data/dataset/miniImageNet_raster_trainvaltest"
# split_dataset_by_class(dataset_path, output_path)

# svg MiniImageNet dataset split
raster_dataset = '/bask/projects/j/jiaoj-multi-modal/miniImageNet_svg/miniImageNet_raster_trainvaltest'
svg_dataset_path = "../miniImageNet_Vtracer"
svg_output_path = "../miniImageNet_Vtracer_trainval"
map_vtracer_dataset(raster_dataset, svg_output_path, svg_dataset_path)
