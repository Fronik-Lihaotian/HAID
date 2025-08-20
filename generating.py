"""

This is the generating script we used to generate HAID, you can generate your own svg dataset by this script.

Before running this scrpt, please set up the CLI tool of Primitive in advance, the Primitive github project 
explained the details of how to do that (https://github.com/fogleman/primitive?tab=readme-ov-file#command-line-usage).

"""



import os
import subprocess
import json
from PIL import Image

STATUS_FILE = "progres.json"  # Progress saving
INPUT_DIR_NAME = "miniImageNet"  # Name of your pixel images (png, jpg, jpeg) dataset
OUTPUT_DIR_NAME = f"{INPUT_DIR_NAME}_svg"

def load_progress():
    """load progress saved"""
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(data):
    """save current progress"""
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def vectorize_image(input_path, output_path, num_shapes=100, mode=1):
    """
    Call Primitives CLI to generate a single image
    :param input_path: input image path
    :param output_path: output image path (.svg)
    :param num_shapes: abstract level
    :param mode: the type of primitives used to generate the images
    """
    try:
        # the parameters setting follows https://github.com/fogleman/primitive?tab=readme-ov-file#command-line-usage
        command = [
            "./go/bin/primitive",  # if you're using Windows，you may need to use "primitive.exe"
            "-i", input_path,
            "-o", output_path,
            "-n", str(num_shapes),
            # "-nth", str(500),  
            "-m", str(mode),
            "-a", str(128),    # alpha, default 128
            "-s", str(max(Image.open(input_path).size[0], Image.open(input_path).size[1]))
        ]
        subprocess.run(command, check=True)
        print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def batch_vectorize(input_dir, output_dir, num_shapes=100, mode=1, progress={}):
    """
    batch generating, start from the interrupt point (STATUS_FILE)
    :param input_dir: input folder
    :param output_dir: output folder
    :param num_shapes: abstract level for each images
    :param mode: type of primitive
    :param progress: progress state
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_from = progress.get("last_file", "")  # Get the file processed last time
    start_found = start_from == ""              # if skip the processed files

    for file_name in sorted(os.listdir(input_dir)):  
        input_path = os.path.join(input_dir, file_name)
        if not (os.path.isfile(input_path) and file_name.lower().endswith((".png", ".jpg", ".jpeg"))):
            continue

        # skip processed folder
        if not start_found:
            if file_name == start_from:
                start_found = True
            continue

        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + "-%d.svg")
        vectorize_image(input_path, output_path, num_shapes, mode)

        # update progress
        progress["last_file"] = file_name
        save_progress(progress)

def get_dir(input_path):
    dirs = []
    for item in os.scandir(input_path):
        if item.is_dir() and item.name != '.ipynb_checkpoints':
            dirs.append(item.name)
    return dirs


if __name__ == "__main__":
    input_folder = f"./Datasets/{INPUT_DIR_NAME}"  # raster image dataset folder
    dir_list = get_dir(input_folder)
    print(dir_list)

    # load progress status
    progress = load_progress()
    last_dir = progress.get("last_dir", "")  # processed folder last time
    last_shapes = progress.get("last_shapes", None)
    last_mode = progress.get("last_mode", None)
    dir_found = last_dir == ""

    for dir_class in dir_list:
        # skip processed folder
        if not dir_found:
            if not dir_class == last_dir:
                continue
            else:
                dir_found = True

        input_sub_folder = os.path.join(input_folder, dir_class)

        shapes = 100
        for mode in [0, 1]:
            if last_mode is not None and mode < last_mode:
                continue

            output_folder = f"./Datasets/{OUTPUT_DIR_NAME}/{dir_class}/mode{mode}"
            print(f"Processing: {dir_class}, Mode: {mode}")

            # update progress
            progress = {
                "last_dir": dir_class,
                "last_shapes": shapes,
                "last_mode": mode,
                "last_file": progress.get("last_file", "")  # maintain current status
            }
            save_progress(progress)

            # batch process
            batch_vectorize(input_sub_folder, output_folder, shapes, mode, progress)

            # finished, reset progress
            progress["last_file"] = ""
            save_progress(progress)

            last_mode = None  # reset mode
        last_shapes = None  # reset abstract level

    print("All tasks completed.")

