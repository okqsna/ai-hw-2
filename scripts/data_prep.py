import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from pycocotools.coco import COCO

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]

def get_data_summary(filepath):
    """
    Function helps to get general information about the COCO dataset.

    :param filepath: path to the dataset
    :return: dict, information about the dataset
    """
    if isinstance(filepath, COCO):
        images = pd.DataFrame(filepath.loadImgs(filepath.getImgIds()))
        clip_column = "video"
    else:
        images = pd.read_csv(filepath)
        clip_column = "clip"

    return {"clips": images[clip_column].nunique(), "frames": len(images),
            "min_width": images["width"].min(), "max_width": images["width"].max(),
            "min_height": images["height"].min(),"max_height": images["height"].max()}

def load_annotations(COCO, split):
    """
    Function helps to load the annotations for a COCO dataset with
    information about frames and bounding boxes.

    :param COCO: class of COCO dataset
    :param split: str, dataset split
    :return: frames, boxes
    """
    frames = pd.DataFrame(COCO.loadImgs(COCO.getImgIds())).rename(columns={"id": "image_id"})
    frames["split"] = split

    annotations = pd.DataFrame(COCO.loadAnns(COCO.getAnnIds()))
    boxes = annotations.merge(frames[["image_id", "file_name",
                                      "video", "width", "height", "split"]], on="image_id",
                              validate="many_to_one")

    class_names = {category["id"]: category["name"] for category in COCO.loadCats(COCO.getCatIds())}

    boxes["class_name"] = boxes["category_id"].map(class_names)
    boxes["bbox_width"], boxes["bbox_height"] = boxes["bbox"].str[2], boxes["bbox"].str[3]
    boxes["bbox_area"] = boxes["bbox_width"] * boxes["bbox_height"]

    object_counts = (boxes.loc[boxes["iscrowd"].eq(0)].groupby("image_id").size())
    frames["objects_per_frame"] = frames["image_id"].map(object_counts).fillna(0).astype(int)

    return frames, boxes

def recalculate_boxes(boxes, width, height):
    """
    Function helps to recalculate the bounding boxes after applied changes.

    :param boxes: bounding boxes of an image
    :param width: width of the image
    :param height:  height of the image
    :return: recalculated bounding boxes
    """
    boxes[["x_min", "y_min", "bbox_width", "bbox_height"]] = pd.DataFrame(
        boxes["bbox"].tolist(),
        columns=["x_min", "y_min", "bbox_width", "bbox_height"], index=boxes.index)

    boxes["x_max"], boxes["y_max"] = boxes["x_min"] + boxes["bbox_width"], boxes["y_min"] + boxes["bbox_height"]
    boxes["bbox_area"] = boxes["bbox_width"] * boxes["bbox_height"]
    boxes["width"], boxes["height"] = width, height

    if "area" in boxes.columns:
        boxes["area"] = boxes["bbox_area"]

    return boxes

def flip_frame(image, boxes,
               horizontal=False, vertical=False):
    """
    Function to flip the frame for more data augmentation

    :param image: frame to be flipped
    :param boxes: bounding boxes of an image
    :param horizontal: bool, horizontal flip
    :param vertical: bool, vertical flip
    :return: changed image and its bounding boxes
    """
    height, width = image.shape[:2]
    flipped_image, flipped_boxes = image.copy(), boxes.copy()

    if horizontal:
        flipped_image = cv2.flip(flipped_image, 1)
        flipped_boxes["bbox"] = [[width - x - w, y, w, h] for x, y, w, h in flipped_boxes["bbox"]]

    if vertical:
        flipped_image = cv2.flip(flipped_image, 0)
        flipped_boxes["bbox"] = [[x, height - y - h, w, h] for x, y, w, h in flipped_boxes["bbox"]]

    flipped_boxes = recalculate_boxes(flipped_boxes, width, height)

    return flipped_image, flipped_boxes

def out_ignored(image, ignored, ignore_index=255):
    """
    Function helps to color the ignored boxes on image.
    :param image: image to be changed
    :param ignored: bounding boxes to be ignored
    :param ignore_index: color to be applied on ignored boxes
    :return: changed image
    """
    result = image.copy()
    height, width = image.shape[:2]

    for box in ignored.itertuples():
        x1, y1 = np.floor([box.x_min, box.y_min]).astype(int)
        x2, y2 = np.ceil([box.x_max, box.y_max]).astype(int)

        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)

        result[y1:y2, x1:x2] = ignore_index

    return result


def prepare_annotations(boxes):
    """
    Function helps to correctly divide the bounding boxes
    of the image into those to be ignored and other.

    :param boxes: bounding boxes of an image
    :return: correct boxes and ignored boxes
    """
    converted_boxes = recalculate_boxes(boxes.copy(), boxes["width"], boxes["height"])
    is_ignored = converted_boxes["iscrowd"].eq(1)
    correct_boxes, ignored_boxes = converted_boxes.loc[~is_ignored].copy(), converted_boxes.loc[is_ignored].copy()

    return correct_boxes, ignored_boxes


def frame_augmentation(image, boxes,
                       horizontal=False, vertical=False,
                       contrast=False, saturation=False):
    """
    Function helps to perform frame augmentation (horizontal or vertical flips,
    apply contrast or saturation)

    :param image: a frame to be augmented
    :param boxes: bounding boxes of the image
    :param horizontal: bool, horizontal flip
    :param vertical: bool, vertical flip
    :param contrast: bool, apply contrast or saturation
    :param saturation: bool, apply saturation
    :return: changed image and its bounding boxes
    """
    augmented_image, augmented_boxes = flip_frame(image, boxes, horizontal=horizontal, vertical=vertical)
    pixels = augmented_image.astype(np.float32)

    if contrast:
        factor = np.random.uniform(0.8, 1.2)
        pixels = pixels * factor + 127.5 * (1 - factor)

    augmented_image = np.clip(pixels, 0, 255).astype(np.uint8)

    if saturation:
        hsv = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2HSV)
        factor = np.random.uniform(0.8, 1.2)

        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return augmented_image, augmented_boxes


def save_fold_yaml(dataset_dir, output_dir, fold, class_names):
    """
    Function helps to save a yaml file for certain
    cross-validation fold (for YOLO26n model)

    :param dataset_dir: path to the dataset
    :param output_dir: path to output directory
    :param fold: number of the fold
    :param class_names:
    :return: path to saved yaml file
    """
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"fold_{fold}.yaml"
    config = {"path": str(dataset_dir),
              "train": str(output_dir / "folds" / f"fold_{fold}_train.txt"),
              "val": str(output_dir / "folds" / f"fold_{fold}_val.txt"), "names": class_names}

    save_yaml(yaml_path, config)

    return yaml_path


def load_yaml(path):
    """
    Function helps to load yaml file of the dataset
    :param path: path to yaml file
    """
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(yaml_path, config_file):
    """
    Function helps to save yaml file.

    :param yaml_path: path to save yaml file
    :param config_file: file parameters to be saved
    return: None
    """
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_file, f, allow_unicode=True, sort_keys=False)
