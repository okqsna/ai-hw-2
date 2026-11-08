import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from pycocotools.coco import COCO

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]

def get_data_summary(filepath):
    images = None
    clip_column = None

    if isinstance(filepath, COCO):
        images = pd.DataFrame(filepath.loadImgs(filepath.getImgIds()))
        clip_column = "video"
    else:
        images = pd.read_csv(filepath)
        clip_column = "clip"

    return {"clips": images[clip_column].nunique(),
            "frames": len(images),
            "min_width": images["width"].min(),
            "max_width": images["width"].max(),
            "min_height": images["height"].min(),
            "max_height": images["height"].max(),
    }

def load_annotations(coco, split):
    frames = pd.DataFrame(coco.loadImgs(coco.getImgIds())).rename(columns={"id": "image_id"})
    frames["split"] = split

    annotations = pd.DataFrame(coco.loadAnns(coco.getAnnIds()))

    boxes = annotations.merge(frames[["image_id", "file_name", "video", "width", "height", "split"]], on="image_id",
                              validate="many_to_one")

    class_names = {category["id"]: category["name"] for category in coco.loadCats(coco.getCatIds())}

    boxes["class_name"] = boxes["category_id"].map(class_names)
    boxes["bbox_width"] = boxes["bbox"].str[2]
    boxes["bbox_height"] = boxes["bbox"].str[3]
    boxes["bbox_area"] = boxes["bbox_width"] * boxes["bbox_height"]

    object_counts = (boxes.loc[boxes["iscrowd"].eq(0)].groupby("image_id").size())
    frames["objects_per_frame"] = frames["image_id"].map(object_counts).fillna(0).astype(int)

    return frames, boxes

def recalculate_boxes(boxes, width, height):
    boxes[["x_min", "y_min", "bbox_width", "bbox_height"]] = pd.DataFrame(
        boxes["bbox"].tolist(), columns=["x_min", "y_min", "bbox_width", "bbox_height"], index=boxes.index)
    boxes["x_max"] = boxes["x_min"] + boxes["bbox_width"]
    boxes["y_max"] = boxes["y_min"] + boxes["bbox_height"]
    boxes["bbox_area"] = boxes["bbox_width"] * boxes["bbox_height"]
    boxes["width"], boxes["height"] = width, height
    if "area" in boxes.columns:
        boxes["area"] = boxes["bbox_area"]
    if "bbox_area_pct" in boxes.columns:
        boxes["bbox_area_pct"] = 100 * boxes["bbox_area"] / (width * height)
    return boxes

def prepare_annotations(boxes):
    converted_boxes = recalculate_boxes(boxes.copy(), boxes["width"], boxes["height"])
    is_ignored = converted_boxes["iscrowd"].eq(1)
    correct_boxes, ignored_boxes = converted_boxes.loc[~is_ignored].copy(), converted_boxes.loc[is_ignored].copy()

    return correct_boxes, ignored_boxes

def resize_frame(image, boxes, scale=0.5):
    old_height, old_width = image.shape[:2]
    new_width = max(1, round(old_width * scale))
    new_height = max(1, round(old_height * scale))

    resized_image = cv2.resize(image,(new_width, new_height), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = new_width / old_width, new_height / old_height

    resized_boxes = boxes.copy()
    resized_boxes["bbox"] = [[x * scale_x, y * scale_y, w * scale_x, h * scale_y] 
                             for x, y, w, h in boxes["bbox"]]
    resized_boxes = recalculate_boxes(resized_boxes, new_width, new_height)

    return resized_image, resized_boxes

def flip_frame(image, boxes, horizontal=False, vertical=False):
    height, width = image.shape[:2]
    flipped_image = image.copy()
    flipped_boxes = boxes.copy()

    if horizontal:
        flipped_image = cv2.flip(flipped_image, 1)
        flipped_boxes["bbox"] = [[width - x - w, y, w, h] for x, y, w, h in flipped_boxes["bbox"]]

    if vertical:
        flipped_image = cv2.flip(flipped_image, 0)
        flipped_boxes["bbox"] = [[x, height - y - h, w, h] for x, y, w, h in flipped_boxes["bbox"]]

    flipped_boxes = recalculate_boxes(flipped_boxes, width, height)

    return flipped_image, flipped_boxes

def out_ignored(image, ignored, ignore_index=255):
    result = image.copy()
    height, width = image.shape[:2]

    for box in ignored.itertuples():
        x1, y1 = np.floor([box.x_min, box.y_min]).astype(int)
        x2, y2 = np.ceil([box.x_max, box.y_max]).astype(int)

        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)

        result[y1:y2, x1:x2] = ignore_index
    return result

def frame_augmentation(image, boxes, horizontal=False, vertical=False, 
                       brightness=False, contrast=False, saturation=False, blur=False):
    augmented_image, augmented_boxes = flip_frame(image, boxes, horizontal=horizontal, vertical=vertical)
    pixels = augmented_image.astype(np.float32)

    if contrast:
        factor = np.random.uniform(0.85, 1.15)
        pixels = (pixels - 127.5) * factor + 127.5

    if brightness:
        offset = np.random.uniform(-0.1, 0.1) * 255
        pixels = pixels + offset

    augmented_image = np.clip(pixels, 0, 255).astype(np.uint8)

    if saturation:
        hsv = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2HSV)
        factor = np.random.uniform(0.9, 1.1)

        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if blur:
        augmented_image = cv2.GaussianBlur(augmented_image, (3, 3), sigmaX=0.5)

    return augmented_image, augmented_boxes


def save_fold_yaml(dataset_dir, output_dir, fold, class_names):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"fold_{fold}.yaml"
    config = {"path": str(dataset_dir), "train": str(output_dir / "folds" / f"fold_{fold}_train.txt"), "val": str(output_dir / "folds" / f"fold_{fold}_val.txt"), "names": class_names}
    
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return yaml_path

def load_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)