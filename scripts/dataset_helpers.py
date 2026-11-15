import cv2
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from .data_prep import prepare_annotations, frame_augmentation, out_ignored

def prepare_sample(directory, annotations_by_image,
                   frame, empty_annotations,
                   validation=False, horizontal=True,
                   vertical=True, contrast=False, saturation=False):
    """
    Function helps to prepare image according to the certain data
    augmentations and dataset type

    :param directory: path to data directory
    :param annotations_by_image: all image annotations
    :param frame: certain frame
    :param empty_annotations: frames with no annotations
    :param validation: dataset type
    :param horizontal: horizontal flip
    :param vertical: vertical flip
    :param contrast: changed contrast
    :param saturation: changed saturation
    :return: image in Tensor form,
    """
    image_path = directory / frame["split"] / frame["file_name"]
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    annotations = annotations_by_image.get(frame["image_id"], empty_annotations).copy()

    if not validation:
        image, annotations = frame_augmentation(image,
            annotations,
            horizontal=horizontal and torch.rand(()).item() < 0.5,
            vertical=vertical and torch.rand(()).item() < 0.5,
            contrast=contrast and torch.rand(()).item() < 0.5,
            saturation=saturation and torch.rand(()).item() < 0.5)

    objects, ignored = prepare_annotations(annotations)

    if not validation:
        image = out_ignored(image, ignored)

    coordinates = ["x_min", "y_min", "x_max", "y_max"]

    target = {"image_id": torch.tensor(int(frame["image_id"]), dtype=torch.int64),
              "labels": torch.tensor(objects["category_id"].tolist(), dtype=torch.int64),
              "boxes": torch.tensor(objects[coordinates].to_numpy().tolist(), dtype=torch.float32).reshape(-1, 4)}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor(image), target


class DetectionDataset(Dataset):
    """
    Helper class to load and prepare images and targets for object detection.
    """
    def __init__(self, frames, directory, annotations_by_image, empty_annotations,
                  horizontal=True, vertical=True, contrast=False, saturation=False, validation=False):
        self.frames = frames.reset_index(drop=True)
        self.directory = Path(directory)
        self.annotations_by_image = annotations_by_image
        self.empty_annotations = empty_annotations
        self.validation = validation

        self.augmentations = {
            "horizontal": horizontal,
            "vertical": vertical,
            "contrast": contrast,
            "saturation": saturation,
        }

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return prepare_sample(directory=self.directory,
                              frame=self.frames.iloc[index],
                              annotations_by_image=self.annotations_by_image,
                              empty_annotations=self.empty_annotations,
                              validation=self.validation, **self.augmentations)

def collate_data(batch):
    """
    Function groups samples into lists of images and targets.

    :param batch: collection of (image, target) pairs
    :return: a list of images and a list of targets
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


class RTDetrCollator:
    """
    Helper class to load and prepare images and targets for object detection
    for the RT-DETR model processor.
    """
    def __init__(self, processor, cat_to_label):
        self.processor = processor
        self.cat_to_label = cat_to_label

    def __call__(self, samples):
        images, targets = zip(*samples)
        annotations = []

        for target in targets:
            objects = []

            boxes = target["boxes"].tolist()
            categories = target["labels"].tolist()

            for (x_min, y_min, x_max, y_max), category in zip(boxes, categories):
                width, height = x_max - x_min, y_max - y_min

                annotation = {"bbox": [x_min, y_min, width, height],
                              "category_id": self.cat_to_label[int(category)],
                              "area": width * height,
                              "iscrowd": 0}
                objects.append(annotation)

            annotations.append({
                "image_id": int(target["image_id"].item()),
                "annotations": objects
            })

        batch = self.processor(images=list(images), annotations=annotations,
                               do_rescale=False, return_tensors="pt")

        return batch, list(targets)