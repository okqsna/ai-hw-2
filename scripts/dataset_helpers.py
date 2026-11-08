import cv2
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from .data_prep import prepare_annotations, frame_augmentation, out_ignored


def prepare_sample(directory, annotations_by_image, frame, empty_annotations, validation=False):
    image_path = directory / frame["split"] / frame["file_name"]
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    annotations = annotations_by_image.get(frame["image_id"], empty_annotations).copy()

    if not validation:
        image, annotations = frame_augmentation(image, annotations,
                                                horizontal=torch.rand(()).item() < 0.5,
                                                vertical=torch.rand(()).item() < 0.5)
    objects, ignored = prepare_annotations(annotations)

    if not validation:
        image = out_ignored(image, ignored)

    coordinates = ["x_min", "y_min", "x_max", "y_max"]

    target = {"boxes": torch.tensor(objects[coordinates].to_numpy().tolist(), dtype=torch.float32).reshape(-1, 4),
              "labels": torch.tensor(objects["category_id"].tolist(), dtype=torch.int64),
              "image_id": torch.tensor(int(frame["image_id"]), dtype=torch.int64)}
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return to_tensor(image), target


class DetectionDataset(Dataset):
    def __init__(self, frames, directory, annotations_by_image, empty_annotations, validation=False):
        self.frames = frames.reset_index(drop=True)
        self.directory = Path(directory)
        self.annotations_by_image = annotations_by_image
        self.empty_annotations = empty_annotations
        self.validation = validation

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return prepare_sample(directory=self.directory,
                              annotations_by_image=self.annotations_by_image,
                              frame=self.frames.iloc[index], 
                              empty_annotations=self.empty_annotations,
                              validation=self.validation)

def collate_data(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class RTDetrCollator:
    def __init__(self, processor, cat_to_label):
        self.processor = processor
        self.cat_to_label = cat_to_label

    def __call__(self, samples):
        images, targets = zip(*samples)
        annotations = []

        for target in targets:
            objects = []

            for box, category in zip(target["boxes"].tolist(), target["labels"].tolist()):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min

                objects.append({"bbox": [x_min, y_min, width, height], 
                                "category_id": self.cat_to_label[int(category)],
                                "area": width * height,
                                "iscrowd": 0
                })

            annotations.append({
                "image_id": int(target["image_id"].item()),
                "annotations": objects,
            })

        batch = self.processor(images=list(images), annotations=annotations,
                               do_rescale=False, return_tensors="pt")

        return batch, list(targets)