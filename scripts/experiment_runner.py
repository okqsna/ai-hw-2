import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO

from .data_prep import load_yaml
from .yolo26_model import YOLOTrainer
from .torch_model import TorchTrainer
from .rdtdetr_model import RTDETRTrainer
from ultralytics import YOLO

def run_torch_cv(cv_fold_loaders, model_method, output_dir,
                 annotations_path, device, seed=42, epochs=5,
                lr=1e-4, weight_decay=1e-4, lr_scheduler=False, step_size=4):
    """
    Function trains a PyTorch based detector on cross-validation fold and saves
    training histories and validation results for the epochs with highest mAP.

    :param cv_fold_loaders: dictionary of training and validation loaders by fold
    :param model_method: function that creates the detection model
    :param output_dir: directory for saving fold histories, metrics, and checkpoints
    :param annotations_path: path to COCO annotations used for validation
    :param device: device used for training
    :param seed: random seed used at the start of each fold
    :param epochs: number of training epochs per fold
    :param lr: initial learning rate
    :param weight_decay: weight decay used by the optimizer
    :param lr_scheduler: whether to enable the learning rate scheduler
    :param step_size: number of epochs between scheduler updates
    :return: dictionary of fold histories and a list of best results per fold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_histories, fold_results = {}, []
    device = torch.device(device)

    for fold, loaders in cv_fold_loaders.items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        coco_gt = COCO(str(annotations_path))

        model = TorchTrainer(model_factory=model_method, train_loader=loaders["train"],
                                  valid_loader=loaders["valid"], device=device, coco_gt=coco_gt,
                                  output_dir=fold_dir, fold=fold, lr=lr, weight_decay=weight_decay,
                                  epochs=epochs, lr_scheduler=lr_scheduler, step_size=step_size)
        model.fit()
        history = pd.DataFrame(model.history)

        if not history.empty:
            best_epoch = history.loc[history["mAP"].idxmax()].to_dict()
            fold_results.append({"fold": fold, **best_epoch})
            fold_histories[fold] = history

            history.to_csv(fold_dir / "history.csv", index=False)
            pd.DataFrame(fold_results).to_csv(output_dir / "fold_results.csv", index=False)
        else:
            print(f"Fold {fold}: no validation completed. Last checkpoint saved.")
            continue

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return fold_histories, fold_results


def run_torch_final(train_loader, valid_loader, model_method, output_dir, annotations_path, device, seed=42, epochs=5,
                    lr=1e-4, weight_decay=1e-4, lr_scheduler=False, step_size=4):
    """
    Function trains a final PyTorch based detector and saves training history and
    validation metrics for the epoch with highest mAP.

    :param train_loader: loader providing training samples
    :param valid_loader: loader providing validation samples
    :param model_method: function that creates the detection model
    :param output_dir: directory for saving history, metrics, and checkpoints
    :param annotations_path: path to COCO annotations used for validation
    :param device: device used for training
    :param seed: random seed used before training
    :param epochs: number of training epochs
    :param lr: initial learning rate
    :param weight_decay: weight decay used by the optimizer
    :param lr_scheduler: whether to enable the learning rate scheduler
    :param step_size: number of epochs between scheduler updates
    :return: training history and results for the best epoch, or an empty dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    coco_gt = COCO(str(annotations_path))
    model = TorchTrainer(model_factory=model_method, train_loader=train_loader, valid_loader=valid_loader,
                         device=device, coco_gt=coco_gt, output_dir=output_dir, fold="final", lr=lr,
                         weight_decay=weight_decay, epochs=epochs, lr_scheduler=lr_scheduler, step_size=step_size)
    model.fit()

    history = pd.DataFrame(model.history)
    history.to_csv(output_dir / "history.csv", index=False)

    result = {}
    if not history.empty:
        result = history.loc[history["mAP"].idxmax()].to_dict()
        pd.DataFrame([result]).to_csv(output_dir / "final_metrics.csv", index=False)
    else:
        print("No validation completed; no best metrics available.")

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history, result


def run_yolo_cv(cv_fold_loaders, fold_yaml_paths, output_dir,
                annotations_path, device, seed=42, epochs=5, batch_size=8, imgsz=640, 
                box_weight=7.5, lr=1e-4, weight_decay=1e-4, optimizer="Adam",
                validate_each_epoch=False):
    """
    Function trains YOLO on each cross-validation fold and saves training
    histories and the available validation results with highest mAP per fold.

    :param cv_fold_loaders: dictionary of loaders by fold
    :param fold_yaml_paths: dictionary of YOLO dataset YAML paths by fold
    :param output_dir: directory for saving fold outputs
    :param annotations_path: path to COCO annotations used for validation
    :param device: device used for training
    :param seed: random seed used at the start of each fold
    :param epochs: number of training epochs per fold
    :param batch_size: number of training images per batch
    :param imgsz: input image size used by YOLO
    :param box_weight: weight assigned to the bounding box loss
    :param lr: initial learning rate
    :param weight_decay: weight decay used by the optimizer
    :param optimizer: optimizer name
    :param validate_each_epoch: whether to run custom validation after each epoch
    :return: dictionary of fold histories and a list of best results per fold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(annotations_path))
    device = torch.device(device)
    first_fold = next(iter(cv_fold_loaders))
    names = load_yaml(fold_yaml_paths[first_fold])["names"]

    if isinstance(names, list):
        names = dict(enumerate(names))

    category_by_name = {category["name"]: category["id"] for category in coco_gt.cats.values()}
    class_to_category = {int(index): category_by_name[name] for index, name in names.items()}
    
    yolo_device = str(device.index if device.index is not None else 0) if device.type == "cuda" else device.type

    fold_histories, fold_results = {}, []
    for fold, loaders in cv_fold_loaders.items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model = YOLOTrainer(data_yaml=fold_yaml_paths[fold], valid_loader=loaders["valid"],
            device=yolo_device, coco_gt=coco_gt, class_to_category=class_to_category, output_dir=output_dir,
            fold=fold, lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size,
            imgsz=imgsz, box_weight=box_weight, optimizer=optimizer, 
            validate_each_epoch=validate_each_epoch, seed=seed)
        
        model.fit()

        history = pd.DataFrame(model.history)
        history.to_csv(fold_dir / "history.csv", index=False)
        fold_dir = model.run_dir
        fold_histories[fold] = history

        if not history.empty:
            best_epoch = history.loc[history["mAP"].idxmax()].to_dict()
            fold_results.append({"fold": fold, **best_epoch})
            pd.DataFrame(fold_results).to_csv(output_dir / "fold_results.csv", index=False)
        else:
            print(f"Fold {fold}: no validation completed. Last checkpoint saved.")
            continue

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return fold_histories, fold_results

def run_rtdetr(model_method, processor, train_loader, valid_loader,
               output_dir, annotations_path, device, seed=42, epochs=5,
               lr=1e-4, weight_decay=1e-4, lr_scheduler=False,
               step_size=4, max_hours=3, resume_training_model=None):
    """
    Function trains RT-DETR model, saves its configuration and processor, and saves
    training history and validation metrics for the epoch with highest mAP.

    :param model_method: function that creates the RT-DETR model
    :param processor: image processor used by RT-DETR
    :param train_loader: loader providing training batches
    :param valid_loader: loader providing validation batches
    :param output_dir: directory for saving model files, history, and metrics
    :param annotations_path: path to COCO annotations used for validation
    :param device: device used for training
    :param seed: random seed used before training
    :param epochs: number of training epochs
    :param lr: initial learning rate
    :param weight_decay: weight decay used by the optimizer
    :param lr_scheduler: whether to enable the learning rate scheduler
    :param step_size: number of epochs between scheduler updates
    :param max_hours: training time limit in hours passed to the trainer
    :param resume_training_model: optional checkpoint path passed to load the trained model
    :return: training history and results for the best epoch, or an empty dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    coco_gt = COCO(str(annotations_path))

    model = RTDETRTrainer(model_factory=model_method, processor=processor,
        train_loader=train_loader, valid_loader=valid_loader,
        device=device, coco_gt=coco_gt,
        output_dir=output_dir,
        lr=lr, weight_decay=weight_decay,
        epochs=epochs, lr_scheduler=lr_scheduler,
        step_size=step_size, max_hours=max_hours
    )

    if resume_training_model:
        model.load_model(resume_training_model)

    model.model.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    model.fit()

    history = pd.DataFrame(model.history, columns=["epoch", "train_loss", "learning_rate", "mAP", "AP50", "AP_small"])
    history.to_csv(output_dir / "history.csv", index=False)

    result = {}

    if not history.empty:
        result = history.loc[history["mAP"].idxmax()].to_dict()
        pd.DataFrame([result]).to_csv(output_dir / "final_metrics.csv", index=False)
    else:
        print("No validation completed; no best metrics available.")

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history, result

def run_yolo_final(name, imgsz, box_weight, train_kwargs, output_root, seed=42):
    """
    Function trains a final YOLO model, validates its best checkpoint, and
    saves the validation metrics.

    :param name: experiment name used for the output folder
    :param imgsz: input image size used by YOLO
    :param box_weight: weight assigned to the bounding box loss
    :param train_kwargs: additional arguments passed to YOLOTrainer
    :param output_root: parent directory for experiment outputs
    :param seed: random seed set before creating the trainer
    :return: training history and validation metrics for the best model checkpoint
    """
    output_root = Path(output_root)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = YOLOTrainer(**train_kwargs,
                        output_dir=output_root / name,
                        imgsz=imgsz,
                        box_weight=box_weight)

    model.fit()
    history = pd.DataFrame(model.history)

    if model.best_checkpoint is None:
        raise RuntimeError(f"No best checkpoint saved for {name}")

    best_model = YOLO(str(model.best_checkpoint))
    metrics = model.validate(model=best_model)

    result = {"mAP": metrics["mAP"], "AP50": metrics["AP50"], "AP_small": metrics["AP_small"]}
    pd.DataFrame([result]).to_csv(model.run_dir / "final_metrics.csv", index=False)

    del model, best_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history, result
