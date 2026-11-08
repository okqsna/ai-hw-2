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
from .rdtdetr_model import RDTDETRTrainer
from ultralytics import YOLO

def run_torch_cv(cv_fold_loaders, model_method, output_dir, annotations_path, device, seed=42, epochs=5,
                lr=1e-4, weight_decay=1e-4, lr_scheduler=False, step_size=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_histories, fold_results = {}, []

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
        if history.empty:
            print(f"Fold {fold}: no validation completed. Last checkpoint saved.")
            continue

        best_epoch = history.loc[history["mAP"].idxmax()].to_dict()
        fold_results.append({"fold": fold, **best_epoch})

        pd.DataFrame(fold_results).to_csv(
            output_dir / "fold_results.csv", index=False
        )
        fold_histories[fold] = history

        history.to_csv(fold_dir / "history.csv", index=False)
        pd.DataFrame(fold_results).to_csv(output_dir / "fold_results.csv", index=False)

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return fold_histories, fold_results


def run_torch_final(train_loader, valid_loader, model_method, output_dir, annotations_path, device, seed=42, epochs=5,
                    lr=1e-4, weight_decay=1e-4, lr_scheduler=False, step_size=4):
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

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history, result


def run_yolo_cv(cv_fold_loaders, fold_yaml_paths, output_dir,
                annotations_path, device, seed=42, epochs=5, batch_size=8, imgsz=640, 
                box_weight=7.5, lr=1e-4, weight_decay=1e-4, optimizer="Adam", validate_each_epoch=False):

    output_dir = Path(output_dir)
    device = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(annotations_path))

    first_fold = next(iter(cv_fold_loaders))
    names = load_yaml(fold_yaml_paths[first_fold])["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names

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

        fold_dir = model.run_dir
        history.to_csv(fold_dir / "history.csv", index=False)
        fold_histories[fold] = history

        best_epoch = history.loc[history["mAP"].idxmax()].to_dict()
        fold_results.append({"fold": fold, **best_epoch})
        pd.DataFrame(fold_results).to_csv(output_dir / "fold_results.csv", index=False)

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return fold_histories, fold_results


def run_rtdetr(model_method, processor, train_loader, valid_loader,
               output_dir, annotations_path, device, seed=42, epochs=5,
               lr=1e-4, weight_decay=1e-4, lr_scheduler=False,
               step_size=4, max_hours=3, checkpoint_every_seconds=600):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    budget = TrainingBudget(max_hours=max_hours)
    coco_gt = COCO(str(annotations_path))

    trainer = RDTDETRTrainer(
        model_factory=model_method,
        processor=processor,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        coco_gt=coco_gt,
        output_dir=output_path,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_scheduler=lr_scheduler,
        step_size=step_size,
        budget=budget,
        checkpoint_every_seconds=checkpoint_every_seconds,
    )

    trainer.model.config.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    trainer.fit()

    history = pd.DataFrame(trainer.history)
    scored_history = (
        history.dropna(subset=["mAP"])
        if not history.empty
        else history
    )

    results = {
        "status": trainer.status,
        "elapsed_seconds": budget.elapsed_seconds,
        "best_epoch": None,
        "best_mAP": None,
        "last_checkpoint": str(output_path / "last_model.pt"),
        "best_checkpoint": None,
    }

    if not scored_history.empty:
        best_epoch = scored_history.loc[
            scored_history["mAP"].idxmax()
        ]
        results["best_epoch"] = int(best_epoch["epoch"])
        results["best_mAP"] = float(best_epoch["mAP"])

    if (output_path / "best_model.pt").exists():
        results["best_checkpoint"] = str(output_path / "best_model.pt")

    del trainer
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history, results

def run_yolo_final(name, imgsz, box_weight, train_kwargs, output_root, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = YOLOTrainer(**train_kwargs, output_dir=output_root / name, 
                          imgsz=imgsz, box_weight=box_weight)

    model.fit()
    history = pd.DataFrame(model.history)

    if model.best_checkpoint is None:
        raise RuntimeError(f"No best checkpoint saved for {name}")

    best_model = YOLO(str(model.best_checkpoint))
    metrics = model.validate(model=best_model)

    result = {
        "mAP": metrics["mAP"],
        "AP50": metrics["AP50"],
        "AP_small": metrics["AP_small"],
    }

    pd.DataFrame([result]).to_csv(model.run_dir / "final_metrics.csv", index=False)

    del model, best_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history, result