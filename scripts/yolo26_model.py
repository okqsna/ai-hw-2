from pathlib import Path
import torch
import json
import shutil
import pandas as pd
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms.functional import to_pil_image

def build_yolo_model():
    return YOLO("yolo26n.pt")

class YOLOTrainer:
    def __init__(self, data_yaml, output_dir, fold, valid_loader, 
                 model_factory=build_yolo_model, device=None, lr=1e-4, weight_decay=1e-4, coco_gt=None,
                 epochs=5, batch_size=8, imgsz=640, box_weight=7.5, 
                 optimizer="Adam", validate_each_epoch=True, class_to_category=None, seed=42):
        self.model = model_factory()

        self.data_yaml = str(data_yaml)
        self.output_dir = str(output_dir)
        self.fold = fold
        self.device = device

        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.coco_gt = coco_gt
        self.class_to_category = class_to_category
        self.valid_loader = valid_loader

        self.history = []
        self.best_map = -float("inf")
        self.epochs = epochs

        self.batch_size = batch_size
        self.imgsz = imgsz
        self.box_weight = box_weight
        self.seed = seed

        self.best_checkpoint = None
        self.training_results = None
        self.validation_results = None
        self.validate_each_epoch = validate_each_epoch
        self.model.add_callback("on_model_save", self._epoch_results)

    def save_model(self, epoch, metrics):
        trainer = self.model.trainer
        save_dir = Path(trainer.save_dir)
        checkpoint = save_dir / "best_model.pt"

        shutil.copy2(trainer.last, checkpoint)
        metadata = {"fold": self.fold, "epoch": epoch, "metrics": metrics}
        (save_dir / "best_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return checkpoint

    def fit(self):
        self.training_results = self.model.train(data=self.data_yaml,
            project=str(Path(self.output_dir).resolve()), name=f"fold_{self.fold}",
            epochs=self.epochs, batch=self.batch_size, imgsz=self.imgsz,
            box=self.box_weight, optimizer=self.optimizer,lr0=self.lr,
            weight_decay=self.weight_decay, device=self.device,
            seed=self.seed, val=False, patience=0,
            save=True, plots=True
        )

        self.run_dir = Path(self.model.trainer.save_dir)
        return self.history


    def validate(self, model=None):
        model = self.model if model is None else model
        predictions, image_ids = [], []

        for images, targets in self.valid_loader:
            images = [to_pil_image(image.cpu()) for image in images]
            outputs = model.predict(source=images, imgsz=self.imgsz, conf=0.001, iou=0.7, max_det=100,
                                    device=self.device, verbose=False)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                image_ids.append(image_id)

                boxes = output.boxes.xyxy.cpu().numpy().copy()
                boxes[:, 2:] -= boxes[:, :2]

                scores = output.boxes.conf.cpu().numpy()
                labels = output.boxes.cls.cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    predictions.append({
                        "image_id": image_id,
                        "category_id": self.class_to_category[int(label.item())],
                        "bbox": box.tolist(),
                        "score": float(score.item())
                    })

        image_ids = sorted(set(image_ids))

        if predictions:
            coco_predictions = self.coco_gt.loadRes(predictions)
        else:
            coco_predictions = COCO()
            coco_predictions.dataset = {"images": [self.coco_gt.imgs[i] for i in image_ids],
                                        "categories": self.coco_gt.dataset["categories"],
                                        "annotations": [],
            }
            coco_predictions.createIndex()

        evaluator = COCOeval(self.coco_gt, coco_predictions, iouType="bbox")
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = sorted(set(self.class_to_category.values()))
        evaluator.params.maxDets = [1, 10, 100]

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        return {
            "mAP": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP_small": float(evaluator.stats[3])
        }

    def _epoch_results(self, trainer):
        epoch = trainer.epoch + 1
        metrics = {}

        if self.validate_each_epoch or epoch == self.epochs:
            epoch_model = YOLO(str(trainer.last))
            metrics = self.validate(model=epoch_model)
            self.validation_results = metrics

        train_loss = sum(float(value.detach().sum().cpu()) if torch.is_tensor(value) else float(value) for value in trainer.tloss.values())
        lr = float(trainer.optimizer.param_groups[0]["lr"])

        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": lr, **metrics,
        })

        pd.DataFrame(self.history).to_csv(Path(trainer.save_dir) / "history.csv", index=False)

        if metrics and metrics["mAP"] >= 0 and metrics["mAP"] > self.best_map:
            self.best_checkpoint = self.save_model(epoch, metrics)
            self.best_map = metrics["mAP"]

        message = [f"Fold {self.fold}",
            f"Epoch {epoch}/{self.epochs}",
            f"loss={train_loss:.4f}",
            f"learning_rate={lr:.6f}",
        ]

        message.extend(f"{name}={value:.4f}" for name, value in metrics.items())

        print(" | ".join(message))
        print(message)
