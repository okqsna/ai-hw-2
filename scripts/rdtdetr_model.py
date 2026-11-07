import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import pandas as pd

class RDTDETRTrainer:
    def __init__(self, model_factory, processor, train_loader, valid_loader,
                 device, coco_gt, output_dir, lr=1e-4,
                 weight_decay=1e-4, epochs=3, lr_scheduler=False, step_size=4, max_hours=3):
        self.model = model_factory().to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device(device)
        self.coco_gt = coco_gt
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.category_ids = sorted(coco_gt.getCatIds())
        self.cat_to_label = {cat: i for i, cat in enumerate(self.category_ids)}
        self.label_to_cat = {i: cat for cat, i in self.cat_to_label.items()}

        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=lr, weight_decay=weight_decay)
        self.history = []
        self.best_map = -float("inf")
        self.epochs = epochs

        self.lr_scheduler = (torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size,
                                                              gamma=0.1) if lr_scheduler else None)
        self.max_hours = max_hours
        self.time_limit_reached = False

    def save_model(self, epoch, metrics=None, filename="last_model.pt", epoch_complete=True):
        torch.save({"epoch": epoch, "epoch_complete": epoch_complete,
            "time_limit_reached": self.time_limit_reached,
            "metrics": metrics, "history": self.history,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
        }, self.output_dir / filename)


    def train_epoch(self):
        self.model.train()
        total_loss, image_count = 0.0, 0

        for batch, targets in self.train_loader:
            if time.monotonic() - self.started >= self.max_hours * 3600:
                self.time_limit_reached = True
                break

            batch = batch.to(self.device)
            batch["labels"] = [
                {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in batch["labels"]
            ]

            self.optimizer.zero_grad()

            outputs = self.model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            image_count += len(targets)
            total_loss += loss.item() * len(targets)

        if image_count == 0:
            raise ValueError("Training loader is empty.")

        return total_loss / image_count if image_count else 0.0


    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        predictions, image_ids = [], []

        for batch, targets in self.valid_loader:
            if time.monotonic() - self.started >= self.max_hours * 3600:
                self.time_limit_reached = True
                return None

            batch = batch.to(self.device)

            inputs = {key: value for key, value in batch.items() if key != "labels"}
            outputs = self.model(**inputs)

            target_sizes = [(self.coco_gt.imgs[int(target["image_id"])]["height"],
                             self.coco_gt.imgs[int(target["image_id"])]["width"]) for target in targets]

            outputs = self.processor.post_process_object_detection(outputs,
                                                                   target_sizes=target_sizes,
                                                                   threshold=0.0,
            )

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                image_ids.append(image_id)

                boxes = output["boxes"].cpu().numpy().copy()
                boxes[:, 2:] -= boxes[:, :2]

                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    predictions.append({
                        "image_id": image_id,
                         "category_id": self.label_to_cat[int(label)],
                        "bbox": box.tolist(),
                        "score": float(score.item())
                    })

        if predictions:
            coco_predictions = self.coco_gt.loadRes(predictions)
        else:
            coco_predictions = COCO()
            coco_predictions.dataset = {"images": [self.coco_gt.imgs[image_id] for image_id in image_ids],
                                        "categories": self.coco_gt.dataset["categories"],
                                        "annotations": []}
            coco_predictions.createIndex()

        evaluator = COCOeval(self.coco_gt, coco_predictions, iouType="bbox")
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = list(range(1, 9))
        evaluator.params.maxDets = [1, 10, 100]

        if time.monotonic() - self.started >= self.max_hours * 3600:
            self.time_limit_reached = True
            return None

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        return {
            "mAP": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP_small": float(evaluator.stats[3]),
        }

    def fit(self):
        self.started = time.monotonic()
    
        for epoch in range(1, self.epochs + 1):
            epoch_complete = False
            metrics = None

            train_loss = self.train_epoch()
            if self.time_limit_reached:
                break

            metrics = self.validate()
            if metrics is None:
                break

            epoch_complete = True
            lr = self.optimizer.param_groups[0]["lr"]

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": lr,
                **metrics
            })

            pd.DataFrame(self.history).to_csv(self.output_dir / "history.csv", index=False)

            if metrics["mAP"] > self.best_map:
                self.best_map = metrics["mAP"]
                self.save_model(epoch, metrics, filename="best_model.pt")

            print(f"Epoch {epoch}/{self.epochs} | "
                            f"loss={train_loss:.4f} | "
                            f"learning_rate={lr:.6f} | "
                            f"mAP={metrics['mAP']:.4f} | "
                            f"AP50={metrics['AP50']:.4f} | "
                            f"AP_small={metrics['AP_small']:.4f}")
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.save_model(epoch, metrics, filename="last_model.pt", epoch_complete=epoch_complete)

        if self.time_limit_reached:
            print("Reached training time limit. Last model saved.")
