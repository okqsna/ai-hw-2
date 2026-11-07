import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class TorchTrainer:
    def __init__(self, model_factory, train_loader, 
                 valid_loader, device, coco_gt, output_dir, fold,
                 lr=1e-4, weight_decay=1e-4, epochs=3, lr_scheduler=False, step_size=4):
        self.model = model_factory().to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.coco_gt = coco_gt
        self.output_dir = output_dir
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], 
                                          lr=lr, weight_decay=weight_decay)
        self.history = []
        self.best_map = -float("inf")
        self.epochs = epochs
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                        step_size=step_size, gamma=0.1) if lr_scheduler else None
        self.fold = fold

    def train_epoch(self):
        self.model.train()

        total_loss, image_count = 0.0, 0

        for images, targets in self.train_loader:
            images = [image.to(self.device) for image in images]

            targets = [{"boxes": target["boxes"].to(self.device), "labels": target["labels"].to(self.device)} 
                       for target in targets]
            self.optimizer.zero_grad()

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()

            image_count += len(images)
            total_loss += loss.item() * len(images)

        return total_loss / image_count


    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        predictions, image_ids = [], []

        for images, targets in self.valid_loader:
            images = [image.to(self.device) for image in images]
            outputs = self.model(images)

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
                        "category_id": int(label.item()),
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

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        return {
            "mAP": float(evaluator.stats[0]),
            "AP50": float(evaluator.stats[1]),
            "AP_small": float(evaluator.stats[3]),
        }

    def save_best_model(self, epoch, metrics):
        torch.save({"fold": self.fold, "epoch": epoch,
                    "metrics": metrics, "model_state_dict": self.model.state_dict(), 
                    "optimizer_state_dict": self.optimizer.state_dict()},
                    self.output_dir / "best_model.pt")

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            metrics = self.validate()
            lr = self.optimizer.param_groups[0]["lr"]
            self.history.append({"epoch": epoch, "train_loss": train_loss, "learning_rate": lr, **metrics})

            if metrics["mAP"] > self.best_map:
                self.best_map = metrics["mAP"]
                self.save_best_model(epoch, metrics)

            print(f"Epoch {epoch}/{self.epochs} | "
                            f"loss={train_loss:.4f} | "
                            f"learning_rate={lr:.6f} | "
                            f"mAP={metrics['mAP']:.4f} | "
                            f"AP50={metrics['AP50']:.4f} | "
                            f"AP_small={metrics['AP_small']:.4f}")
            

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
