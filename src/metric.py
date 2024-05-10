from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")  # True positives
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")  # False positives
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")  # False negatives

    def update(self, preds, target):
        # Convert predictions to binary by taking the argmax (assumes that the input is logits or probabilities for each class)
        preds = preds.argmax(dim=1)
        
        # Ensure that predictions and targets have the same shape
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape")

        # Calculate true positives, false positives, and false negatives
        tp = (preds & target).sum()  # True positives: prediction and target are both 1
        fp = (preds & ~target).sum()  # False positives: prediction is 1, target is 0
        fn = (~preds & target).sum()  # False negatives: prediction is 0, target is 1

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        precision = self.tp.float() / (self.tp + self.fp + 1e-6)  # add a small constant to avoid division by zero
        recall = self.tp.float() / (self.tp + self.fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # again, add a small constant to avoid division by zero
        return f1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = preds.argmax(dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape")

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum().item()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
