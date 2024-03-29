import torch
from typing import List, Tuple


def calculate_accuracy(output, target, topk_list: List[int]) -> Tuple[List[float], torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk_list)
        batch_size = target.size(0)

        topk_confidences, pred = output.topk(maxk, 1, True, True)
        
        topk_confidences = topk_confidences.t()
        
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # correct.reshape(-1).numpy().astype(int)

        res = []
        confidences = []
        for k in topk_list:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

            confidences_k = topk_confidences[:k].transpose(0, 1)
            confidences.append(confidences_k)

        # print(confidences[0].reshape(-1))

        return res, correct.reshape(-1), confidences[0].reshape(-1)


class ClassificationAccuracyTracker:
    def __init__(self) -> None:
        self.total_cnt = 0
        self.avg_acc1 = 0
        self.avg_acc5 = 0
        self.corrects = []
        self.confidences = []

    def reset(self):
        self.total_cnt = 0
        self.avg_acc1 = 0
        self.avg_acc5 = 0
        self.corrects = []
        self.confidences = []

    def update(self, acc1, acc5, batch_size):
        self.avg_acc1 = (self.avg_acc1 * self.total_cnt + acc1 * batch_size) / (self.total_cnt + batch_size)
        self.avg_acc5 = (self.avg_acc5 * self.total_cnt + acc5 * batch_size) / (self.total_cnt + batch_size)
        self.total_cnt += batch_size


class LossTracker:
    def __init__(self) -> None:
        self.total_cnt = 0
        self.avg_loss = 0

    def reset(self):
        self.total_cnt = 0
        self.avg_loss = 0

    def update(self, loss, batch_size):
        self.avg_loss = (self.avg_loss * self.total_cnt + loss * batch_size) / (self.total_cnt + batch_size)
        self.total_cnt += batch_size


if __name__ == "__main__":
    pass