import torch
from typing import List, Tuple


def calculate_accuracy(output, target, topk_list: List[int]) -> Tuple[List[float], torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk_list)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # correct.reshape(-1).numpy().astype(int)

        res = []
        for k in topk_list:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res, correct.reshape(-1)


class ClassificationAccuracyTracker:
    def __init__(self) -> None:
        self.total_cnt = 0
        self.avg_acc1 = 0
        self.avg_acc5 = 0
        self.corrects = []

    def reset(self):
        self.total_cnt = 0
        self.avg_acc1 = 0
        self.avg_acc5 = 0
        self.corrects = []

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


# TODO: implement them later referring above new logics
# class mApTracker:
#     def __init__(self) -> None:
#         self.total_cnt = 0
#         self.avg_map50_95 = 0
#         self.avg_map50 = 0

#     def reset(self):
#         self.total_cnt = 0
#         self.avg_map50_95 = 0
#         self.avg_map50 = 0

#     def update(self, map50_95, map50):
#         self.avg_map50_95 = (self.avg_map50_95 * self.total_cnt + map50_95) / (self.total_cnt + 1)
#         self.avg_map50 = (self.avg_map50 * self.total_cnt + map50) / (self.total_cnt + 1)
#         self.total_cnt += 1


if __name__ == "__main__":
    # validate accuracy calculation
    batch_size = 4
    outputs = torch.zeros(size=(batch_size, 10))
    targets = torch.zeros(size=(batch_size,), dtype=torch.long)
    for i in range(batch_size):
        outputs[i][0] = 1.
        outputs[i][1] = 0.9
        outputs[i][2] = 0.7
        outputs[i][3] = 0.5
        outputs[i][4] = 0.3
        targets[i] = 4
    print(targets)
    res = calculate_accuracy(outputs, targets, topk_list=[1, 2, 3, 4, 5])
    print(res)
    exit()
    # validate calculate_accuracy
    acc1, acc5 = [10, 20, 30, 40], [20, 40, 60, 80]

    acc_tracker = ClassificationAccuracyTracker()

    for i in range(len(acc1)):
        acc_tracker.update(acc1[i], acc5[i], batch_size=1)
    print(f"{i}th update -> acc1: {acc_tracker.avg_acc1}, acc5: {acc_tracker.avg_acc5}")


    acc1, acc5, batches = [(10 + 20 + 30) / 3., 40], [(20 + 40 + 60) / 3., 80], [3, 1]
    acc_tracker.reset()
    for i in range(len(acc1)):
        acc_tracker.update(acc1[i], acc5[i], batch_size=batches[i])
    print(f"{i}th update -> acc1: {acc_tracker.avg_acc1}, acc5: {acc_tracker.avg_acc5}")