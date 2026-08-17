import os
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    if platform.system() == "Windows":
        os.environ["USE_LIBUV"] = "0"
        init_process_group("gloo")
    else:
        init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# Dataset class
class ToyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]
    
def prepare_data():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    factor = 4
    X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    y_train = y_train.repeat(factor)
    X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    # dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        sampler=DistributedSampler(train_ds),
        drop_last=True,
        pin_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    return train_loader, test_loader

class DummyNeuralNet(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 30),
            nn.ReLU(),

            nn.Linear(30, 15),
            nn.ReLU(),

            nn.Linear(15, 2)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct_sample = 0.0
    total_sample = 0
    print(len(dataloader))

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = predictions == labels
        correct_sample += torch.sum(compare)
        total_sample += len(compare)

    print(f"Accuracy: {correct_sample / total_sample:0.4f}")
    return (correct_sample / total_sample).item()

def main(num_epoch):
    rank = ddp_setup()

    train_loader, test_loader = prepare_data()
    model = DummyNeuralNet(2, 2)
    model.to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    model = DDP(model, device_ids=[rank])   # core model is now accessible as model.module

    for ep in range(num_epoch):
        train_loader.sampler.set_epoch(ep)
        model.train()

        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)

            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"GPU{rank} Epoch: {ep+1:03d}|{num_epoch:03d}"
                  f"| batch size: {labels.shape[0]:03d}"
                  f"| Train/val loss: {loss:0.2f}")

    model.eval()

    try:
        train_accuracy = compute_accuracy(model, train_loader, device=rank)
        print(f"GPU{rank} Training accuracy: {train_accuracy}")
        test_accuracy = compute_accuracy(model, test_loader, device=rank)
        print(f"GPU{rank} Test accuracy: {test_accuracy}")
    except ZeroDivisionError as e:
        raise ZeroDivisionError("This script is designed to run on multi GPUs")

    destroy_process_group()


if __name__ == "__main__":

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print("Pytorch version: ", torch.__version__)
        print("Number of GPUs available: ", torch.cuda.device_count())
    
    torch.manual_seed(123)
    num_epoch = 3
    main(num_epoch)
