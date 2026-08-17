import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_NODE_NAME", "localhost")
    os.environ["MASTER_PORT"] = "29500"

    if platform.system() == "Windows":
        os.environ["USE_LIBUV"] = 0
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

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
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, pin_memory=True, drop_last=True, sampler=DistributedSampler(train_ds))
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    return train_loader, test_loader

# Neural Net
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


def main(rank, world_size, num_epoch):
    ddp_setup(rank, world_size)

    train_loader, test_loader = prepare_data()
    model = DummyNeuralNet(2, 2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    model = DDP(model, device_ids = [rank]) 

    for ep in range(num_epoch):
        train_loader.sampler.set_epoch(ep)
        model.train()

        for idx, (features, labels) in enumerate(train_loader):

            features, labels = features.to(rank), labels.to(rank)
            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {ep+1:03d}/{num_epoch:03d} | batch {idx+1:03d}/{len(train_loader):03d} | train loss: {loss:0.2f}")

    model.eval()

    try:
        train_accuracy = compute_accuracy(model, train_loader, rank)
        print(f"GPU{rank} Training accuracy: {train_accuracy:0.2f}")
        test_accuracy = compute_accuracy(model, test_loader, rank)
        print(f"GPU{rank} Test Accuracy: {test_accuracy:0.2f}")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e} \n This script is designed for running in the GPU"
            f"You have {torch.cuda.device_count()} GPUs in your environment"
        )

    destroy_process_group()

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

if __name__ == "__main__":
    print("Pytorch version", torch.__version__)
    print("CUDA available", torch.cuda.is_available())
    print("Number of GPU available: ", torch.cuda.device_count())

    torch.manual_seed(42)
    num_epoch = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epoch), nprocs=world_size)
