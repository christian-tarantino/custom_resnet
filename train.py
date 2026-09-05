import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms.v2 as v2
from torchinfo import summary
from CNN.model.custom_resnet import ResNet



def load_to_gpu(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    data, labels = next(iter(loader))
    return data.cuda(), labels.cuda()

class GPUDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]

# Carica tutto in VRAM
full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
all_data, all_labels = load_to_gpu(full_dataset)

# Split 40k/10k
train_data, val_data = all_data[:40000], all_data[40000:]
train_labels, val_labels = all_labels[:40000], all_labels[40000:]


train_dataset = GPUDataset(train_data, train_labels)
val_dataset   = GPUDataset(val_data, val_labels)

#carica il test set in VRAM
test_full = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_data, test_labels = load_to_gpu(test_full)
test_dataset = GPUDataset(test_data, test_labels)

# Loader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=0)





def train(model, train_loader, criterion, optimizer, gpu_transform, device, batch_transform, p_aug, epoch, debug_first_batch=False,):
    model.train()
    running_loss = 0.0
    batch_count = 0
    correct = 0
    total = 0
    epoch_start = time.time()

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        if debug_first_batch and batch_count == 0:
            start_time = time.time()
            images = gpu_transform(images)
            load_time = time.time() - start_time
            print(f" Tempo caricamento primo batch (VRAM transform): {load_time:.4f}s")
        else:
            images = gpu_transform(images)
        if epoch <= 135:

            if torch.rand(1).item() <= p_aug:
        
                images, labels = batch_transform(images, labels)
        elif epoch > 135 and epoch <= 145:
            if torch.rand(1).item() <= (p_aug-0.50):
        
                images, labels = batch_transform(images, labels)



        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(images)
            if labels.dim() == 2:
                log_probs = F.log_softmax(outputs, dim=1)
                loss = -(labels.float() * log_probs).sum(dim=1).mean()
            else:
                loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        if labels.dim() == 2:
            hard_labels = labels.argmax(dim=1)
        else:
            hard_labels = labels
        total += hard_labels.size(0)
        correct += predicted.eq(hard_labels).sum().item()
        batch_count += 1

    epoch_time = time.time() - epoch_start
    samples_per_sec = len(train_loader.dataset) / epoch_time
    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total if total else 0.0
    return epoch_loss, epoch_time, samples_per_sec, accuracy


def validate(model, val_loader, criterion, device, gpu_transform):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = gpu_transform(images)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total if total else 0.0
    return epoch_loss, accuracy


def test(model, test_loader, criterion, device, gpu_transform):
    model.eval()
    running_loss = 0.0      
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = gpu_transform(images)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total if total else 0.0
    return epoch_loss, accuracy


def main():
    total_start_time = time.perf_counter()

    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    

    train_gpu_transforms = v2.Compose([
    v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    val_gpu_transforms = v2.Compose([
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    mixup = v2.MixUp(alpha=1.0, num_classes=10)
    cutmix = v2.CutMix(alpha=1.0, num_classes=10)

    #50% di probabilità Mixup, 50% Cutmix
    cutmix_or_mixup = v2.RandomChoice([mixup, cutmix])

    for epoch in range(num_epochs):

        debug = (epoch == 0)

        train_loss, train_time, samples_per_sec, train_accuracy = train(
            model, train_loader, criterion, optimizer, train_gpu_transforms, device,
            debug_first_batch=debug, batch_transform=cutmix_or_mixup, p_aug=1, epoch=epoch
        )

        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device, val_gpu_transforms
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f} - "
            f"Time: {train_time:.2f}s ({samples_per_sec:.0f} samples/s)"
        )



    
    # Valutazione finale sul test set
    torch.save(model.state_dict(), "resnet_cifar10.pth")
    test_loss, test_accuracy = test(model, test_loader, criterion, device, val_gpu_transforms)
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    # Plot delle curve
    plt.figure(figsize=(12, 4))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o', color='blue')
    plt.plot(val_accuracies, label='Val Accuracy', marker='s', color='green')
    plt.axhline(y=test_accuracy, color='red', linestyle='--', label=f'Test Accuracy: {test_accuracy:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    total_training_time = time.perf_counter() - total_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Tempo totale di training: {int(hours)}h "
        f"{int(minutes)}m {seconds:.2f}s"
    )
    
    return train_losses, val_losses, val_accuracies

if __name__ == "__main__":
    model = ResNet()
    num_epochs = 150
    warmup_epochs = 10
    # loss function e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,                       
        steps_per_epoch=len(train_loader), 
        epochs=num_epochs,
        pct_start=warmup_epochs / num_epochs, 
        div_factor=25.0,                    # LR iniziale = max_lr / 10
        final_div_factor=1e3,              
        anneal_strategy='cos'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    model = model.to(device)
    summary(model, input_size=(1, 3, 32, 32), device=device.type)
    
    main()
