import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class FlexibleCNN(nn.Module):
    # Set-up pod CNN dający możliwość eksperymentowania z różnymi rozmiarami filtrów i typami poolingów.
    def __init__(self, kernel_size=3, pool_type='max'):
        super(FlexibleCNN, self).__init__()
        padding = kernel_size // 2
        
        # Warstwy konwolucyjne
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self._get_pool(pool_type),
            
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._get_pool(pool_type),
            
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Warstwy w pełni połączone
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def _get_pool(self, pool_type):
        if pool_type == 'max':
            return nn.MaxPool2d(2, 2)
        elif pool_type == 'avg':
            return nn.AvgPool2d(2, 2)
        else:
            return nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_dataloaders(batch_size=64):
    # Pobiera i przygotowuje dataloadery dla CIFAR-10 z odpowiednimi transformacjami
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, trainset, testset

def analyze_dataset(trainset):
    classes = trainset.classes
    data = trainset.data
    
    print(f"--- Statystyki CIFAR-10 ---")
    print(f"Liczba próbek treningowych: {len(trainset)}")
    print(f"Wymiary pojedynczego obrazu: {data[0].shape} (Wys x Szer x Kanały)")
    print(f"Zakres pikseli: min {data.min()}, max {data.max()}")
    print(f"Liczba klas: {len(classes)} {classes}")
    
    # Rozkład klas
    targets = np.array(trainset.targets)
    unique, counts = np.unique(targets, return_counts=True)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=classes, y=counts, palette='viridis', hue=classes, legend=False)
    plt.title('Rozkład klas w zbiorze treningowym')
    plt.ylabel('Liczba próbek')
    plt.show()

    # Przykładowe obrazy
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        idx = np.random.randint(len(trainset))
        img, label = trainset.data[idx], trainset.targets[idx]
        axes[i].imshow(img)
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    plt.suptitle("Przykładowe obrazy")
    plt.show()

# Funkcja do treningu modelu
def train_model(model, trainloader, testloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Walidacja
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        history['train_loss'].append(running_loss / len(trainloader))
        history['val_loss'].append(val_loss / len(testloader))
        history['val_acc'].append(100 * correct / total)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.3f} | Val Loss: {history['val_loss'][-1]:.3f} | Val Acc: {history['val_acc'][-1]:.2f}%")
        
    return history

def evaluate_best_model(model, testloader, device='cuda'):
    model.to(device)
    model.eval()
    
    all_preds, all_labels, misclassified = [], [], []
    classes = testloader.dataset.classes
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            wrong_idx = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in wrong_idx:
                if len(misclassified) < 5:
                    misclassified.append((inputs[idx].cpu(), labels[idx].cpu(), predicted[idx].cpu()))
                    
    print("\n--- RAPORT KLASYFIKACJI ---")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Macierz Pomyłek")
    plt.ylabel("Rzeczywiste")
    plt.xlabel("Przewidziane")
    plt.show()
    
    if misclassified:
        fig, axes = plt.subplots(1, len(misclassified), figsize=(15, 3))
        for i, (img, true_lbl, pred_lbl) in enumerate(misclassified):
            # Odkodowanie obrazu po normalizacji z powrotem na 0-1
            img = img / 2 + 0.5 
            axes[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
            axes[i].set_title(f"True: {classes[true_lbl]}\nPred: {classes[pred_lbl]}", color='red')
            axes[i].axis('off')
        plt.suptitle("Przykłady błędnych klasyfikacji")
        plt.show()

def plot_learning_curves(histories, labels, title, metric='val_acc'):
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        plt.plot(hist[metric], marker='o', label=label)
    
    plt.title(title)
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy [%]" if metric == 'val_acc' else "Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()