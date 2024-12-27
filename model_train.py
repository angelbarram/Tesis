import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch.nn.functional as F
import random

# Establecer la semilla
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Definición del dataset personalizado
class PulsarDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.data = datasets.ImageFolder(root=image_folder, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Preprocesamiento de las imágenes
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

# Crear datasets de entrenamiento, validación y test
train_dataset = PulsarDataset(image_folder='../datos/data_pool_10/train', transform=transform)
val_dataset = PulsarDataset(image_folder='../datos/data_pool_10/validation', transform=transform)
test_dataset = PulsarDataset(image_folder='../datos/data_pool_10/test', transform=transform)

# Definición de la CNN
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):  # Mejor dropout encontrado
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mejores parámetros
best_batch_size = 16  
best_lr = 1e-5 
best_weight_decay = 1e-5 
best_dropout = 0.2 

# Inicializar y entrenar el modelo
model = SimpleCNN(dropout_rate=best_dropout).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)

# Definir el scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7)

# Función de entrenamiento y validación con early stopping y scheduler
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=5):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) 
            preds = (preds > 0.5).int() 
            correct_train += torch.sum(preds == labels.int())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train.double() / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

        # Validación
        model.eval()
        val_loss = 0.0
        correct_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1)  
                outputs = model(inputs)
                loss = criterion(outputs, labels.float()) 
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).int()
                correct_val += torch.sum(preds == labels.int())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val.double() / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())
        logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # Scheduler: actualizar la tasa de aprendizaje según la pérdida de validación
        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs.')
                logging.info(f'Early stopping after {epoch+1} epochs.')
                break

    # Cargar el mejor modelo encontrado durante la validación
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        torch.save(model.state_dict(), 'model_for_fine_tuning.pth')

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

# Preparar los loaders para los conjuntos de entrenamiento, validación y test
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

# Entrenamiento y validación del modelo
train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100
)

# Guardar gráficos de entrenamiento y validación
os.makedirs('training_validation_images', exist_ok=True)

plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_validation_images/loss_curves.png')
plt.close()

plt.figure()
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_validation_images/accuracy_curves.png')
plt.close()

# Evaluación en el conjunto de test
model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        
        epsilon = 1e-7
        probs = torch.clamp(probs, epsilon, 1 - epsilon)
        
        preds = (probs > 0.5).int()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Calcular las métricas para el conjunto de test
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)
log_loss_value = log_loss(all_labels, all_probs)

print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print(f'Test ROC AUC: {roc_auc:.4f}')
print(f'Test Log Loss: {log_loss_value:.4f}')

# Guardar las métricas en un archivo de log
logging.basicConfig(filename='test_metrics.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.info(f'Test Precision: {precision:.4f}')
logging.info(f'Test Recall: {recall:.4f}')
logging.info(f'Test F1 Score: {f1:.4f}')
logging.info(f'Test ROC AUC: {roc_auc:.4f}')
logging.info(f'Test Log Loss: {log_loss_value:.4f}')

# Matriz de confusión normalizada
cm = confusion_matrix(all_labels, all_preds, normalize='true')

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=['No Glitch', 'Glitch'], yticklabels=['No Glitch', 'Glitch'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized Confusion Matrix')
plt.savefig('training_validation_images/normalized_confusion_matrix.png')
plt.close()
