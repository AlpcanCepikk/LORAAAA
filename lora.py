import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tempfile
import tarfile
import io
from urllib.request import urlopen

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe, Vectors, vocab
from torchtext.data.functional import to_map_style_dataset

class GloVe_override(Vectors):
    url = {"6B": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tQdezXocAJMBMPfUJx_iUg/glove-6B.zip"}
    
    def __init__(self, name="6B", dim=100, **kwargs):
        url = self.url[name]
        name = f"glove.{name}.{dim}d.txt"
        super().__init__(name, url=url, **kwargs)

class IMDBDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = os.path.join(root_dir, "train" if train else "test")
        self.neg_files = [os.path.join(self.root_dir, "neg", f) 
                         for f in os.listdir(os.path.join(self.root_dir, "neg")) 
                         if f.endswith('.txt')]
        self.pos_files = [os.path.join(self.root_dir, "pos", f) 
                         for f in os.listdir(os.path.join(self.root_dir, "pos")) 
                         if f.endswith('.txt')]
        self.files = self.neg_files + self.pos_files
        self.labels = [0] * len(self.neg_files) + [1] * len(self.pos_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r', encoding='utf-8') as file:
            content = file.read()
        return self.labels[idx], content

class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, linear_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        
    def forward(self, x):
        original_output = self.linear(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)
        return original_output + lora_output

class TextClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=100, hidden_dim=128, freeze=False):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            glove_embedding.vectors, freeze=freeze
        )
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def setup_data():
    tokenizer = get_tokenizer("basic_english")
    
    try:
        glove_embedding = GloVe_override(name="6B", dim=100)
    except:
        glove_embedding = GloVe(name="6B", dim=100)
    
    vocab_obj = vocab(glove_embedding.stoi, 0, specials=('<unk>', '<pad>'))
    vocab_obj.set_default_index(vocab_obj["<unk>"])
    
    def text_pipeline(x):
        return vocab_obj(tokenizer(x))
    
    return tokenizer, glove_embedding, vocab_obj, text_pipeline

def load_imdb_dataset():
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/35t-FeC-2uN1ozOwPs7wFg.gz'
    urlopened = urlopen(url)
    tar = tarfile.open(fileobj=io.BytesIO(urlopened.read()))
    tempdir = tempfile.TemporaryDirectory()
    tar.extractall(tempdir.name)
    tar.close()
    
    root_dir = os.path.join(tempdir.name, 'imdb_dataset')
    train_iter = IMDBDataset(root_dir=root_dir, train=True)
    test_iter = IMDBDataset(root_dir=root_dir, train=False)
    
    return train_iter, test_iter

def collate_batch(batch, text_pipeline, device):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(label)
        text_list.append(torch.tensor(text_pipeline(text), dtype=torch.int64))
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True)
    
    return label_list.to(device), text_list.to(device)

def evaluate(dataloader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for label, text in dataloader:
            outputs = model(text)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return 100 * correct / total

def train_model(model, optimizer, criterion, train_dataloader, valid_dataloader, 
                epochs=10, device='cpu'):
    loss_history = []
    acc_history = []
    best_acc = 0
    
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        
        for label, text in train_dataloader:
            optimizer.zero_grad()
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss += loss.item()
        
        loss_history.append(epoch_loss)
        acc_val = evaluate(valid_dataloader, model, device)
        acc_history.append(acc_val)
        
        if acc_val > best_acc:
            best_acc = acc_val
            print(f"Epoch {epoch+1}: New best accuracy: {acc_val:.2f}%")
    
    return loss_history, acc_history

def predict(text, model, text_pipeline, device):
    model.eval()
    with torch.no_grad():
        text_tensor = torch.tensor(text_pipeline(text)).unsqueeze(0).to(device)
        output = model(text_tensor)
        prediction = output.argmax(1).item()
        labels = {0: "negative review", 1: "positive review"}
        return labels[prediction]

def plot_training_history(loss_history, acc_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(loss_history, 'r-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    
    ax2.plot(acc_history, 'b-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show()

# Ana çalışma akışı
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Veri hazırlığı
    tokenizer, glove_embedding, vocab_obj, text_pipeline = setup_data()
    train_iter, test_iter = load_imdb_dataset()
    
    # Veri setlerini hazırla
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    
    num_train = int(len(train_dataset) * 0.95)
    split_train, split_valid = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train]
    )
    
    # DataLoader'ları oluştur
    batch_size = 64
    train_dataloader = DataLoader(
        split_train, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, text_pipeline, device)
    )
    valid_dataloader = DataLoader(
        split_valid, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, text_pipeline, device)
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, text_pipeline, device)
    )
    
    # Model oluştur ve eğit
    model = TextClassifier(num_classes=2, freeze=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    print("Training base model...")
    loss_hist, acc_hist = train_model(
        model, optimizer, criterion, train_dataloader, valid_dataloader, 
        epochs=5, device=device
    )
    
    # LoRA adaptasyonu
    print("\nApplying LoRA adaptation...")
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc1 = LinearWithLoRA(model.fc1, rank=2, alpha=0.1)
    model.fc2 = nn.Linear(128, 2).to(device)
    
    optimizer_lora = torch.optim.SGD(model.parameters(), lr=1.0)
    
    print("Training with LoRA...")
    lora_loss_hist, lora_acc_hist = train_model(
        model, optimizer_lora, criterion, train_dataloader, valid_dataloader,
        epochs=5, device=device
    )
    
    # Test sonuçları
    test_acc = evaluate(test_dataloader, model, device)
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    
    # Örnek tahmin
    sample_text = "This was a great movie with excellent acting!"
    prediction = predict(sample_text, model, text_pipeline, device)
    print(f"\nSample prediction:")
    print(f"Text: {sample_text}")
    print(f"Prediction: {prediction}")
    
    # Eğitim grafiği
    plot_training_history(lora_loss_hist, lora_acc_hist)