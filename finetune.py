import torch
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

dataset = load_dataset("yelp_review_full")
dataset["train"] = dataset["train"].select(range(1000))
dataset["test"] = dataset["test"].select(range(200))

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=2)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=2)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
optimizer = AdamW(model.parameters(), lr=5e-4)

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (1 - step / num_training_steps))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, tr_dataloader):
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    tr_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tr_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        tr_losses.append(total_loss / len(tr_dataloader))
    
    plt.plot(tr_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    return tr_losses

def evaluate_model(model, evl_dataloader):
    metric = Accuracy(task="multiclass", num_classes=5).to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in evl_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric(predictions, batch["labels"])
    
    accuracy = metric.compute()
    print(f"Accuracy: {accuracy.item():.4f}")
    return accuracy.item()

# Model kullanımı
train_losses = train_model(model, train_dataloader)
accuracy = evaluate_model(model, eval_dataloader)