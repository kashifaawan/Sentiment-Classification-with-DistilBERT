import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import argparse

class SentimentClassification:
    def __init__(self, train_file, output_dir, batch_size=32, epochs=3):
        self.train_file = train_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        if self.device == "cpu":
           print("Warning: You are running the code on a CPU. Consider using a GPU for faster training.")
        
    def load_data(self):
        train_data = pd.read_excel(self.train_file)
        return train_data

    def tokenize_text(self, text):
        if isinstance(text, str):
            tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, pad_to_max_length=True)
        elif isinstance(text, (list, tuple)) and all(isinstance(item, str) for item in text):
            tokens = self.tokenizer.encode(" ".join(text), add_special_tokens=True, truncation=True, max_length=512, pad_to_max_length=True)
        else:
            tokens = []  # Skip tokenization for invalid input
        return tokens

    def create_dataloader(self, data):
        data['tokenized'] = data['Reviews'].apply(lambda x: self.tokenize_text(x))
        # Remove rows with empty tokenized sequences (invalid input)
        data = data[data['tokenized'].apply(len) > 0]
        # Split the dataset into training and validation sets
        train_df, valid_df = train_test_split(data, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        train_input_ids = torch.tensor([ids for ids in train_df['tokenized']], dtype=torch.long)
        label_map = {'neg': 0, 'pos': 1}   
        train_labels = torch.tensor(train_df['Sentiment'].map(label_map).values, dtype=torch.long)
        valid_input_ids = torch.tensor([ids for ids in valid_df['tokenized']], dtype=torch.long)
        valid_labels = torch.tensor(valid_df['Sentiment'].map(label_map).values, dtype=torch.long)

        # Create DataLoader for batching
        train_dataset = TensorDataset(train_input_ids, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataset = TensorDataset(valid_input_ids, valid_labels)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

        return train_loader,valid_loader

    def train_model(self):
        best_val_accuracy = 0.0
        best_epoch = 0
        train_loss_history = []  
        valid_loss_history = []  
        train_accuracy_history = []  
        valid_accuracy_history = []
        model=self.model
        # Define the optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=2e-5)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0.0
            correct_train_predictions = 0
            
            for batch in self.train_loader:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, labels=labels)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                correct_train_predictions += (logits.argmax(dim=1) == labels).sum().item()
            
        # Calculate average training loss and accuracy for the epoch
        average_train_loss = total_train_loss / len(self.train_loader.dataset)
        train_accuracy = correct_train_predictions / len(self.train_loader.dataset)
        
    # Validation loop
        model.eval()
        val_predictions = []
        val_labels = []
        total_val_loss = 0.0
        correct_val_predictions = 0
        
        with torch.no_grad():
            for batch in self.valid_loader:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)

                outputs = model(input_ids, labels=labels)
                logits = outputs.logits

                total_val_loss += criterion(logits, labels).item()
                correct_val_predictions += (logits.argmax(dim=1) == labels).sum().item()
                
                val_predictions.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate average validation loss and accuracy for the epoch
        average_val_loss = total_val_loss / len(self.valid_loader.dataset)
        val_accuracy = correct_val_predictions / len(self.valid_loader.dataset)

        train_loss_history.append(average_train_loss)
        valid_loss_history.append(average_val_loss)
        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{self.epochs}")
        print(f"Training Loss: {average_train_loss:.4f} | Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {average_val_loss:.4f} | Validation Accuracy: {val_accuracy * 100:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            print(f"Validation accuracy improved to {best_val_accuracy * 100:.2f}% at epoch {best_epoch + 1}.")
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print("Model saved.")

        return train_loss_history, valid_loss_history, train_accuracy_history, valid_accuracy_history

    def plot_curves(self, train_loss, valid_loss, train_accuracy, valid_accuracy):
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(valid_loss, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.subplot(2, 1, 2)
        plt.plot(train_accuracy, label='Training Accuracy')
        plt.plot(valid_accuracy, label='Validation Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.tight_layout()
        plt.savefig('training_stats.png')  # Save the plot to a file
        plt.show()
    def run_training(self):
        data = self.load_data()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.train_loader,self.valid_loader = self.create_dataloader(data)
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.criterion = torch.nn.CrossEntropyLoss()

        train_loss, valid_loss, train_accuracy, valid_accuracy = self.train_model()

        self.plot_curves(train_loss, valid_loss, train_accuracy, valid_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DistilBERT sentiment analysis model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_file", type=str, default="/content/drive/MyDrive/train.xlsx", help="Path to the training data file")
    parser.add_argument("--output_dir", type=str, default="distillbert_sentiment_model", help="Output directory to save the trained model")

    args = parser.parse_args()

    sentiment_model = SentimentClassification(args.train_file, args.output_dir, args.batch_size, args.epochs)
    sentiment_model.run_training()
