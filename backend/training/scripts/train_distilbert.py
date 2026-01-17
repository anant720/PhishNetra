#!/usr/bin/env python3
"""
Training script for PhishNetra DistilBERT scam detection model
Fine-tunes DistilBERT for contextual scam classification
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScamDataset(Dataset):
    """Dataset for scam detection training"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DistilBERTTrainer:
    """Trainer for DistilBERT scam detection model"""

    def __init__(self, model_path: str = "./models/distilbert_scam", data_dir: str = "./data/processed"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

        # Model parameters
        self.max_length = 512
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.warmup_steps = 0

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._setup_model()

    def _setup_model(self):
        """Initialize model and tokenizer"""
        logger.info("Setting up DistilBERT model...")

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Check if model already exists
        if self.model_path.exists() and (self.model_path / 'pytorch_model.bin').exists():
            logger.info(f"Loading existing model from {self.model_path}")
            self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_path))
        else:
            logger.info("Creating new DistilBERT model")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2  # Binary classification: legitimate vs scam
            )

        self.model.to(self.device)

    def load_data(self):
        """Load training and validation data"""
        logger.info("Loading training data...")

        # Load datasets
        train_df = pd.read_csv(self.data_dir / "train.csv")
        val_df = pd.read_csv(self.data_dir / "validation.csv")

        # Create datasets
        train_dataset = ScamDataset(
            train_df['cleaned_text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )

        val_dataset = ScamDataset(
            val_df['cleaned_text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            self.max_length
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        """Train the model"""
        logger.info("Starting training...")

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        training_stats = []

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")
            for batch in train_progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                train_progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{train_correct/train_total:.4f}"
                })

            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total

            # Validation phase
            val_loss, val_accuracy, val_f1, val_auc = self.evaluate(val_loader)

            logger.info(f"Epoch {epoch + 1} Results:")
            logger.info(".4f")
            logger.info(".4f")
            logger.info(".4f")
            logger.info(".4f")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                logger.info("Saved best model checkpoint")

            # Record stats
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_auc': val_auc
            })

        # Save training statistics
        with open(self.model_path / 'training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=2)

        logger.info("Training completed!")
        return training_stats

    def evaluate(self, data_loader):
        """Evaluate model on validation/test data"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        # ROC AUC for positive class (scam)
        scam_probabilities = [prob[1] for prob in all_probabilities]
        auc = roc_auc_score(all_labels, scam_probabilities)

        return avg_loss, accuracy, f1, auc

    def save_model(self):
        """Save the trained model"""
        self.model_path.mkdir(exist_ok=True)

        logger.info(f"Saving model to {self.model_path}")
        self.model.save_pretrained(str(self.model_path))
        self.tokenizer.save_pretrained(str(self.model_path))

        # Save additional metadata
        metadata = {
            'model_type': 'DistilBERT',
            'max_length': self.max_length,
            'num_classes': 2,
            'class_names': ['legitimate', 'scam'],
            'training_params': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps
            }
        }

        with open(self.model_path / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def test_model(self, test_loader):
        """Test the final model"""
        logger.info("Testing final model...")

        test_loss, test_accuracy, test_f1, test_auc = self.evaluate(test_loader)

        logger.info("Final Test Results:")
        logger.info(".4f")
        logger.info(".4f")
        logger.info(".4f")
        logger.info(".4f")

        # Generate detailed classification report
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Classification report
        report = classification_report(all_labels, all_predictions, target_names=['legitimate', 'scam'])
        logger.info("Classification Report:")
        logger.info(report)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        logger.info("Confusion Matrix:")
        logger.info(cm)

        # Save test results
        test_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        with open(self.model_path / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        return test_results

def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT scam detection model')
    parser.add_argument('--model-path', type=str, default='./models/distilbert_scam',
                       help='Path to save/load the model')
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')

    args = parser.parse_args()

    # Create trainer
    trainer = DistilBERTTrainer(args.model_path, args.data_dir)

    # Update training parameters
    trainer.batch_size = args.batch_size
    trainer.num_epochs = args.epochs
    trainer.learning_rate = args.learning_rate

    # Load data
    train_loader, val_loader = trainer.load_data()

    # Load test data for final evaluation
    test_df = pd.read_csv(Path(args.data_dir) / "test.csv")
    test_dataset = ScamDataset(
        test_df['cleaned_text'].tolist(),
        test_df['label'].tolist(),
        trainer.tokenizer,
        trainer.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=trainer.batch_size, shuffle=False)

    # Train model
    training_stats = trainer.train(train_loader, val_loader)

    # Test final model
    test_results = trainer.test_model(test_loader)

    print("
🎉 Training completed successfully!"    print(f"📊 Best validation F1: {max([stat['val_f1'] for stat in training_stats]):.4f}")
    print(f"📊 Test F1: {test_results['test_f1']:.4f}")
    print(f"📊 Test AUC: {test_results['test_auc']:.4f}")
    print(f"💾 Model saved to: {args.model_path}")

if __name__ == "__main__":
    main()