#!/usr/bin/env python3
"""
Data preparation script for PhishNetra
Handles data cleaning, augmentation, and preprocessing
"""

import os
import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparator:
    """Handles data preparation for scam detection models"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Scam patterns for augmentation
        self.scam_templates = {
            "phishing": [
                "Your account has been suspended. Click here to verify: {url}",
                "Security alert: Unusual activity detected. Confirm your identity at {url}",
                "Your payment failed. Update your information here: {url}",
                "Account verification required. Please login to {url}",
            ],
            "financial": [
                "Congratulations! You've won ${amount}. Send ${fee} to claim your prize.",
                "Urgent: Transfer ${amount} to this account or face penalties.",
                "Investment opportunity: Double your money in 24 hours. Send ${amount} now.",
                "Bank transfer needed: Send ${amount} immediately to avoid account closure.",
            ],
            "authority": [
                "Official notice from IRS: You owe ${amount} in taxes. Pay now or face arrest.",
                "FBI investigation: Your account is linked to fraud. Verify here: {url}",
                "Police department: Warrant issued for your arrest. Pay ${amount} fine now.",
                "Government refund: Claim your ${amount} stimulus check. Provide details here.",
            ],
            "social_engineering": [
                "Hi {name}, I need ${amount} urgently. Can you send it to this account?",
                "Mom/dad, my phone broke. Send ${amount} for a new one. Love you.",
                "Friend in trouble: Send ${amount} to help me out. I'll pay you back.",
                "Emergency: Hospital bill of ${amount}. Need money immediately.",
            ]
        }

        # Common scam URLs and amounts
        self.urls = [
            "http://secure-bank-login.com",
            "http://account-verify.net",
            "http://paypal-secure.org",
            "http://amazon-support.co",
            "http://irs-gov.us",
            "http://fbi-investigation.com",
            "https://bankofamerica-secure.com",
            "https://chase-login.net",
            "https://irs-refund.org",
            "https://social-security.gov-verify.com"
        ]

        self.amounts = ["500", "1000", "2500", "5000", "10000", "25000"]
        self.names = ["John", "Mike", "Sarah", "Emma", "David", "Lisa"]

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw scam and legitimate text data"""
        logger.info("Loading raw data...")

        # This is a placeholder - in real implementation, load from various sources
        scam_texts = []
        legitimate_texts = []

        # Generate synthetic scam data
        for category, templates in self.scam_templates.items():
            for template in templates:
                # Generate multiple variations
                for _ in range(50):  # Generate 50 variations per template
                    text = self._augment_template(template)
                    scam_texts.append({
                        'text': text,
                        'label': 1,  # 1 = scam
                        'category': category,
                        'source': 'synthetic'
                    })

        # Generate synthetic legitimate data
        legitimate_templates = [
            "Hi {name}, how are you doing today?",
            "Thank you for your order #{number}. It will be delivered by {date}.",
            "Your appointment is confirmed for {date} at {time}.",
            "Welcome to our newsletter! Here's what's new this week.",
            "Payment received: Thank you for your ${amount} donation.",
            "Your package has been shipped. Track it here: {url}",
            "Meeting reminder: Project review at {time} tomorrow.",
            "Invoice #{number} for ${amount} is now available.",
            "Password reset successful. You can now login to your account.",
            "Thank you for contacting customer support. We'll respond within 24 hours."
        ]

        for template in legitimate_templates:
            for _ in range(30):  # Generate 30 variations per template
                text = self._augment_legitimate_template(template)
                legitimate_texts.append({
                    'text': text,
                    'label': 0,  # 0 = legitimate
                    'category': 'legitimate',
                    'source': 'synthetic'
                })

        # Combine and create DataFrame
        all_data = scam_texts + legitimate_texts
        df = pd.DataFrame(all_data)

        logger.info(f"Generated {len(scam_texts)} scam texts and {len(legitimate_texts)} legitimate texts")
        return df

    def _augment_template(self, template: str) -> str:
        """Augment scam template with random values"""
        text = template

        # Replace placeholders
        if '{url}' in text:
            text = text.replace('{url}', np.random.choice(self.urls))

        if '{amount}' in text:
            text = text.replace('{amount}', np.random.choice(self.amounts))

        if '{fee}' in text:
            fee = str(np.random.randint(50, 500))
            text = text.replace('{fee}', fee)

        if '{name}' in text:
            text = text.replace('{name}', np.random.choice(self.names))

        # Add variations
        variations = [
            lambda t: t,  # Original
            lambda t: t.upper(),  # ALL CAPS
            lambda t: t.lower(),  # all lowercase
            lambda t: re.sub(r'(\w+)', lambda m: m.group(1).capitalize(), t),  # Title case
            lambda t: self._add_typos(t),  # Add typos
            lambda t: self._add_emojis(t),  # Add emojis
            lambda t: self._add_urgency(t),  # Add urgency markers
        ]

        variation = np.random.choice(variations)
        text = variation(text)

        return text

    def _augment_legitimate_template(self, template: str) -> str:
        """Augment legitimate template with random values"""
        text = template

        # Replace placeholders
        if '{name}' in text:
            text = text.replace('{name}', np.random.choice(self.names))

        if '{number}' in text:
            number = str(np.random.randint(10000, 99999))
            text = text.replace('{number}', number)

        if '{amount}' in text:
            text = text.replace('{amount}', np.random.choice(self.amounts))

        if '{date}' in text:
            dates = ["Monday", "tomorrow", "next week", "Friday"]
            text = text.replace('{date}', np.random.choice(dates))

        if '{time}' in text:
            times = ["2 PM", "10 AM", "3:30 PM", "9 AM"]
            text = text.replace('{time}', np.random.choice(times))

        if '{url}' in text:
            urls = ["https://tracking.example.com", "https://support.company.com", "https://portal.business.com"]
            text = text.replace('{url}', np.random.choice(urls))

        return text

    def _add_typos(self, text: str) -> str:
        """Add realistic typos to text"""
        # Simple typo generation (can be enhanced)
        words = text.split()
        if len(words) > 0:
            # Randomly add typos to some words
            for i in range(len(words)):
                if np.random.random() < 0.1:  # 10% chance
                    word = words[i]
                    if len(word) > 3:
                        # Swap two adjacent characters
                        pos = np.random.randint(0, len(word) - 1)
                        word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                        words[i] = word

        return ' '.join(words)

    def _add_emojis(self, text: str) -> str:
        """Add emojis to text"""
        emojis = ["❗", "🚨", "⚠️", "💰", "🔒", "🏦", "📧", "📱"]
        if np.random.random() < 0.3:  # 30% chance
            emoji = np.random.choice(emojis)
            text += f" {emoji}"

        return text

    def _add_urgency(self, text: str) -> str:
        """Add urgency markers"""
        urgency_markers = [
            "URGENT: ", "IMPORTANT: ", "ACTION REQUIRED: ",
            "ATTENTION: ", "IMMEDIATE: ", "CRITICAL: "
        ]

        if np.random.random() < 0.2:  # 20% chance
            marker = np.random.choice(urgency_markers)
            text = marker + text

        return text

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text data"""
        logger.info("Cleaning text data...")

        def clean_single_text(text: str) -> str:
            if not isinstance(text, str):
                return ""

            # Convert to lowercase
            text = text.lower()

            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove URLs (replace with placeholder)
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)

            # Remove email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

            # Remove phone numbers
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

            # Normalize money amounts
            text = re.sub(r'\$[\d,]+(?:\.\d{2})?', '[MONEY]', text)

            # Remove extra whitespace
            text = text.strip()

            return text

        df['cleaned_text'] = df['text'].apply(clean_single_text)
        return df

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset to have equal scam/legitimate samples"""
        logger.info("Balancing dataset...")

        scam_count = len(df[df['label'] == 1])
        legit_count = len(df[df['label'] == 0])

        logger.info(f"Original distribution: {scam_count} scams, {legit_count} legitimate")

        if scam_count > legit_count:
            # Oversample legitimate texts
            legit_df = df[df['label'] == 0]
            oversample_count = scam_count - legit_count
            oversampled_legit = legit_df.sample(n=oversample_count, replace=True, random_state=42)
            df = pd.concat([df, oversampled_legit], ignore_index=True)
        elif legit_count > scam_count:
            # Oversample scam texts
            scam_df = df[df['label'] == 1]
            oversample_count = legit_count - scam_count
            oversampled_scam = scam_df.sample(n=oversample_count, replace=True, random_state=42)
            df = pd.concat([df, oversampled_scam], ignore_index=True)

        final_scam_count = len(df[df['label'] == 1])
        final_legit_count = len(df[df['label'] == 0])

        logger.info(f"Balanced distribution: {final_scam_count} scams, {final_legit_count} legitimate")

        return df

    def create_splits(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits"""
        logger.info("Creating data splits...")

        # First split: train+val and test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )

        # Second split: train and validation
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=42, stratify=train_val_df['label']
        )

        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def save_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save processed data to disk"""
        logger.info("Saving processed data...")

        output_dir = self.data_dir / "processed"
        output_dir.mkdir(exist_ok=True)

        # Save as CSV
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "validation.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        # Save as JSON for easy loading
        datasets = {
            'train': train_df.to_dict('records'),
            'validation': val_df.to_dict('records'),
            'test': test_df.to_dict('records')
        }

        with open(output_dir / "datasets.json", 'w') as f:
            json.dump(datasets, f, indent=2)

        # Save statistics
        stats = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'scam_ratio': (len(train_df[train_df['label'] == 1]) + len(val_df[val_df['label'] == 1]) + len(test_df[test_df['label'] == 1])) / (len(train_df) + len(val_df) + len(test_df)),
            'categories': df['category'].value_counts().to_dict()
        }

        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Data saved to {output_dir}")

    def prepare_full_pipeline(self):
        """Run the complete data preparation pipeline"""
        logger.info("Starting data preparation pipeline...")

        # Load raw data
        df = self.load_raw_data()

        # Clean text
        df = self.clean_text(df)

        # Balance dataset
        df = self.balance_dataset(df)

        # Create splits
        train_df, val_df, test_df = self.create_splits(df)

        # Save processed data
        self.save_data(train_df, val_df, test_df)

        logger.info("Data preparation pipeline completed successfully!")

        return train_df, val_df, test_df


if __name__ == "__main__":
    # Run data preparation
    preparator = DataPreparator()
    train_df, val_df, test_df = preparator.prepare_full_pipeline()

    print("
📊 Data Preparation Summary:"    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Scam ratio: {len(train_df[train_df['label'] == 1]) / len(train_df):.2%}")