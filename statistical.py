#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical corrector for Chinese Text Correction task.
This module implements statistical methods for correcting errors in Chinese text.
"""

import re
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from gensim.models import Word2Vec, KeyedVectors
from collections import Counter, defaultdict
from sklearn.model_selection import KFold
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizer

# Try to import optional dependencies
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Some features will be disabled.")

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # Import CRF if available
    try:
        import sklearn_crfsuite
        from sklearn_crfsuite import metrics

        CRF_AVAILABLE = True
    except ImportError:
        CRF_AVAILABLE = False
        print("Warning: sklearn_crfsuite not available. CRF features will be disabled.")

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    CRF_AVAILABLE = False
    print("Warning: scikit-learn not available. Some features will be disabled.")


EMBBEDING_PATH = 'light_Tencent_AILab_ChineseEmbedding.bin'


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=250):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item['source']
        target = item['target']

        # 编码 source
        encoding = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)           
        attention_mask = encoding['attention_mask'].squeeze(0)

        tokens = self.tokenizer.tokenize(source)
        labels = [int(s != t) for s, t in zip(source, target)]  # char level diff
        labels = labels[:self.max_length - 2]
        labels = [0] + labels + [0] 
        labels += [-100] * (self.max_length - len(labels))

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'char_labels': labels
        }


class CharLevelDetector(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2,
                              batch_first=True, bidirectional=True, num_layers=1)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state  # (B, T, H)
        
        x = self.dropout(hidden_states)
        norm_out = self.norm(x)
        logits = self.classifier(norm_out)               # (B, T, 2)
        return logits
   
class CorrectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=250):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item['source']
        target = item['target']

        source_ids = self.tokenizer(source)
        target_ids = self.tokenizer(target)

        source_ids = source_ids[:self.max_len] + [0] * (self.max_len - len(source_ids))
        target_ids = target_ids[:self.max_len] + [0] * (self.max_len - len(target_ids))

        return torch.tensor(source_ids), torch.tensor(target_ids)

class LSTMCorrector(nn.Module):
    def __init__(self, vocab_size, embed_size=200, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, 
                               batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, src):
        embed = self.dropout(self.embedding(src))           # (B, T, E)
        # embed = self.embedding(src)
        enc_out, _ = self.encoder(embed)                    # (B, T, 2H)
        enc_out = self.norm(enc_out)
        logits = self.fc(enc_out)                           # (B, T, V)
        return logits


class CharTokenizer:
    def __init__(self):
        self.char2idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.idx2char = ['<PAD>', '<UNK>', '<EOS>']

    def build_vocab(self, texts):
        for text in texts:
            for ch in text:
                if ch not in self.char2idx:
                    self.char2idx[ch] = len(self.idx2char)
                    self.idx2char.append(ch)
        return self.char2idx

    def __call__(self, text):
        ids = [self.char2idx.get(ch, 1) for ch in text]
        ids.append(self.char2idx['<EOS>'])  # 添加结束符
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            if i == self.char2idx['<PAD>']:
                continue
            if i == self.char2idx['<EOS>']:
                break
            chars.append(self.idx2char[i])
        return ''.join(chars)

    def vocab_size(self):
        return len(self.idx2char)


class StatisticalCorrector:
    """
    A statistical corrector for Chinese text.
    """

    def __init__(self, method='ngram'):
        """
        Initialize the statistical corrector.

        Args:
            method: The statistical method to use. Options: 'ngram', 'ml', 'crf'.
        """
        self.method = method
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"device: {self.device}")
        self.detector_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # N-gram language model
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.fourgram_counts = Counter()  # 4-gram for better context modeling

        # Character-level confusion matrix
        self.confusion_matrix = defaultdict(Counter)

        # Character error probabilities
        self.error_probs = defaultdict(float)

        # Phonetic and visual similarity matrices
        self.phonetic_similarity = defaultdict(dict)
        self.visual_similarity = defaultdict(dict)

        # Interpolation weights for different n-gram models
        self.lambda_1 = 0.1  # Weight for unigram
        self.lambda_2 = 0.3  # Weight for bigram
        self.lambda_3 = 0.4  # Weight for trigram
        self.lambda_4 = 0.2  # Weight for 4-gram

        # Smooth coefficients
        self.smooth_coeff = 1

        # Machine learning models
        self.ml_model = None
        self.vectorizer = None
        self.feature_scaler = None

        # Character corrections dictionary
        self.char_corrections = defaultdict(Counter)

        self.detector_model = None
        self.correct_model = None

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train the statistical corrector using the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        if self.method == 'ngram':
            self._train_ngram_model(train_data)
        elif self.method == 'ml' and SKLEARN_AVAILABLE:
            self._train_ml_model(train_data)
        else:
            print(f"Warning: Method '{self.method}' not available. Falling back to n-gram model.")
            self._train_ngram_model(train_data)

    def _train_ngram_model(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train an n-gram language model for text correction.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # Build n-gram language model from correct sentences
        for sample in train_data:
            # Use target (correct) text for building the language model
            text = sample['target']

            # Count unigrams
            for char in text:
                self.unigram_counts[char] += 1

            # Count bigrams
            for i in range(len(text) - 1):
                c1, c2 = text[i], text[i + 1]
                self.bigram_counts[c1 + c2] += 1

            # Count trigrams
            for i in range(len(text) - 2):
                c1, c2, c3 = text[i], text[i + 1], text[i + 2]
                self.trigram_counts[c1 + c2 + c3] += 1

            # Count 4-grams
            for i in range(len(text) - 3):
                c1, c2, c3, c4 = text[i], text[i + 1], text[i + 2], text[i + 3]
                self.fourgram_counts[c1 + c2 + c3 + c4] += 1

            # Confusion matrix and error probabilities
            if sample['label'] == 1:
                source = sample['source']
                target = sample['target']

                if len(source) == len(target):
                    for i, (s_char, t_char) in enumerate(zip(source, target)):
                        if s_char != t_char:
                            left_context = source[max(0, i - 2) : i]
                            right_context = source[i + 1 : min(len(source), i + 3)]
                            context = left_context + '_' + right_context

                            self.confusion_matrix[(s_char, context)][t_char] += 1
                            self.confusion_matrix[(s_char, '')][t_char] += 1
                            self.error_probs[s_char] += 1
                            self.char_corrections[s_char][t_char] += 1

        # Normalize error probabilities
        for char, count in self.error_probs.items():
            self.error_probs[char] = count / self.unigram_counts.get(char, 1)

        # Reduce duplicate calculations and accelerate reasoning
        vocab = list(self.unigram_counts.keys())
        V = len(vocab)

        total_unigrams = sum(self.unigram_counts.values())
        self.unigram_probs = {
            c: (self.unigram_counts[c] + 1 * self.smooth_coeff) / (total_unigrams + V * self.smooth_coeff)
            for c in vocab
        }

        print(
            f"Trained n-gram model with {len(self.unigram_counts)} unigrams, "
            f"{len(self.bigram_counts)} bigrams, "
            f"{len(self.trigram_counts)} trigrams, and "
            f"{len(self.fourgram_counts)} fourgrams."
        )

    def train_detector(self, train_data, epochs=3, max_len=200):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        device = self.device

        # 构建训练集 DataLoader
        train_dataset = TextDataset(train_data, tokenizer, max_length=max_len)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        model = CharLevelDetector().to(device)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)
            for batch in train_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['char_labels'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)  # (B, T, 2)
                loss = loss_fn(logits.view(-1, 2), labels.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        return model
    
    def load_tencent_embeddings(self, file_path: str, vocab: Dict[str, int], embedding_dim: int = 200) -> torch.Tensor:
        print(f"加载预训练词向量: {file_path}")
        wv = KeyedVectors.load_word2vec_format(file_path, binary=True, unicode_errors='ignore')
        
        embedding_matrix = torch.zeros((len(vocab), embedding_dim))
        for word, idx in vocab.items():
            try:
                if word in wv:
                    embedding_matrix[idx] = torch.tensor(wv[word], dtype=torch.float)
                else:
                    embedding_matrix[idx] = torch.randn(embedding_dim) * 0.01
                    if word == '<PAD>':
                        embedding_matrix[idx] = torch.zeros(embedding_dim)

            except KeyError:
                embedding_matrix[idx] = torch.randn(embedding_dim) * 0.01
                if word == '<PAD>':
                    embedding_matrix[idx] = torch.zeros(embedding_dim)
        
        return embedding_matrix

    def train_corrector(self, train_data: List[Dict[str, Any]], tokenizer, vocab_size, vocab = None, max_len=200, epochs=10, batch_size=16):
        device = self.device
        dataset = CorrectionDataset(train_data, tokenizer, max_len=max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        vocab_size = len(vocab) if vocab else vocab_size
        model = LSTMCorrector(vocab_size).to(device)
        embedding_matrix = self.load_tencent_embeddings(EMBBEDING_PATH, vocab)
            
        with torch.no_grad():
            model.embedding.weight.copy_(embedding_matrix)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for src, tgt in tqdm(loader, desc=f"Epoch {epoch+1}", ncols=100):
                src, tgt = src.to(device), tgt.to(device)

                optimizer.zero_grad()
                logits = model(src)
                loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

        return model

    def _train_ml_model(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train a machine learning model for text correction.

        Args:
            train_data: List of dictionaries containing the training data.
        """

        if not SKLEARN_AVAILABLE:
            print("Cannot train ML model: scikit-learn not available.")
            return

        import os

        max_len = max(max(len(item['source']), len(item['target'])) for item in train_data)
        if os.path.exists('models/detector_model'):
            self.detector_model = CharLevelDetector()
            self.detector_model.load_state_dict(torch.load('models/detector_model', map_location=self.device))
            self.detector_model.to(self.device)
            print("Loaded existing detector model.")
        else:
            self.detector_model = self.train_detector(train_data, epochs=5, max_len=max_len)
            torch.save(self.detector_model.state_dict(), 'models/detector_model')
            print("Detector model trained.")

        train_data_wrong = [item for item in train_data if item['label'] == 1]
        self.correct_tokenizer = CharTokenizer()
        # self.correct_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        vocab = self.correct_tokenizer.build_vocab([item['source'] for item in train_data] + [item['target'] for item in train_data_wrong])

        max_len = max(max(len(item['source']), len(item['target'])) for item in train_data_wrong)
        print("Max_len:", max_len)

        if os.path.exists('models/correct_model_tencent.pt') :
            self.correct_model = LSTMCorrector(self.correct_tokenizer.vocab_size()).to(self.device)
            self.correct_model.load_state_dict(torch.load('models/correct_model_tencent.pt', map_location=self.device))
            print("Loaded existing corrector model.")
        else:
            # self.correct_model = self.train_corrector(train_data_wrong, self.correct_tokenizer, vocab = vocab, vocab_size=self.correct_tokenizer.vocab_size(), max_len=max_len, epochs=20)
            self.correct_model = self.train_corrector(train_data_wrong, self.correct_tokenizer, vocab = vocab, vocab_size=len(vocab), max_len=max_len, epochs=10)
            
            # save the correct model
            torch.save(self.correct_model.state_dict(), 'models/correct_model_tencent.pt')
            print("Corrector model trained.")

    def correct(self, text: str) -> str:
        """
        Apply statistical correction to the input text.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text.
        """
        if self.method == 'ngram':
            return self._correct_with_ngram(text)
        elif self.method == 'ml' and SKLEARN_AVAILABLE:
            return self._correct_with_ml(text)
        else:
            return self._correct_with_ngram(text)

    def _correct_with_ngram(self, text: str) -> str:
        """
        Correct text using the n-gram language model.

        Args:
            text: Input text.

        Returns:
            Corrected text.
        """
        corrected_text = list(text)  # Convert to list for character-by-character editing

        # Check each character for potential errors
        for i in range(len(text)):
            char = text[i]

            # Skip characters with low error probability
            if self.error_probs.get(char, 0) < 0.01:
                continue

            # Get context for this character
            left_context = text[max(0, i - 3) : i]
            right_context = text[i + 1 : min(len(text), i + 4)]
            context = left_context + '_' + right_context

            # Check if we have seen this character in this context before
            if (char, context) in self.confusion_matrix and self.confusion_matrix[(char, context)]:
                # Get the most common correction for this character in this context
                correction = self.confusion_matrix[(char, context)].most_common(1)[0][0]
                corrected_text[i] = correction
                continue

            # If no specific context match, check general confusion matrix
            if (char, '') in self.confusion_matrix and self.confusion_matrix[(char, '')]:
                # Get the most common correction for this character
                correction = self.confusion_matrix[(char, '')].most_common(1)[0][0]
                # Only apply if it's a common error
                if self.confusion_matrix[(char, '')][correction] > 3 and self.confusion_matrix[(char, '')][correction] < 100 and self.confusion_matrix[(char, '')][correction] / self.unigram_counts.get(char, 1) >= 0.05:
                    corrected_text[i] = correction
                    continue
            

            # If no direct match, use interpolated n-gram model for characters with high error probability
            if self.error_probs.get(char, 0) >= 0.05 and i > 0 and i < len(text) - 1:
                # Generate candidate corrections
                candidates = set()

                # Add common characters as candidates
                # candidates.update(list(self.unigram_counts.keys())[:300])  # Top 300 most common characters
                top_chars = [char for char, _ in self.unigram_counts.most_common(300)]
                candidates.update(top_chars)

                # Add correction candidates from confusion matrix
                for context_key in self.confusion_matrix:
                    if context_key[0] == char:
                        candidates.update(self.confusion_matrix[context_key].keys())

                # Try all candidates and find the one with highest probability
                best_score = -float('inf')
                best_char = char

                for candidate in candidates:
                    # Skip the original character
                    if candidate == char:
                        continue

                    # Calculate interpolated score using all n-gram models
                    score = 0

                    # Unigram probability (with smoothing)
                    unigram_prob = self.unigram_probs[candidate]
                    score += self.lambda_1 * unigram_prob

                    # Bigram, trigram, and 4-gram probabilities
                    if len(left_context) > 0:
                        prev_char = left_context[-1]
                        bigram_prob = (self.bigram_counts.get(prev_char + candidate, 0) + 1 * self.smooth_coeff) / (
                            self.unigram_counts.get(prev_char, 0) + len(self.unigram_counts) * self.smooth_coeff
                        )
                    
                        score += self.lambda_2 * bigram_prob

                    # Trigram probability
                    if len(left_context) > 1:
                        prev_char = left_context[-2] + left_context[-1]
                        trigram_prob = (self.trigram_counts.get(prev_char + candidate, 0) + 1 * self.smooth_coeff) / (
                            self.bigram_counts.get(prev_char, 0) + len(self.unigram_counts) * self.smooth_coeff
                        )
                        score += self.lambda_3 * trigram_prob

                    # 4-gram probability
                    if len(left_context) > 2:
                        prev_char = left_context[-3] + left_context[-2] + left_context[-1]
                        fourgram_prob = (self.fourgram_counts.get(prev_char + candidate, 0) + 1 * self.smooth_coeff) / (
                            self.trigram_counts.get(prev_char, 0) + len(self.unigram_counts) * self.smooth_coeff
                        )
                        score += self.lambda_4 * fourgram_prob

                    if score > best_score:
                        best_score = score
                        best_char = candidate

                # Calculate score for the original character
                original_score = 0

                # Unigram probability
                original_unigram_prob = (self.unigram_counts.get(char, 0) + 1 * self.smooth_coeff) / (
                    sum(self.unigram_counts.values()) + len(self.unigram_counts) * self.smooth_coeff
                )
                original_score += self.lambda_1 * original_unigram_prob

                # Bigram probabilities
                if len(left_context) > 0:
                    original_bigram_left = left_context[-1] + char
                    original_bigram_left_prob = (self.bigram_counts.get(original_bigram_left, 0) + 1 * self.smooth_coeff) / (
                        self.unigram_counts.get(left_context[-1], 0) + len(self.unigram_counts) * self.smooth_coeff
                    )
                    original_score += self.lambda_2 * original_bigram_left_prob

                # Trigram probabilities
                if len(left_context) > 1:
                    original_trigram_left = left_context[-2] + left_context[-1] + char
                    original_trigram_left_prob = (self.trigram_counts.get(original_trigram_left, 0) + 1 * self.smooth_coeff) / (
                        self.bigram_counts.get(left_context[-2] + left_context[-1], 0) + len(self.unigram_counts) * self.smooth_coeff
                    )
                    original_score += self.lambda_3 * original_trigram_left_prob

                # 4-gram probabilities
                if len(left_context) > 2:
                    original_fourgram_left = left_context[-3] + left_context[-2] + left_context[-1] + char
                    original_fourgram_left_prob = (self.fourgram_counts.get(original_fourgram_left, 0) + 1 * self.smooth_coeff) / (
                        self.trigram_counts.get(left_context[-3] + left_context[-2] + left_context[-1], 0) + len(self.unigram_counts) * self.smooth_coeff
                    )
                    original_score += self.lambda_4 * original_fourgram_left_prob

                # Only replace if the new score is significantly better
                threshold = (1.2 + self.error_probs.get(char, 0) * 3) * 50
                if best_score > original_score * threshold:
                    corrected_text[i] = best_char

        return ''.join(corrected_text)

    def correct_sentence(self, model, sentence, tokenizer, max_len=250):
        model.eval()
        device = next(model.parameters()).device
        input_ids = tokenizer(sentence)[:max_len]
        input_tensor = torch.tensor([input_ids + [0]*(max_len - len(input_ids))]).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
            return tokenizer.decode(pred_ids)

    def _correct_with_ml(self, text: str) -> str:
        self.detector_model.eval()
        self.correct_model.eval()

        max_len = 434
        encoding = self.detector_tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        

        # 检测模型输出每个字符的是否错误的概率
        with torch.no_grad():
            logits = self.detector_model(input_tensor, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            error_probs = probs[0, :, 1]

        if (error_probs > 0.65).any().item():
            corrected_sentence = self.correct_sentence(self.correct_model, text, self.correct_tokenizer, max_len=max_len)
            # return self.correct_sentence(self.correct_model, text, self.correct_tokenizer, max_len=max_len)
        else:
            return text

        corrected_chars = list(corrected_sentence)
        final_chars = []
        for i in range(len(text)):
            if i >= len(error_probs):
                break
            if error_probs[i] > 0.007 and i < len(corrected_chars):
                final_chars.append(corrected_chars[i])
            else:
                final_chars.append(text[i])

        return ''.join(final_chars)
    
