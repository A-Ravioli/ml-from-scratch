# BERT Family: Bidirectional Encoder Representations from Transformers

## Prerequisites
- Transformer encoder architecture
- Masked language modeling
- Pre-training and fine-tuning paradigms
- Transfer learning concepts

## Learning Objectives
- Master bidirectional language modeling with Transformers
- Understand masked language modeling and next sentence prediction
- Implement BERT pre-training and fine-tuning
- Analyze the evolution from BERT to modern encoder models
- Connect to downstream NLP applications

## Mathematical Foundations

### 1. Bidirectional Context Modeling

#### Masked Language Modeling (MLM)
Randomly mask 15% of tokens and predict them:
```
P(x_masked | x_context) = softmax(W · h_masked + b)
```

#### Masking Strategy
- 80%: Replace with [MASK] token
- 10%: Replace with random token  
- 10%: Keep original token

### 2. BERT Architecture

#### Encoder-Only Design
Uses only the encoder stack from Transformer:
- Multi-head self-attention
- Position-wise feed-forward
- Residual connections and layer normalization

#### Special Tokens
- **[CLS]**: Classification token (sentence-level representation)
- **[SEP]**: Separator between sentences
- **[MASK]**: Masked token placeholder
- **[PAD]**: Padding token

### 3. Pre-training Objectives

#### Masked Language Modeling (MLM)
Predict randomly masked tokens using bidirectional context

#### Next Sentence Prediction (NSP)
Binary classification: are two sentences consecutive?
```
P(IsNext | [CLS] sentence_A [SEP] sentence_B [SEP])
```

### 4. BERT Variants Evolution

#### BERT Base/Large
- **Base**: 12 layers, 768 hidden, 110M parameters
- **Large**: 24 layers, 1024 hidden, 340M parameters

#### RoBERTa (2019)
- Remove NSP objective
- Dynamic masking
- Larger batch sizes and learning rates
- More training data

#### ALBERT (2019)
- Parameter sharing across layers
- Factorized embeddings
- Sentence-order prediction instead of NSP

#### DistilBERT (2019)
- Knowledge distillation from BERT
- 6 layers, 97% performance, 60% size

#### ELECTRA (2020)
- Replaced token detection instead of MLM
- More efficient pre-training
- Generator-discriminator framework

### 5. Fine-tuning Applications

#### Sequence Classification
Add classification head to [CLS] token:
```
logits = W_cls · h_[CLS] + b_cls
```

#### Token Classification (NER, POS)
Classify each token individually:
```
logits_i = W_token · h_i + b_token
```

#### Question Answering
Predict start and end positions for answer span:
```
start_logits = W_start · h + b_start
end_logits = W_end · h + b_end
```

## Implementation Details

See `exercise.py` for implementations of:
1. BERT encoder architecture
2. Masked language modeling pre-training
3. Fine-tuning for various downstream tasks
4. Attention visualization and analysis
5. Model variants comparison