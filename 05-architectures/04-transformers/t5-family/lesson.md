# T5 Family: Text-to-Text Transfer Transformer

## Prerequisites
- Encoder-decoder Transformer architecture
- Sequence-to-sequence modeling
- Transfer learning and multi-task learning
- Text-to-text formulation concepts

## Learning Objectives
- Master the text-to-text unified framework
- Understand encoder-decoder pre-training strategies
- Implement T5 architecture and span denoising
- Analyze multi-task learning in NLP
- Connect to modern encoder-decoder models

## Mathematical Foundations

### 1. Text-to-Text Framework

#### Unified Formulation
All NLP tasks as text-to-text:
```
Input: "translate English to French: Hello world"
Output: "Bonjour le monde"

Input: "summarize: [long document]"
Output: "[summary]"

Input: "sentiment: This movie is great!"
Output: "positive"
```

### 2. T5 Architecture

#### Encoder-Decoder Design
- **Encoder**: Bidirectional self-attention
- **Decoder**: Causal self-attention + encoder-decoder attention
- **Relative Position Embeddings**: Instead of absolute positions

#### Span Denoising Pre-training
- Corrupt spans of text with sentinel tokens
- Train to reconstruct original spans
```
Input: "Thank you <X> me to your party <Y> week"
Target: "<X> for inviting <Y> last <Z>"
```

### 3. Model Variants

#### T5 Sizes
- **Small**: 60M parameters
- **Base**: 220M parameters  
- **Large**: 770M parameters
- **3B**: 3B parameters
- **11B**: 11B parameters

#### mT5 (Multilingual)
- Trained on 101 languages
- Same architecture as T5
- Cross-lingual transfer capabilities

#### UL2 (Unified Language Learner)
- Multiple pre-training objectives
- R-Denoiser, S-Denoiser, X-Denoiser
- Better few-shot performance

### 4. Training Methodology

#### Pre-training
- **Objective**: Span denoising
- **Data**: Colossal Clean Crawled Corpus (C4)
- **Masking**: 15% corruption rate, average span length 3

#### Fine-tuning
- Task-specific adaptation
- Different learning rates for encoder/decoder
- Multi-task fine-tuning possible

## Implementation Details

See `exercise.py` for implementations of:
1. T5 encoder-decoder architecture
2. Span denoising pre-training
3. Text-to-text task formulations
4. Multi-task fine-tuning
5. Generation strategies