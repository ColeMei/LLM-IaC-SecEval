### Stage 2 — Model Training Pipeline (Pilot Phase)

Objective: Build training pipelines for both generative and encoder approaches to classify GLITCH detections as TP/FP.

#### Pipeline Steps

1. **Generative Approach**: CodeLLaMA with LoRA fine-tuning

   - Input: `with_prompt` field (full static analysis prompt)
   - Target: Generate "TP" or "FP" text
   - Matrix: `[batch_size, max_length=512]`

2. **Encoder Approach**: CodeBERT binary classification
   - Input: `content` field (code snippet only)
   - Target: Binary classification (0=FP, 1=TP)
   - Matrix: `[batch_size, max_length=256]`

---

### Files and Usage

#### 1. Generative Training (CodeLLaMA + LoRA)

`src/iac_filter_training/generative_trainer.py`

- Uses `with_prompt` field as input
- LoRA fine-tuning for efficiency
- Generates "TP" or "FP" responses
- Default: CodeLlama-7b-hf model

#### 2. Encoder Training (CodeBERT Classification)

`src/iac_filter_training/encoder_trainer.py`

- Uses `content` field as input
- Binary classification (TP=1, FP=0)
- Full fine-tuning approach
- Default: microsoft/codebert-base model

#### 3. Main Training Interface

`experiments/iac_filter_training/train_models.py`

Key flags:

- `--approach` generative|encoder (required)
- `--model-name` custom model name (optional)
- `--batch-size` training batch size (default: 2)
- `--num-epochs` training epochs (default: 2)

Examples:

```bash
# Train generative model
python experiments/iac_filter_training/train_models.py --approach generative

# Train encoder model
python experiments/iac_filter_training/train_models.py --approach encoder

# Custom parameters
python experiments/iac_filter_training/train_models.py \
    --approach generative \
    --batch-size 1 \
    --num-epochs 2
```

Outputs (dir): `experiments/iac_filter_training/models/{approach}/`

- Trained model weights
- Tokenizer files
- Training logs and checkpoints

#### Current Status

- ✅ Training pipelines implemented
- ✅ Input matrix design finalized
- 🔄 Ready for feasibility testing
- 🔄 Ready for HPC scaling

#### Next Steps

1. Test training on demo dataset (14 samples)
2. Scale to larger dataset (600 train, 75 val)
3. Compare approaches on Oracle test set
