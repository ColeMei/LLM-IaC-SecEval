import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChefDetectionDataset(Dataset):
    """Dataset for Chef detection classification using generative approach."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data and format for training."""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                # Use with_prompt field as input
                input_text = sample['with_prompt']
                # Target: TP or FP
                target = sample['label']
                
                data.append({
                    'input_text': input_text,
                    'target': target,
                    'smell': sample['smell'],
                    'confidence': sample['confidence']
                })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Format the input with target
        full_text = f"{sample['input_text']}\n\nAnswer: {sample['target']}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

class GenerativeTrainer:
    """Trainer for generative approach using CodeLLaMA with LoRA."""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        output_dir: str = "experiments/iac_filter_training/models/generative",
        lora_config: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_config = lora_config or self._default_lora_config()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add LoRA adapters
        self.model = get_peft_model(self.model, LoraConfig(**self.lora_config))
        
    def _default_lora_config(self) -> Dict:
        """Default LoRA configuration for efficiency."""
        return {
            "task_type": TaskType.CAUSAL_LM,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
    
    def prepare_datasets(self, train_path: str, val_path: str):
        """Prepare train and validation datasets."""
        self.train_dataset = ChefDetectionDataset(train_path, self.tokenizer)
        self.val_dataset = ChefDetectionDataset(val_path, self.tokenizer)
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def train(
        self,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        save_steps: int = 100,
        eval_steps: int = 50
    ):
        """Train the model."""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def predict(self, input_text: str) -> str:
        """Make prediction on new input."""
        self.model.eval()
        
        # Format input
        input_text = f"{input_text}\n\nAnswer:"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer (last part after "Answer:")
        answer = generated_text.split("Answer:")[-1].strip()
        
        return answer

def main():
    """Main training script for generative approach."""
    
    # Initialize trainer
    trainer = GenerativeTrainer(
        model_name="codellama/CodeLlama-7b-hf",  # Start with 7B for demo
        output_dir="experiments/iac_filter_training/models/generative"
    )
    
    # Prepare datasets
    train_path = "experiments/iac_filter_training/data/formatted_dataset/chef_train.jsonl"
    val_path = "experiments/iac_filter_training/data/formatted_dataset/chef_val.jsonl"
    
    trainer.prepare_datasets(train_path, val_path)
    
    # Train the model
    trainer.train(
        batch_size=1,  # Small batch size for demo
        learning_rate=5e-5,
        num_epochs=2,  # Few epochs for demo
        warmup_steps=10,
        save_steps=50,
        eval_steps=25
    )
    
    # Test prediction
    test_input = "You are a static analyzer...\n\n### RAW CODE INPUT (chef)\n\npassword = 'secret123'\n\nBased on the static analysis rules above, does this code contain a true instance of \"Hard-coded secret\"?\n\nAnswer (YES or NO only):"
    prediction = trainer.predict(test_input)
    print(f"Test prediction: {prediction}")

if __name__ == "__main__":
    main()
