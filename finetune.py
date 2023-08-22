import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import os
from peft import PeftModel
from peft import get_peft_model,LoraConfig,TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
os.environ["WANDB_DISABLED"] = 'true'
class Lora_finetune:
    def __init__(self,last_name,timestamp):
        self.last_name = last_name
        self.timestamp = timestamp
        self.tokenizer = AutoTokenizer.from_pretrained(self.last_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.last_name, trust_remote_code=True).cuda()
        self.max_source_length = 512
        self.max_target_length = 512
        self.source_prefix = ''

    #no history处理
    def preprocess_no_history(self, examples):
        max_seq_length = self.max_source_length + self.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples['key'])):
            if examples['key'][i] and examples['value'][i]:
                query, answer = examples['key'][i], examples['value'][i]

                history = []
                #prompt = self.tokenizer.build_prompt(query, history)

                prompt = query
                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                              max_length=self.max_source_length)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                              max_length=self.max_target_length)

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len
                labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
        return model_inputs

    # history处理
    def preprocess(self,examples):
        max_seq_length = self.max_source_length + self.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples['key'])):
            if examples['key'][i] and examples['value'][i]:
                query, answer = examples['key'][i], examples['value'][i]

                history = examples['history'][i]
                prompt = self.tokenizer.build_prompt(query, history)

                prompt = prompt
                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                        max_length=self.max_source_length)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                        max_length=self.max_target_length)

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len
                labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
        return model_inputs

    def train(self):
        print("lora start")
        ds_train_raw = load_dataset("json",data_files=os.path.join('datas',self.timestamp+'.json'))['train']


        ds_train = ds_train_raw.map(
            self.preprocess,
            batched = True,
            remove_columns=ds_train_raw.column_names
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["key","value"]
        )

        peft_model = get_peft_model(self.model,peft_config)
        
        trainer = Trainer(
            model=peft_model,
            train_dataset=ds_train,
            args=TrainingArguments(
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=50,
                learning_rate=5e-5,
                logging_steps=1,
                output_dir='outputs'
            ),
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer
            )
        )

        #训练时节约GPU占用
        self.model.config.use_cache = True
        self.model.supports_gradient_checkpointing = True 
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        trainer.train()
        ckpt_path = self.timestamp
        peft_model.save_pretrained(ckpt_path)
        model_old = AutoModel.from_pretrained(self.last_name,trust_remote_code=True)
        peft_loaded = PeftModel.from_pretrained(model_old,ckpt_path)
        model_new = peft_loaded.merge_and_unload()


        save_path = 'chatglm2-update'
        model_new.save_pretrained(save_path,max_shard_size = '2GB')
        self.tokenizer.save_pretrained(save_path)
        print('lora end')

# if __name__ == '__main__':
#     lora = Lora_finetune('THUDM/chatglm2-6b','2023-08-21-20-15-11')
#     lora.train()
