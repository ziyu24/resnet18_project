from transformers import TrainingArguments, Trainer
from transformers import AutoImageProcessor
import numpy as np

# model_name_or_path = 'google/vit-base-patch16-224-in21k'
model_name_or_path = 'microsoft/resnet-18'


processor = AutoImageProcessor.from_pretrained(model_name_or_path)

def transform(example_batch):
    # 强制将图像转换为RGB格式, RS里面有灰度图
    inputs = processor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

from datasets import load_dataset
# dataset_name_or_path = 'beans'
# dataset_name_or_path = '/home/user/data/Butterfly'
dataset_name_or_path = '/home/user/data/025-Push-and-Pull-Network/FGSC-23-NoAug'
ds = load_dataset(dataset_name_or_path)
prepared_ds = ds.with_transform(transform)

ds_train = prepared_ds["train"]
ds_val = prepared_ds["validation"]

labels = ds['train'].features['label'].names


from transformers import AutoModelForImageClassification, AutoConfig
config = AutoConfig.from_pretrained(model_name_or_path, num_labels=len(labels))
model = AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    config=config,
    ignore_mismatched_sizes=True
)


import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


training_args = TrainingArguments(
    output_dir='saves_r18_fgsc23',                         # 
    save_strategy='epoch',               
    save_steps=1,                            # 
    save_total_limit=2,                         # 留存的检查点最大数目
    seed=3407,                                   # 

    num_train_epochs=20,                         # epochs
    per_device_train_batch_size=16,              # 每卡bs
    gradient_accumulation_steps=1,              # 小模型没必要, 到了8b就要设置bs 累计1k
    optim='adamw_torch',                        # 显示指定optimizers
    lr_scheduler_type='cosine',                 # cosine 
    learning_rate=1e-3,                         # lr
    warmup_ratio=0.1,    # 
    
    remove_unused_columns=False,
    
    logging_steps=10,                           # 
    report_to='wandb', 
    
    eval_strategy="epoch",
    eval_steps=1, 
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
                                 
)

trainer = Trainer(
    model=model,                    # 模型实例
    args=training_args,             # 训练参数
    train_dataset=ds_train,         # 训练集
    tokenizer=processor,            # preprocess
    data_collator=collate_fn,    # data collator
    # eval
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,

)

train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# eval
metrics = trainer.evaluate(ds_val)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)