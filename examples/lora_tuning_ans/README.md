## LoRA Tuning on Customization Dataset

This example follows [HF's Prompt Tuning example](https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning)
which demonstrates how to apply our tuning to any HF example.

### Dataset

The first step is to make a `SFTTrainer`-compatible dataset. 

Update the script ```ansible_fms_hf_tuning.py```in the variable ```data_file_path``` with the path to your data (e.g., ans_train_data.json)

Make sure the above script put your data in the `SFTTrainer` style (following instruction [here](https://huggingface.co/docs/trl/main/en/sft_trainer#format-your-input-prompts)).

This script should produce a file in the format accepted by `SFTTrainer` (e.g., ans_train_data_ftt.json)

### Update the FSDP and PEFT configs

For a Granite model, use the following configurations:

* Update the configuration for the peft tuning by modifing the following file ```tuning/config/fsdp_config.json```.

```
{
  "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
  "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
  "fsdp_cpu_ram_efficient_loading": "False",
  "fsdp_forward_prefetch": "True",
  "fsdp_offload_params": "False",
  "fsdp_state_dict_type": "SHARDED_STATE_DICT",
  "fsdp_sync_module_states": "False",
  "fsdp_transformer_layer_cls_to_wrap": "GPTBigCodeBlock",
  "fsdp_use_orig_params": "True",
  "activation_checkpointing": "True"
}
```

* Update the configuration for the peft tuning by modifing the following file ```tuning/config/peft_config.json```.


```
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"])
    bias = "none"
    lora_dropout: float = 0.05
```

You can play with other values for the parameters of the Lora Configuration once you make sure the above configuration works for you.

### Prompt Tuning
Using PEFT method:  LORA with a Granite model.

```bash
# replace these with your values
MODEL_PATH=GRANITE_MODEL_PATH
DATA_PATH=ans_train_data_ftt.json
OUTPUT_PATH=out_granite_lora_ans_train_data

export CUDA_VISIBLE_DEVICES=0

torchrun \
--nnodes=1 \
--nproc_per_node=1  \
--node_rank 0 \
--master_port=1234  \
tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--data_path $DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--peft_method "lora" \
--tokenizer_name_or_path $MODEL_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 1  \
--per_device_eval_batch_size 1  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "epoch"  \
--learning_rate 1e-5  \
--weight_decay 0.  \
--warmup_ratio 0.03  \
--lr_scheduler_type "cosine"  \
--logging_steps 1  \
--include_tokens_per_second  \
--packing False  \
--response_template "\n### Response:"  \
--dataset_text_field "output" 
```
