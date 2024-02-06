from dataclasses import dataclass, field
from typing import List

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"])
    bias = "none"
    lora_dropout: float = 0.05


@dataclass
class PromptTuningConfig:
    prompt_tuning_init: str = "TEXT"
    num_virtual_tokens: int = 8
    prompt_tuning_init_text: str = "Classify if the tweet is a complaint or not:"
    tokenizer_name_or_path: str = "YOUR_GRANITE_MODEL_PATH"
    #tokenizer_name_or_path: str = "llama-7b-hf"
