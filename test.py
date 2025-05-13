from typing import Dict, Any, Optional, Sequence, Union, List, Callable
import numpy as np
from streaming import Stream, StreamingDataset
from omegaconf import OmegaConf
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer, 
    PreTrainedTokenizerFast
)
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as torch_dist
from datasets import load_dataset
from composer.callbacks import SpeedMonitor
from composer.models import HuggingFaceModel
from composer.utils import dist
from composer import Trainer

from main import build_dataloader

from dotenv import load_dotenv

load_dotenv()

class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(self, base_collator: Callable, eos_token_id: Optional[int] = None, bos_token_id: Optional[int] = None):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError("Must supply a value for either eos_token_id or bos_token_id, but got None for both.")
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                "Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. "
                + "Please supply `eos_token_id` if sequences end with an EOS token, or use "
                + "`bos_token_id` if sequences start with a BOS token."
            )
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True
        else:
            self.split_token_id = eos_token_id
            self.bos_mode = False

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.split_token_id is not None
        is_separator = torch.eq(batch["input_ids"], self.split_token_id)
        cumulative_sep = torch.cumsum(is_separator, dim=1).to(batch["input_ids"].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)



class StreamingTextDataset(StreamingDataset):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_len: int,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[int] = None,
        predownload: int = 100_000,
        partition_algo: str = "orig",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1s",
        shuffle_seed: int = 9176,
        cache_limit: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            cache_limit=cache_limit,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError("If tokenizing on-the-fly, tokenizer must have a pad_token_id")

        encoded = self.tokenizer(text_sample["text"], 
                                 truncation=True, 
                                 padding="max_length", 
                                 max_length=self.max_seq_len, 
                                #  return_tensors="pt",
                                 )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded.get("token_type_ids", [0] * self.max_seq_len),
        }

    def _read_binary_tokenized_sample(self, sample):
        seq_len = sample["len"] if "len" in sample else len(sample["input_ids"])

        input_ids = np.frombuffer(sample["input_ids"], dtype=np.int64).copy()
        if "attention_mask" in sample:
            attention_mask = np.frombuffer(sample["attention_mask"], dtype=np.int64).copy()
        else:
            attention_mask = np.ones_like(input_ids)

        # calculate padding
        pad_len = self.max_seq_len - seq_len

        # pad or truncate input_ids and attention_mask
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), constant_values=self.tokenizer.pad_token_id)
            attention_mask = np.pad(attention_mask, (0, pad_len), constant_values=0)
        elif pad_len < 0:
            input_ids = input_ids[: self.max_seq_len]
            attention_mask = attention_mask[: self.max_seq_len]

        token_type_ids = np.zeros(self.max_seq_len, dtype=np.int64)

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "token_type_ids": token_type_ids.tolist(),
            # "source": sample["source"],
        }

    # How to process a sample
    def __getitem__(self, idx: int) -> Union[Dict[str, Any], torch.Tensor]:
        sample = super().__getitem__(idx)
        if "input_ids" in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        elif "text" in sample:
            token_sample = self._tokenize(sample)
        else:
            raise RuntimeError("StreamingTextDataset needs samples to have a `text` or `input_ids` column")
        return token_sample


def get_dataloader(tokenizer, 
                   batch_size, 
                   collate_fn,
                   max_seq_len,
                   shuffle=False,
                   num_workers=8,
                   pin_memory=True,
                   predownload=100_000):
    # Set up the dataset
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        local="/home/nlp/achimoa/workspace/hebrew_text_retrieval/data/hebrew_modernbert/v20250428",         # Path to your local MDS directory
        remote=None,                 # No remote; purely local
        split="train",               # MDS split name
        shuffle=shuffle,                # Enable shuffling
        shuffle_algo="py1s",         # Optional: fast shuffling algorithm
        batch_size=batch_size,                # Optional: to control shuffle buffering
        predownload=predownload,            # How many samples to prefetch into the shuffle buffer
    )
    # dataset = load_dataset("imdb", split="train")

    def tokenize_fn(example):
        if "text" not in example or not example["text"]:
            return {}  # drop invalid rows
    
        encodings = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,             
            return_attention_mask=True,
            # return_tensors="pt"  # ← Important: keep as lists
        )
        return encodings
        # return {
        #     "input_ids": encodings["input_ids"][0],  # extract single sample
        #     "attention_mask": encodings["attention_mask"][0],
        #     "token_type_ids": encodings.get("token_type_ids", torch.zeros(max_seq_len, dtype=torch.long))
        # }


    # Apply tokenization
    # dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # if torch_dist.is_available() and not torch_dist.is_initialized():
    #     torch_dist.init_process_group(backend="nccl")

    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=torch_dist.get_world_size(),
    #     rank=torch_dist.get_rank(),
    #     # shuffle=True,
    #     drop_last=True,
    # )

    collate_fn = DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer, mlm=cfg.mlm_probability is not None, mlm_probability=cfg.mlm_probability
    )

    # Wrap in a DataLoader
    dataloader = DataLoader(dataset, 
                            sampler=None,
                            batch_size=batch_size, 
                            collate_fn=collate_fn, 
                            num_workers=num_workers, 
                            pin_memory=pin_memory,
                            prefetch_factor=cfg.get("prefetch_factor", 2),
                            persistent_workers=cfg.get("persistent_workers", True),
                            timeout=cfg.get("timeout", 0),)
    return dataloader



# Load YAML config
cfg = OmegaConf.load("yamls/main/base/test.yaml")

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer.name)
tokenizer.model_max_length = cfg.tokenizer.max_seq_length

# Build BERT config from scratch
bert_config = BertConfig(
    vocab_size=cfg.model.vocab_size,
    hidden_size=cfg.model.hidden_size,
    num_hidden_layers=cfg.model.num_hidden_layers,
    num_attention_heads=cfg.model.num_attention_heads,
    intermediate_size=cfg.model.intermediate_size,
    max_position_embeddings=cfg.model.max_position_embeddings,
    type_vocab_size=cfg.model.type_vocab_size,
    hidden_act=cfg.model.hidden_act,
    hidden_dropout_prob=cfg.model.hidden_dropout_prob,
    attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
    initializer_range=cfg.model.initializer_range,
    layer_norm_eps=cfg.model.layer_norm_eps,
)

# Initialize model from scratch
model = BertForMaskedLM(config=bert_config)


# MLM Collator
hf_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=cfg.train_loader.dataset.mlm_probability,
)

def wrapped_collate_fn(features):
    cleaned = []
    for f in features:
        # If f is a tuple, extract the data dict (assume f[0])
        if isinstance(f, tuple):
            f = f[0]
        cleaned.append({
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
            "token_type_ids": f.get("token_type_ids", [0] * len(f["input_ids"])),
        })
    return hf_collator(cleaned)

# Dataloader
train_dataloader = get_dataloader(
    tokenizer=tokenizer,
    batch_size=cfg.global_train_batch_size,
    max_seq_len=cfg.tokenizer.max_seq_length,
    collate_fn=wrapped_collate_fn,
    shuffle=cfg.train_loader.dataset.shuffle,
    predownload=cfg.train_loader.dataset.predownload,
)

# Composer model wrapper
composer_model = HuggingFaceModel(model=model)

# Trainer
trainer = Trainer(
    model=composer_model,
    train_dataloader=train_dataloader,
    max_duration=cfg.trainer.max_duration,
    precision=cfg.trainer.precision,
    device_train_microbatch_size=cfg.device_train_microbatch_size,
    # save_folder=cfg.trainer.save_folder,
    callbacks=[SpeedMonitor(window_size=50)]
)

# Train!
trainer.fit()
