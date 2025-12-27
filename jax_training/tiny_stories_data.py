import datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
vocab_size = len(tokenizer)

np.random.seed(0)


def _add_border_tokens(batch):
    batch["input_ids"] = [list(ids) + [0] for ids in batch["input_ids"]]
    if "attention_mask" in batch:
        batch["attention_mask"] = [list(mask) + [1] for mask in batch["attention_mask"]]
    return batch


def load_split(split_name, n_samples):
    return (
        datasets.load_dataset(
            "roneneldan/TinyStories",
            streaming=True,
            split="train",
        )
        .take(n_samples)
        .map(
            lambda examples: tokenizer(examples["text"], return_tensors="np"),
            batched=True,
        )
        .map(_add_border_tokens, batched=True)
    )


def stream_batches(data_loader, context_window, batch_size):
    """Sammelt Token und gibt nur volle Batches zurück."""
    token_buffer = []
    segment_len = context_window + 1
    tokens_per_batch = batch_size * segment_len

    for batch_of_texts in data_loader.batch(batch_size=1024):  
        for text_ids in batch_of_texts["input_ids"]:
            token_buffer.extend(text_ids)

        while len(token_buffer) >= tokens_per_batch:
            chunk = token_buffer[:tokens_per_batch]
            token_buffer = token_buffer[tokens_per_batch:]  

            yield np.array(chunk, dtype=np.int32).reshape(batch_size, segment_len)


def text_to_batches(texts, context_window, batch_size):
    all_tokens = np.concatenate([np.array(t) for t in texts])

    segment_length = context_window + 1
    num_segments = len(all_tokens) // segment_length

    if num_segments < batch_size:
        return np.empty((0, batch_size, segment_length), dtype=np.int32)

    segments = all_tokens[: num_segments * segment_length].reshape(-1, segment_length)

    num_batches = num_segments // batch_size
    return segments[: num_batches * batch_size].reshape(
        num_batches, batch_size, segment_length
    )


class TinyStoriesDataset(IterableDataset):
    def __init__(self, hf_dataset, context_window, batch_size):
        self.hf_dataset = hf_dataset
        self.context_window = context_window
        self.batch_size = batch_size

    def __iter__(self):
        segment_len = self.context_window + 1
        tokens_per_batch = self.batch_size * segment_len
        token_buffer = []

        for batch_of_texts in self.hf_dataset.batch(batch_size=1024):
            for text_ids in batch_of_texts["input_ids"]:
                token_buffer.extend(text_ids)

            while len(token_buffer) >= tokens_per_batch:
                chunk = token_buffer[:tokens_per_batch]
                token_buffer = token_buffer[tokens_per_batch:]

                yield np.array(chunk, dtype=np.int32).reshape(
                    self.batch_size, segment_len
                )


def numpy_collate(batch):
    """Verhindert, dass PyTorch alles in Torch-Tensoren umwandelt."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    return np.array(batch)


if __name__ == "__main__":
    print(vocab_size)
