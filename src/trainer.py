import torch
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import has_length


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(length != 0 for length in lengths), "Samples should not have zero length."
    if all(length > 0 for length in lengths) or all(length < 0 for length in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    mm_pairs = [(i, length) for i, length in enumerate(lengths) if length > 0]
    lang_pairs = [(i, -length) for i, length in enumerate(lengths) if length < 0]
    mm_indices, mm_lengths = zip(*mm_pairs)
    lang_indices, lang_lengths = zip(*lang_pairs)

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    additional_batch = mm_megabatches[-1] + lang_megabatches[-1]
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    order = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in order]
    if additional_batch:
        megabatches.append(sorted(additional_batch))
    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    def __init__(self, batch_size, world_size, lengths, group_by_modality=False, generator=None):
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.group_by_modality = group_by_modality
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        return iter(indices)


class QwenVLTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        if getattr(self.args, "group_by_modality_length", False):
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=train_dataset.modality_lengths,
                group_by_modality=True,
            )
        if train_dataset is self.train_dataset:
            return super()._get_train_sampler()
        return super()._get_train_sampler(train_dataset)
