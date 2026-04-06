import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.utils import load_annotations_activitynetcaptions, load_annotations_charadessta, \
    load_annotations_tacos
from src.utils.vl_utils import word_tokenize, resample, Vocabulary
import torchtext
import torch.nn as nn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class GlanceDataset(Dataset):
    """ This class loads the annotations of the following datasets for NLVL task.
    * ActivityNet Captions
    * Charades-STA
    * TACoS

    Key attributes:
        - self.glance_duration
        - self.annotations
    """

    vocab1 = torchtext.vocab.pretrained_aliases["glove.840B.300d"]()
    vocab1.itos.extend(['<unk>'])
    vocab1.stoi['<unk>'] = vocab1.vectors.shape[0]
    vocab1.vectors = torch.cat([vocab1.vectors, torch.zeros(1, vocab1.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab1.vectors)

    def __init__(self, dataset_name, config, split):
        """
        Args:
            config: dict object from reading yaml
            split: "train"/"valid"/"test"
        """
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name

        self.split = split
        self.config = config

        self.examples = eval("load_annotations_{}".format(self.dataset_name))(self.split)

        self.vocab = None

    def build_vocab_and_encode_queries(self, existing_vocab=None):
        # 1. build vocab
        word_set = set()
        for i in range(len(self)):
            example = self.examples[i]
            example["words"] = word_tokenize(example["annotation"])
            word_set.update(example["words"])

        if existing_vocab is None:
            assert self.split == "train"
            vocab = Vocabulary(word_set)
        else:
            assert self.split == "valid" or self.split == "test"
            vocab = existing_vocab

        # 2. encode queries
        for i in range(len(self)):
            example = self.examples[i]
            example["query_label"] = torch.tensor(
                [vocab.wtoi.get(word, 1) for word in example["words"]], dtype=torch.int64  # 1: <UNK>
            )
            example["query_mask"] = torch.ones(len(example["words"]))
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Args:
            idx: int
        """
        res = dict(self.examples[idx])
        resampled_feature, ori_nframes = resample(
            torch.from_numpy(
                np.load(
                    os.path.join(self.config[self.dataset_name]["feature_dir"], "{}.npy".format(res["video_id"]))
                ).astype(np.float32)
            ), target_length=self.config[self.dataset_name]["video_feature_len"]
        )
        video_mask = torch.zeros(self.config[self.dataset_name]["video_feature_len"], dtype=torch.float32)
        video_mask[:ori_nframes] = 1.0  # ori_nframes may overshoot
        video_len = torch.sum(video_mask, dim=0).to(torch.long).item()

        start_frame = min(max(round(res["start_frac"] * video_len), 0), video_len - 1)
        end_frame = min(max(round(res["end_frac"] * video_len), start_frame), video_len - 1)
        glance_frame = min(max(round(res["glance_frac"] * video_len), 0), end_frame)

        word_idxs = torch.tensor([self.vocab1.stoi.get(word, 400000) for word in res["words"]],
                                 dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        # if self.split == "train":
            # print(self.config["noise_std"])
            # resampled_feature = resampled_feature + torch.randn_like(resampled_feature) * self.config["noise_std"]
            # resampled_feature = torch.clamp((resampled_feature + torch.randn_like(resampled_feature) * self.config["noise_std"]), min=0)


        res.update({
            "video": resampled_feature,#torch.Size([256, 500])
            "video_mask": video_mask,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "glance_frame": glance_frame,
            "word_vectors": word_vectors
        })
        # print("res")
        # print(torch.mean(resampled_feature))
        # print("std")
        # print(torch.std(resampled_feature))
        # print("min")
        # print(torch.min(resampled_feature))
        # print("max")
        # print(torch.max(resampled_feature))
        return res

    def collate_fn(self, examples):
        """ To be used in creating batches in DataLoader.
        Args:
            examples: [example]
        """
        batch = {k: [d[k] for d in examples] for k in examples[0].keys()}

        to_tensor = ["start_frac", "end_frac", "start_frame", "end_frame", "glance_frac", "glance_frame"]
        for k in to_tensor:
            batch[k] = torch.tensor(batch[k])

        # all videos have the same length
        to_stack = ["video", "video_mask"]
        for k in to_stack:
            batch[k] = torch.stack(batch[k], dim=0)

        # queries have different lengths
        to_pad = ["query_label", "query_mask", "word_vectors"]
        for k in to_pad:
            batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True)

        return batch


def get_dataloader(dataset, batch_size, dev):
    """ Get a dataloader from a dataloader.
    """
    shuffle = True if dataset.split == "train" else False
    drop_last = True if dataset.split == "train" else False
    dataloader = DataLoader(
        dataset if not dev else torch.utils.data.Subset(dataset, random.sample(range(len(dataset)), batch_size * 5)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=2,
        pin_memory=True,  # set True when loading data on CPU and training on GPU
        persistent_workers= True,
        prefetch_factor=5,
        collate_fn=dataset.collate_fn
    )
    return dataloader


def prepare_data(config, dataset_name):
    """ Prepare data.
    """
    train_ds = GlanceDataset(dataset_name, config, "train")
    train_ds.build_vocab_and_encode_queries()
    vocab = train_ds.vocab
    valid_ds = GlanceDataset(dataset_name, config, "valid")
    valid_ds.build_vocab_and_encode_queries(vocab)
    test_ds = GlanceDataset(dataset_name, config, "test")
    test_ds.build_vocab_and_encode_queries(vocab)

    dev = config["train"]["dev"]
    batch_size = config[dataset_name]["batch_size"]
    train_dl = get_dataloader(train_ds, batch_size, dev)
    valid_dl = get_dataloader(valid_ds, batch_size, dev)
    test_dl = get_dataloader(test_ds, 128, dev)  # fixed batch size for eval

    return {
        "train_dl": train_dl,
        "valid_dl": valid_dl,
        "test_dl": test_dl,
        "vocab": vocab
    }
