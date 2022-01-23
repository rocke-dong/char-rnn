import string
import unicodedata
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from pathlib import Path
from utilities.common_utils import get_logger
import logging

logger = get_logger(name=__name__, log_file=None, log_level=logging.DEBUG, log_level_name='')

all_letters = string.ascii_letters + " .,;'"


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


class Vocab(object):
    """
    docstring
    """
    def __init__(self, max_len:int, lower_case=True) -> None:
        self.max_len = max_len
        self.pad_token = '<pad>'
        self.pad_token_id = 0
        self.char2id = {self.pad_token: 0}
        self.id2char = {0: self.pad_token}
        self.lower_case = lower_case

    def __len__(self):
        return len(self.char2id)

    def string_to_ids(self, item_name):
        if self.lower_case:
            item_name = item_name.lower()
        ids = []
        for char in item_name:
            if char in self.char2id:
                ids.append(self.char2id[char])
            else:
                new_id = len(self.char2id)
                self.char2id[char] = new_id
                self.id2char[new_id] = char
                ids.append(new_id)
        if len(ids) > self.max_len:
            logger.info(f'item_name length {len(ids)} is larger than max_len {self.max_len}')
            ids = ids[: self.max_len]
        else:
            ids += [self.pad_token_id] * (self.max_len - len(ids))
        return ids


class NameDataset(Dataset):
    """
    docstring
    """
    def __init__(self, datapath, max_len=25, vocab=None) -> None:
        super().__init__()
        self.names = []
        self.labels = []
        self.label_map = {}

        self.vocab = Vocab(max_len=max_len) if vocab is None else vocab
        self.root_dir = datapath

        self.prepare_data()        

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return torch.tensor(self.names[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    def prepare_data(self):
        for file in Path(self.root_dir).glob('*.txt'):
            nation_name = file.stem
            label_id = len(self.label_map)
            self.label_map[nation_name] = label_id
            names = readLines(file)
            for name in names:
                if name:
                    name_ids = self.vocab.string_to_ids(name)
                    self.names.append(name_ids)
                    self.labels.append(label_id)


def get_dataloader(datapath):
    dataset = NameDataset(datapath)
    train_num = int(len(dataset) * 0.8)
    val_num = len(dataset)-train_num
    split_lengths = [train_num, val_num]
    train_set, val_set = random_split(dataset, split_lengths)
    logger.info(f'train_num: {train_num}; val_num: {val_num}')
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.vocab, dataset.label_map, train_num, val_num


if __name__ == "__main__":
    root_dir = Path('~/corpus_general/nlp_corpus/names_crnn/names')