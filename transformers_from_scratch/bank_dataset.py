from torch.utils.data import Dataset
from torch import tensor
import polars as pl
import nltk


class BankDataset(Dataset):
    def __init__(self, file_path: str):
        self.df = pl.read_csv(file_path)
        words = " ".join(self.df["Satz"])
        self.c = nltk.Cistem(True)

        self.index2word = ["<unk>"]
        self.context_window_size = 0
        for w in nltk.word_tokenize(words, language="german"):
            self.index2word.append(self.c.stem(w))
            if len(w) > self.context_window_size:
                self.context_window_size = len(w)

        self.index2word = set(self.index2word)
        self.unique_word_count = len(self.index2word)

        self.word2index = {}

        self.label2index = {"Flussbank": 0, "Geldbank": 1, "Sitzbank": 2}

        for i, e in enumerate(self.index2word):
            self.word2index[e] = i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return (
            tensor(self.sentence_to_indexes(self.df["Satz"][index])),
            tensor(self.label2index[self.df["Bedeutung"][index]]),
        )

    def word_to_index(self, word: str):
        self.word2index[word]

    def index_to_word(self, index: int):
        self.index2word[index]

    def index2label(self, index: int):
        return ["Flussbank", "Geldbank", "Sitzbank"][index]

    def sentence_to_indexes(self, sentence: str):
        nltk_sentence = nltk.word_tokenize(sentence, language="german")
        indexes = [0 for _ in range(self.context_window_size - len(nltk_sentence))]
        for w in nltk_sentence:
            indexes.append(self.word2index[self.c.stem(w)])

        return indexes
