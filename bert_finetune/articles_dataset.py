from torch.utils.data import Dataset
from torch import tensor, randint
import polars as pl
import tiktoken


class ArticlesDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        context_window_size=100,
    ):
        self.df = pl.read_csv(file_path, encoding="utf8")

        self.context_window_size = context_window_size

        self.tokenizer = tiktoken.get_encoding("o200k_base")
        # self.tokenizer = tokenizer.Tokenizer()
        # if train_tokenizer_add_tokens is not None:
        #     self.tokenizer.fit(self.all_articles_merged(), train_tokenizer_add_tokens)

        self.labels = [
            "Etat",
            "Inland",
            "International",
            "Kultur",
            "Panorama",
            "Sport",
            "Web",
            "Wirtschaft",
            "Wissenschaft",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return (
            tensor(self.sentence_to_indexes(self.df["Article"][index])),
            tensor(self.label2index(self.df["Segment"][index])),
        )

    # def word_to_index(self, word: str):
    #     self.word2index[word]

    # def index_to_word(self, index: int):
    #     return self.index2word[index]

    def label2index(self, label: str):
        return self.labels.index(label)

    def index2label(self, index: int):
        return self.labels[index]

    def sentence_to_indexes(self, sentence: str):
        # nltk_sentence = nltk.word_tokenize(sentence, language="german")
        # indexes = []
        # for w in nltk_sentence:
        #     indexes.append(self.word2index[self.c.stem(w)])

        # Right-pad instead of left-pad
        # indexes = self.tokenizer.tokenize(sentence)
        indexes = self.tokenizer.encode(sentence)

        token_len = len(indexes)
        if token_len > self.context_window_size:
            start_index = randint(0, token_len - self.context_window_size, (1,)).item()
            return indexes[start_index : start_index + self.context_window_size]

        while len(indexes) < self.context_window_size:
            indexes.append(0)  # Pad at the end

        return indexes

    def all_articles_merged(self):
        return " ".join(self.df["Article"])

    def max_context_size(self):
        context_size = 0
        for s in self.df["Article"]:
            t = self.tokenizer.encode(s)
            if len(t) > context_size:
                context_size = len(t)

        return context_size
