import numpy as np
import torch
import torch.utils.data

from transformer import Constants as c
from tqdm import tqdm


def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)


def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [c.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != c.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_word2idx, tgt_word2idx, src_insts=None, tgt_insts=None):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))

        src_idx2word = {idx: word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts[idx], self._tgt_insts[idx]
        return self._src_insts[idx]


class CodeDocstringDataset(torch.utils.data.Dataset):
    def __init__(self, src_vocab, tgt_vocab, src_insts, tgt_insts, src_max_len=512, tgt_max_len=48):

        assert len(src_insts) == len(tgt_insts)

        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

        self._src_word2idx = dict()
        self._tgt_word2idx = dict()
        for idx, word in [(c.PAD, c.PAD_WORD), (c.UNK, c.UNK_WORD), (c.BOS, c.BOS_WORD), (c.EOS, c.EOS_WORD)]:
            self._src_word2idx[word] = idx
            self._tgt_word2idx[word] = idx

        src_len_init = len(self._src_word2idx)
        tgt_len_init = len(self._tgt_word2idx)

        for idx, word in enumerate(src_vocab):
            self._src_word2idx[word] = idx + src_len_init

        for idx, word in enumerate(tgt_vocab):
            self._tgt_word2idx[word] = idx + tgt_len_init

        self._src_idx2word = {idx: word for word, idx in self._src_word2idx.items()}
        self._tgt_idx2word = {idx: word for word, idx in self._tgt_word2idx.items()}

        self._src_insts = []
        self._tgt_insts = []
        for src, tgt in tqdm(zip(src_insts, tgt_insts), total=len(src_insts),
                             desc="[ Converting Words to Idx ]"):
            self._src_insts.append([self.src_find_word2idx(word) for word in src])
            self._tgt_insts.append([self.tgt_find_word2idx(word) for word in tgt])

    def src_find_word2idx(self, word):
        if word in self._src_word2idx:
            return self._src_word2idx[word]
        return c.UNK

    def tgt_find_word2idx(self, word):
        if word in self._tgt_word2idx:
            return self._tgt_word2idx[word]
        return c.UNK

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        src = self._src_insts[idx]
        tgt = self._tgt_insts[idx]

        len_src, len_tgt = len(src), len(tgt)

        if len_src > (self._src_max_len - 2):
            start = np.random.randint(0, len_src - (self._src_max_len - 2))
            src = src[start:start+self._src_max_len - 2]

        if len_tgt > (self._tgt_max_len - 2):
            start = np.random.randint(0, len_tgt - (self._tgt_max_len - 2))
            tgt = tgt[start:start+self._tgt_max_len - 2]

        src = [c.BOS] + src + [c.EOS]
        tgt = [c.BOS] + tgt + [c.EOS]

        return src, tgt


class CodeDocstringDatasetPreprocessed(torch.utils.data.Dataset):
    def __init__(self, src_word2idx, tgt_word2idx, src_insts, tgt_insts, src_max_len=512, tgt_max_len=48):

        assert len(src_insts) == len(tgt_insts)

        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

        for idx, word in [(c.PAD, c.PAD_WORD), (c.UNK, c.UNK_WORD), (c.BOS, c.BOS_WORD), (c.EOS, c.EOS_WORD)]:
            assert src_word2idx[word] == idx
            assert tgt_word2idx[word] == idx

        self._src_word2idx = src_word2idx
        self._tgt_word2idx = tgt_word2idx

        self._src_idx2word = {idx: word for word, idx in self._src_word2idx.items()}
        self._tgt_idx2word = {idx: word for word, idx in self._tgt_word2idx.items()}

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

    def src_find_word2idx(self, word):
        if word in self._src_word2idx:
            return self._src_word2idx[word]
        return c.UNK

    def tgt_find_word2idx(self, word):
        if word in self._tgt_word2idx:
            return self._tgt_word2idx[word]
        return c.UNK

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        src = self._src_insts[idx]
        tgt = self._tgt_insts[idx]

        len_src, len_tgt = len(src), len(tgt)

        if len_src > (self._src_max_len - 2):
            start = np.random.randint(0, len_src - (self._src_max_len - 2))
            src = src[start:start+self._src_max_len - 2]

        if len_tgt > (self._tgt_max_len - 2):
            start = np.random.randint(0, len_tgt - (self._tgt_max_len - 2))
            tgt = tgt[start:start+self._tgt_max_len - 2]

        src = [c.BOS] + src + [c.EOS]
        tgt = [c.BOS] + tgt + [c.EOS]

        return src, tgt
