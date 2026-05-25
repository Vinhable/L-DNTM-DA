import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse
import scipy.io
from . import file_utils


class _SequentialDataset(Dataset):
    def __init__(self, bow, times, time_wordfreq, doc_emb): 
        super().__init__()
        self.bow = bow
        self.times = times
        self.time_wordfreq = time_wordfreq
        self.doc_embedding = doc_emb

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, index):
        time_idx = self.times[index]
        return_dict = {
            'bow': self.bow[index],
            'times': time_idx,
            'time_wordfreq': self.time_wordfreq[time_idx],
            'doc_embedding': self.doc_embedding[index],
        }
        return return_dict


class DynamicDataset:
    def __init__(self, dataset_dir, batch_size=200, read_labels=False, device='cpu', as_tensor=True):
        self.load_data(dataset_dir, read_labels)
        
        self.vocab_size = len(self.vocab)
        self.train_size = len(self.train_bow)
        self.num_times = len(np.unique(self.train_times))
        self.train_time_wordfreq = self.get_time_wordfreq(self.train_bow, self.train_times)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        
        print('train size: ', len(self.train_bow))
        print('test size: ', len(self.test_bow))
        print('vocab size: ', len(self.vocab))
        print('average length: {:.3f}'.format(self.train_bow.sum(1).mean().item()))
        print('num of each time slice: ', self.num_times, np.bincount(self.train_times))

        if as_tensor:
            self.train_bow = torch.from_numpy(self.train_bow).float().to(device)
            self.test_bow = torch.from_numpy(self.test_bow).float().to(device)
            self.train_times = torch.from_numpy(self.train_times).long().to(device)
            self.test_times = torch.from_numpy(self.test_times).long().to(device)
            self.train_time_wordfreq = torch.from_numpy(self.train_time_wordfreq).float().to(device)
            self.train_doc_emb = torch.from_numpy(self.train_doc_emb).float().to(device)
            self.test_doc_emb = torch.from_numpy(self.test_doc_emb).float().to(device)

            self.train_dataset = _SequentialDataset(self.train_bow, self.train_times, self.train_time_wordfreq, self.train_doc_emb)
            self.test_dataset = _SequentialDataset(self.test_bow, self.test_times, self.train_time_wordfreq, self.test_doc_emb)

            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def load_data(self, path, read_labels):
        self.train_bow = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        self.word_embeddings = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

        self.train_times = np.loadtxt(f'{path}/train_times.txt').astype('int32')
        self.test_times = np.loadtxt(f'{path}/test_times.txt').astype('int32')

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')

        self.pretrained_WE = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        try:
            self.train_doc_emb = np.load(f'{path}/train_doc_emb.npy')
            self.test_doc_emb = np.load(f'{path}/test_doc_emb.npy')
            print("Successfully loaded pre-computed doc embeddings.")
        except FileNotFoundError:
            print("ERROR: Missing doc_emb.npy files. Run main.py to generate them.")
            exit() 
        
        if read_labels:
            self.train_labels = np.loadtxt(f'{path}/train_labels.txt').astype('int32')
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt').astype('int32')

    def get_time_wordfreq(self, bow, times):
        train_time_wordfreq = np.zeros((self.num_times, self.vocab_size))
        for time in range(self.num_times):
            idx = np.where(times == time)[0]
            train_time_wordfreq[time] += bow[idx].sum(0)
        cnt_times = np.bincount(times)
        train_time_wordfreq = train_time_wordfreq / cnt_times[:, np.newaxis]
        return train_time_wordfreq