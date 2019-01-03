from torchtext import data, datasets
from torchtext.vocab import GloVe
import torch

def get_dataset(dataset, use_cuda, batch_size):
    TEXT = data.Field(sequential=True)#, lower=True)
    LABEL = data.Field(sequential=False)

    dataset = dataset.lower()

    if dataset == "sst-2":
        train, test, _ = datasets.SST.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral')
    elif dataset == "sst-5":
        train, test, _ = datasets.SST.splits(TEXT, LABEL, fine_grained=True)
    elif dataset == "snli":
        train, test, _ = datasets.SNLI.splits(TEXT, LABEL)

    TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=300))
    LABEL.build_vocab(train)
    
    # make iterator for splits
    # device gives a CUDA enabled device (-1 disables it)
    train_iter, test_iter, _ = data.BucketIterator.splits((train, test, _),
                                                        batch_size=batch_size,
                                                        device=0 if use_cuda else -1)

    return train_iter, test_iter, TEXT, LABEL

if __name__ == "__main__":
    train_iter, _, _, LABEL = get_dataset('SST-5', True)
    for label in LABEL.vocab.itos:
        print(label)