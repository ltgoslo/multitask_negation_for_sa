
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from Utils.WordVecs import WordVecs
from Utils.utils import *
import numpy as np


import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import defaultdict
from Utils.sst import SSTDataset
from torch.utils.data import DataLoader

import os
import argparse
import pickle

from hierarchical_model import *

class SetVocab(dict):
    def __init__(self, vocab):
        self.update(vocab)

    def ws2ids(self, ws):
        return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]

class Vocab(defaultdict):
    def __init__(self, train=True):
        super().__init__(lambda : len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def train_model(vocab,
                new_matrix,
                tag_to_ix,
                num_labels,
                task2label2id,
                embedding_dim,
                hidden_dim,
                num_lstm_layers,
                train_embeddings,
                auxiliary_trainX,
                auxiliary_trainY,
                auxiliary_testX,
                auxiliary_testY,
                maintask_loader,
                maintask_train_iter,
                maintask_dev_iter,
                AUXILIARY_TASK=None,
                epochs=10,
                sentiment_learning_rate=0.001,
                auxiliary_learning_rate=0.0001,
                BATCH_SIZE=50,
                number_of_runs=5,
                random_seeds=[123, 456, 789, 101112, 131415],
                DATASET="SST",
                FINE_GRAINED="fine"
                ):

    # Save the model parameters
    param_file = (dict(vocab.items()),
                  new_matrix.shape,
                  tag_to_ix,
                  num_labels,
                  task2label2id)

    basedir = os.path.join("saved_models",
                           "{0}-{1}".format(DATASET, FINE_GRAINED),
                           args.AUXILIARY_TASK)
    outfile = os.path.join(basedir,
                           "params.pkl")
    print("Saving model parameters to " + outfile)
    os.makedirs(basedir, exist_ok=True)

    with open(outfile, "wb") as out:
        pickle.dump(param_file, out)

    for i, run in enumerate(range(number_of_runs)):

        model = Hierarchical_Model(vocab,
                                   new_matrix,
                                   tag_to_ix,
                                   num_labels,
                                   task2label2id,
                                   embedding_dim,
                                   hidden_dim,
                                   1,
                                   train_embeddings=train_embeddings)

        # Set our optimizers
        sentiment_params = list(model.word_embeds.parameters()) + \
                           list(model.lstm1.parameters()) +\
                           list(model.lstm2.parameters()) +\
                           list(model.linear.parameters())

        auxiliary_params = list(model.word_embeds.parameters()) + \
                           list(model.lstm1.parameters()) +\
                           list(model.hidden2tag.parameters()) +\
                           [model.transitions]

        sentiment_optimizer = torch.optim.Adam(sentiment_params, lr=sentiment_learning_rate)
        auxiliary_optimizer = torch.optim.Adam(auxiliary_params, lr=auxiliary_learning_rate)

        print("RUN {0}".format(run + 1))
        best_dev_acc = 0.0

        # set random seed for reproducibility
        np.random.seed(random_seeds[i])
        torch.manual_seed(random_seeds[i])

        for j, epoch in enumerate(range(epochs)):

            # If AUXILIARY_TASK is None, defaults to single task
            if AUXILIARY_TASK not in ["None", "none", 0, None]:

                print("epoch {0}: ".format(epoch + 1), end="")
                for k in tqdm(range(len(auxiliary_trainX))):
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
                    model.zero_grad()

                    # Step 2. Get our inputs ready for the network, that is,
                    # turn them into Tensors of word indices.
                    sentence_in = torch.tensor(auxiliary_trainX[k])
                    targets = torch.tensor(auxiliary_trainY[k][AUXILIARY_TASK])

                    # Step 3. Run our forward pass.
                    loss = model.neg_log_likelihood(sentence_in, targets)

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    # calling optimizer.step()
                    loss.backward()
                    auxiliary_optimizer.step()

                #model.eval_aux(auxiliary_testX, auxiliary_testY,
                #               taskname=AUXILIARY_TASK)

            batch_losses = 0
            num_batches = 0
            model.train()

            print("epoch {0}".format(epoch + 1))

            for sents, targets in maintask_loader:
                model.zero_grad()

                loss = model.pooled_sentiment_loss(sents, targets)
                batch_losses += loss.data
                num_batches += 1

                loss.backward()
                sentiment_optimizer.step()

            print()
            print("loss: {0:.3f}".format(batch_losses / num_batches))
            model.eval()
            f1, acc, preds, ys = model.eval_sent(maintask_train_iter,
                                                 batch_size=BATCH_SIZE)
            f1, acc, preds, ys = model.eval_sent(maintask_dev_iter,
                                                 batch_size=BATCH_SIZE)

            if acc > best_dev_acc:
                best_dev_acc = acc
                print("NEW BEST DEV ACC: {0:.3f}".format(acc))


                basedir = os.path.join("saved_models", "{0}-{1}".format(DATASET, FINE_GRAINED),
                                       AUXILIARY_TASK,
                                       "{0}".format(run + 1))
                outname = "epochs:{0}-lstm_dim:{1}-lstm_layers:{2}-devacc:{3:.3f}".format(epoch + 1, model.lstm1.hidden_size, model.lstm1.num_layers, acc)
                modelfile = os.path.join(basedir,
                                         outname)
                os.makedirs(basedir, exist_ok=True)
                print("saving model to {0}".format(modelfile))
                torch.save(model.state_dict(), modelfile)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=300, type=int)
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_false")
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
    parser.add_argument("--EMBEDDINGS", "-emb",
                        default="../../embeddings/google.txt")
    parser.add_argument("--DATA_DIR", "-dd",
                        default="../data/datasets/en")
    parser.add_argument("--DATASET", "-data",
                        default="SST")
    parser.add_argument("--AUXILIARY_DATASET", "-auxdata",
                        default="preprocessed/starsem_negation/cdt.conllu")
    parser.add_argument("--SENTIMENT_LR", "-slr", default=0.001, type=float)
    parser.add_argument("--AUXILIARY_LR", "-alr", default=0.0001, type=float)
    parser.add_argument("--FINE_GRAINED", "-fg",
                        default="fine",
                        help="Either 'fine' or 'binary' (defaults to 'fine'.")

    args = parser.parse_args()
    print(args)

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"


    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    embeddings = WordVecs(args.EMBEDDINGS)
    print("loaded embeddings from {0}".format(args.EMBEDDINGS))
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by one to avoid overwriting the UNK token
    # at index 0
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 1
    vocab.update(with_unk)

    # Import datasets
    # This will update vocab with words not found in embeddings
    datadir = os.path.join(args.DATA_DIR, args.DATASET, args.FINE_GRAINED)
    sst = SSTDataset(vocab, False, datadir)

    maintask_train_iter = sst.get_split("train")
    maintask_dev_iter = sst.get_split("dev")
    maintask_test_iter = sst.get_split("test")

    maintask_loader = DataLoader(maintask_train_iter,
                                 batch_size=args.BATCH_SIZE,
                                 collate_fn=maintask_train_iter.collate_fn,
                                 shuffle=True)

    if args.AUXILIARY_TASK in ["speculation_scope"]:
        X, Y, org_X, org_Y, word2id, char2id, task2label2id =\
         get_conll_data(os.path.join(args.DATA_DIR, "preprocessed/SFU/filtered_speculation_scope.conll"),
                        ["speculation_scope"],
                        word2id=vocab)


    if args.AUXILIARY_TASK in ["negation_scope"]:
        X, Y, org_X, org_Y, word2id, char2id, task2label2id =\
         get_conll_data(os.path.join(args.DATA_DIR, args.AUXILIARY_DATASET),
                        ["negation_scope"],
                        word2id=vocab)


    if args.AUXILIARY_TASK in ["xpos", "upos", "multiword", "supersense"]:
        X, Y, org_X, org_Y, word2id, char2id, task2label2id =\
        get_conll_data(os.path.join(args.DATA_DIR, "preprocessed/streusle/train/streusle.ud_train.conllulex"),
                       ["xpos", "upos", "multiword", "supersense"],
                       word2id=vocab)


    if args.AUXILIARY_TASK not in ["None", "none", 0, None]:
        train_n = int(len(X) * .9)
        tag_to_ix = task2label2id[args.AUXILIARY_TASK]
        tag_to_ix[START_TAG] = len(tag_to_ix)
        tag_to_ix[STOP_TAG] = len(tag_to_ix)

        X, char_X = zip(*X)

        auxiliary_trainX = X[:train_n]
        auxiliary_trainY = Y[:train_n]
        auxiliary_testX = X[train_n:]
        auxiliary_testY = Y[train_n:]

    else:
        # Set all relevant auxiliary task parameters to None
        tag_to_ix = {"None": 0}
        tag_to_ix[START_TAG] = len(tag_to_ix)
        tag_to_ix[STOP_TAG] = len(tag_to_ix)
        task2label2id = None

        auxiliary_trainX = None
        auxiliary_trainY = None
        auxiliary_testX = None
        auxiliary_testY = None


    # Get new embedding matrix so that words not included in pretrained embeddings have a random embedding

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, 300))
    new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))


    train_model(vocab,
                new_matrix,
                tag_to_ix,
                len(sst.labels),
                task2label2id,
                args.EMBEDDING_DIM,
                args.HIDDEN_DIM,
                args.NUM_LAYERS,
                args.TRAIN_EMBEDDINGS,
                auxiliary_trainX,
                auxiliary_trainY,
                auxiliary_testX,
                auxiliary_testY,
                maintask_loader,
                maintask_train_iter,
                maintask_dev_iter,
                AUXILIARY_TASK=args.AUXILIARY_TASK,
                epochs=10,
                sentiment_learning_rate=args.SENTIMENT_LR,
                auxiliary_learning_rate=args.AUXILIARY_LR,
                BATCH_SIZE=50,
                number_of_runs=5,
                random_seeds=[123, 456, 789, 101112, 131415],
                DATASET=args.DATASET,
                FINE_GRAINED=args.FINE_GRAINED
                )

