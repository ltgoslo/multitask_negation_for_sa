import argparse
import pickle

from hierarchical_training import *
from hierarchical_model import *
from Utils.sst import *
from Utils.utils import *

def get_best_run(weightdir):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_acc = 0.0
    best_weights = ''
    for file in os.listdir(weightdir):
        epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
        lstm_dim = int(re.findall('[0-9]+', file.split('-')[-3])[0])
        lstm_layers = int(re.findall('[0-9]+', file.split('-')[-2])[0])
        acc = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
        if acc > best_acc:
            best_params = [epochs, lstm_dim, lstm_layers]
            best_acc = acc
            weights = os.path.join(weightdir, file)
            best_weights = weights
    return best_acc, best_params, best_weights

def test_model(aux_task, DATA_DIR, DATASET, num_runs=5, metric="acc",
               FINE_GRAINED="fine", AUXILIARY_DATASET="preprocessed/SFU/filtered_negation_scope.conll"):


    f1s = []
    accs = []
    preds = []
    ys = []

    aux_f1s = []
    aux_preds = []
    aux_ys = []

    print("opening model params...")
    with open(os.path.join("saved_models",
                           "{0}-{1}".format(DATASET, FINE_GRAINED),
                           aux_task,
                           "params.pkl"), "rb") as infile:
        params = pickle.load(infile)

    (w2idx,
     matrix_shape,
     tag_to_ix,
     len_labels,
     task2label2id) = params



    char2idx = {'UNK': 0, '<w>': 1, '</w>': 2, 'i': 3, 't': 4, 's': 5, ',': 6, 'h': 7, 'o': 8, 'w': 9, 'e': 10, 'v': 11, 'r': 12, 'u': 13, 'n': 14, 'f': 15, 'a': 16, 'l': 17, 'y': 18, 'm': 19, 'p': 20, 'b': 21, 'c': 22, 'd': 23, 'g': 24, '.': 25, 'k': 26, '-': 27, 'x': 28, "'": 29, '`': 30, '!': 31, 'j': 32, '?': 33, 'q': 34, 'z': 35, ';': 36, ':': 37}

    vocab = SetVocab(w2idx)

    #datadir = os.path.join(DATA_DIR, DATASET, FINE_GRAINED)
    datadir = os.path.join(DATA_DIR, "SST", FINE_GRAINED)
    sst = SSTDataset(vocab, False, datadir)
    maintask_test_iter = sst.get_split("test")

    chfile = "../data/challenge_dataset/sst-{0}.txt".format(args.FINE_GRAINED)
    challenge_dataset = ChallengeDataset(vocab, False, chfile)
    challenge_test = challenge_dataset.get_split()

    new_matrix = np.zeros(matrix_shape)



    if aux_task in ["negation_scope", "negation_scope_starsem"]:
        X, Y, org_X, org_Y, word2id, char2id, task2label2id =\
         get_conll_data(os.path.join(DATA_DIR, AUXILIARY_DATASET),
                        ["negation_scope"],
                        word2id=vocab,
                        char2id=char2idx,
                        task2label2id=task2label2id,
                        train=False)


        START_TAG = "<START>"
        STOP_TAG = "<STOP>"

        train_n = int(len(X) * .9)
        tag_to_ix = task2label2id["negation_scope"]

        print(tag_to_ix)

        X, char_X = zip(*X)

        auxiliary_testX = X[train_n:]
        auxiliary_testY = Y[train_n:]
        print(Y[0])

    print("finding best weights for runs 1 - {0}".format(num_runs))
    for i in range(num_runs):
        run = i + 1
        weight_dir = os.path.join("saved_models",
                                  "{0}-{1}".format(DATASET, FINE_GRAINED),
                                  aux_task,
                                  str(run))
        print("finding best weights from {0}".format(weight_dir))
        best_acc, (epochs, lstm_dim, lstm_layers), best_weights =\
                                                   get_best_run(weight_dir)

        model = Hierarchical_Model(vocab,
                                   new_matrix,
                                   tag_to_ix,
                                   len_labels,
                                   task2label2id,
                                   300,
                                   lstm_dim,
                                   1,
                                   train_embeddings=True)



        model.load_state_dict(torch.load(best_weights))
        model.eval()

        print("Run {0}".format(run))
        f1, acc, pred, y = model.eval_sent(maintask_test_iter, batch_size=50)
        print()

        f1s.append(f1)
        accs.append(acc)
        preds.append(pred)
        ys.append(y)

        if aux_task in ["negation_scope", "negation_scope_starsem"]:
            #print(auxiliary_testX[0])
            #print(auxiliary_testY[0])
            pred = model.eval_aux(auxiliary_testX, auxiliary_testY,
                            taskname="negation_scope", verbose=False)
            #print(pred[0])
            ys = [i["negation_scope"] for i in auxiliary_testY]
            #print(ys[0])
            f1 = average_scope_f1(ys, pred, tag_to_ix)
            print("AUX F1: {0:.3f}".format(f1))
            aux_f1s.append(f1)
            aux_preds.append(pred)
            aux_ys.append(ys)


        print("Eval on challenge data")
        chf1, chacc, chpred, chy = model.eval_sent(challenge_test, batch_size=1)
        print()

        # print challenge predictions to check
        prediction_dir = os.path.join("predictions", "SST-{0}".format(FINE_GRAINED), aux_task)
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, "run{0}_challenge_pred.txt".format(run)), "w") as out:
            for line in chpred:
                out.write("{0}\n".format(line))
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)


    print("#"*20 + "FINAL" + "#"*20)

    if metric == "f1":
        print("MEAN F1: {0:.3f}".format(mean_f1))
        print("STD F1: {0:.3f}".format(std_f1))

    if metric == "acc":
        print("MEAN ACC: {0:.2f} ({1:.1f})".format(mean_acc * 100, std_acc * 100))

    if aux_task not in ["none", "None"]:
        aux_mean_f1 = np.mean(aux_f1s)
        aux_std_f1 = np.std(aux_f1s)
        print("MEAN AUXILIARY F1: {0:.2f} ({1:.1f})".format(aux_mean_f1 * 100, aux_std_f1 * 100))

        return f1s, accs, preds, ys, aux_f1s, aux_preds, aux_ys, org_X

    else:
        return f1s, accs, preds, ys, None, None, None, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
    parser.add_argument("--NUM_RUNS", "-nr", default=5, type=int)
    parser.add_argument("--METRIC", "-m", default="acc")
    parser.add_argument("--DATA_DIR", "-dd",
                        default="../data/datasets/en")
    parser.add_argument("--DATASET", "-data",
                        default="SST")
    parser.add_argument("--AUX_DATA", "-auxdata",
                        default="preprocessed/starsem_negation/cde.conllu")
    parser.add_argument("--FINE_GRAINED", "-fg",
                        default="fine",
                        help="Either 'fine' or 'binary' (defaults to 'fine'.")
    args = parser.parse_args()

    f1s, accs, preds, ys, aux_f1s, aux_preds, aux_ys, org_X = \
                           test_model(args.AUXILIARY_TASK,
                                      args.DATA_DIR,
                                      args.DATASET,
                                      num_runs=args.NUM_RUNS,
                                      metric=args.METRIC,
                                      FINE_GRAINED=args.FINE_GRAINED,
                                      AUXILIARY_DATASET=args.AUX_DATA)

