import os
import re
from glob import glob
import itertools
from sklearn.metrics import f1_score

# special tokens and number regex
UNK = 'UNK'  # unk/OOV word/char
WORD_START = '<w>'  # word start
WORD_END = '</w>' # word end
NUM = 'NUM'  # number normalization string
NUMBERREGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")

def normalize(word):
    """Normalize a word by lower-casing it or replacing it if it is a number."""
    return NUM if NUMBERREGEX.match(word) else word.lower()

class ConllEntry:
    """Class representing an entry, i.e. word and its annotations in CoNLL
    """
    def __init__(self, form, tasks, xpos=None,
                 upos=None,
                 chunk=None,
                 multi_word = None,
                 supersense=None, negation_scope=None,
                 speculation_scope=None, sentiment=None):
        """
        Initializes a CoNLL entry.
        :param form: the word form
        :param tasks: the tasks for which this entry has annotations
        :param pos: the part-of-speech tag
        :param chunk: the chunk tag
        """

        self.form = form  # the word form
        self.tasks = tasks

        # normalize form (lower-cased and numbers replaced with NUM)
        self.norm = normalize(form)
        self.xpos = xpos  # language-specific POS
        self.upos = upos  # universal POS tags
        self.chunk = chunk
        self.multi_word = multi_word
        self.supersense = supersense
        self.negation_scope = negation_scope
        self.speculation_scope = speculation_scope
        self.sentiment = sentiment

def read_conll_file(file_path, tasks=None):

    with open(file_path, encoding="utf-8") as f:
        conll_entries = []
        for i, line in enumerate(f):
            if i == 0:
                """ Get the mapping of columns to annotation labels
                    This assumes the first line of the corpus starts
                    with a '#' and is followed by tab separated labels
                    for each column.
                """

                labels2column = {}
                for label in line[1:].strip().split("\t"):
                    labels2column[label] = len(labels2column)

            if line == "\n" or line.startswith("#"):
                if len(conll_entries) > 1:
                    yield conll_entries
                conll_entries = []
            else:
                if "negation_scope" in tasks:

                    token, neg = line.strip().split("\t")
                    conll_entries.append(ConllEntry(token, tasks,
                                                    negation_scope=neg))

                if "speculation_scope" in tasks:
                    token, neg = line.strip().split("\t")
                    conll_entries.append(ConllEntry(token, tasks,
                                                    speculation_scope=neg))


                if "upos" in tasks:
                    anns = line.strip().split("\t")
                    token = anns[1]
                    upos = anns[3]
                    xpos = anns[4]
                    mw = anns[10]
                    supersense = anns[13]
                    conll_entries.append(ConllEntry(token, tasks,
                                                    upos=upos,
                                                    xpos=xpos,
                                                    multi_word=mw,
                                                    supersense=supersense))
        if len(conll_entries) > 1:
            yield conll_entries


def get_conll_data(data_file, task_names, word2id=None, char2id=None,
             task2label2id=None, data_dir=None, train=True, verbose=False):
    """
    :param domains: a list of domains from which to obtain the data
    :param task_names: a list of task names
    :param word2id: a mapping of words to their ids
    :param char2id: a mapping of characters to their ids
    :param task2label2id: a mapping of tasks to a label-to-id dictionary
    :param data_dir: the directory containing the data
    :param train: whether data is used for training (default: True)
    :param verbose: whether to print more information re file reading
    :return X: a list of tuples containing a list of word indices and a list of
               a list of character indices;
            Y: a list of dictionaries mapping a task to a list of label indices;
            org_X: the original words; a list of lists of normalized word forms;
            org_Y: a list of dictionaries mapping a task to a list of labels;
            word2id: a word-to-id mapping;
            char2id: a character-to-id mapping;
            task2label2id: a dictionary mapping a task to a label-to-id mapping.
    """
    X = []
    Y = []
    org_X = []
    org_Y = []

    # for training, we initialize all mappings; for testing, we require mappings
    if train:
        #assert word2id is None, ('Error: Word-to-id mapping should not be '
        #                         'provided for training.')
        #assert char2id is None, ('Error: Character-to-id mapping should not '
        #                         'be provided for training.')

        # create word-to-id, character-to-id, and task-to-label-to-id mappings
        if word2id is None:
            word2id = {}

            # set the indices of the special characters
            word2id[UNK] = 0  # unk word / OOV

        if char2id is None:
            char2id = {}
            char2id[UNK] = 0  # unk char
            char2id[WORD_START] = 1  # word start
            char2id[WORD_END] = 2  # word end index

        if task2label2id is None:
            task2label2id = {task: {} for task in task_names}

        # manually add tags only available in some domains for POS tagging
        if "pos" in task_names:
            for label in ['NFP', 'ADD', '$', '', 'CODE', 'X', 'VERB']:
                task2label2id["pos"][label] = len(task2label2id["pos"])
    else:
        assert word2id is not None, 'Error: Word-to-id mapping is required.'
        assert char2id is not None, 'Error: Char-to-id mapping is required.'
        assert task2label2id is not None, 'Error: Task mapping is required.'
        assert UNK in word2id
        assert UNK in char2id
        assert WORD_START in char2id
        assert WORD_END in char2id


    num_sentences = 0
    num_tokens = 0

    file_reader = read_conll_file(data_file, task_names)


    # the file reader should returns a list of CoNLL entries; we then get
    # the relevant labels for each task
    for sentence_idx, conll_entries in enumerate(file_reader):
        num_sentences += 1
        sentence_word_indices = []  # sequence of word indices
        sentence_char_indices = []  # sequence of char indices
        # keep track of the label indices and labels for each task
        sentence_task2label_indices = {}
        sentence_task2labels = {}

        # keep track of the original word forms
        org_X.append([conll_entry.norm for conll_entry in conll_entries])

        for i, conll_entry in enumerate(conll_entries):
            num_tokens += 1
            word = conll_entry.norm

            # add words and chars to the mapping
            if train and word not in word2id:
                word2id[word] = len(word2id)
            sentence_word_indices.append(word2id.get(word, word2id[UNK]))

            chars_of_word = [char2id[WORD_START]]
            for char in word:
                if train and char not in char2id:
                    char2id[char] = len(char2id)
                chars_of_word.append(char2id.get(char, char2id[UNK]))
            chars_of_word.append(char2id[WORD_END])
            sentence_char_indices.append(chars_of_word)

            # get the labels for the task if we have annotations
            for task in task2label2id.keys():
                if task in conll_entry.tasks:
                    if task == "upos":
                        label = conll_entry.upos
                    elif task == "xpos":
                        label = conll_entry.xpos
                    elif task == "chunk":
                        label = conll_entry.chunk
                    elif task == "multi_word":
                        label = conll_entry.multi_word
                    elif task == "supersense":
                        label = conll_entry.supersense
                    elif task == "negation_scope":
                        label = conll_entry.negation_scope
                    elif task == "speculation_scope":
                        label = conll_entry.speculation_scope
                    elif task == "sentiment":
                        label = conll_entry.sentiment

                    if task not in sentence_task2label_indices:
                        sentence_task2label_indices[task] = []
                    if task not in sentence_task2labels:
                        sentence_task2labels[task] = []
                    assert label is not None, ('Label is None for task '
                                               '%s.' % task)
                    if not train and label not in task2label2id[task]:
                        print('Error: Unknown label %s for task %s not '
                              'valid during testing.' % (label, task))
                        print('Assigning id of another label as we only '
                              'care about main task scores...')
                        task2label2id[task][label] =\
                            len(task2label2id[task]) - 1
                    if train and label not in task2label2id[task]:
                        task2label2id[task][label] = \
                            len(task2label2id[task])
                    sentence_task2label_indices[task].\
                        append(task2label2id[task].get(label))
                    sentence_task2labels[task].append(label)

        if len(task_names) == 1:
            if len(sentence_task2label_indices) == 0:
                continue
        assert len(sentence_task2label_indices) > 0,\
            'Error: No label/task available for entry.'
        X.append((sentence_word_indices, sentence_char_indices))
        Y.append(sentence_task2label_indices)
        org_Y.append(sentence_task2labels)

    assert num_sentences != 0 and num_tokens != 0, ('No data read')
    print('Number of sentences: %d. Number of tokens: %d.'
          % (num_sentences, num_tokens))
    print("%s sentences %s tokens" % (num_sentences, num_tokens))
    print("%s w features, %s c features " % (len(word2id), len(char2id)))

    for task, label2id in task2label2id.items():
        print('Task %s. Labels: %s' % (task, [l for l in label2id.keys()]))
        print()

    assert len(X) == len(Y)
    return X, Y, org_X, org_Y, word2id, char2id, task2label2id



def print_args(args):
    for arg in vars(args):
        print('{0}:\t{1}'.format(arg, getattr(args, arg)))
    print()


def print_prediction(prediction, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        for i in prediction:
            f.write("{0}\n".format(i))

def print_results(args, dev_f1, test_f1,
                  clf='aBLSE'):
    outfile = os.path.join('results', args.src_dataset, '{0}-{1}'.format(args.src_lang, args.trg_lang),
                            '{0}-binary:{1}.txt'.format(
                                clf, args.binary))
    if clf in ['sentBLSE', 'aBLSE', 'aBLSE_target', 'aBLSE_weighted']:
        header = "Epochs\tLR\tWD\tBS\talpha\tDev F1\tTest F1\n"
        body = "{0}\t{1}\t{2}\t{3}\t{4}\t{5:0.3f}\t{6:0.3f}\n".format(
                args.epochs, args.learning_rate, args.weight_decay,
                args.batch_size, args.alpha, dev_f1, test_f1)
    else:
        header = "Epochs\tLR\tWD\tBS\tDev F1\tTest F1\n"
        body = "{0}\t{1}\t{2}\t{3}\t{4:0.3f}\t{5:0.3f}\n".format(
                args.epochs, args.learning_rate, args.weight_decay,
                args.batch_size, dev_f1, test_f1)


    if not os.path.exists(outfile):
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'w') as f:
            f.write(header)

    with open(outfile, "a") as f:
        f.write(body)



def to_array(X, n=2):
    """
    Converts a list scalars to an array of size len(X) x n
    >>> to_array([0,1], n=2)
    >>> array([[ 1.,  0.],
               [ 0.,  1.]])
    """
    return np.array([np.eye(n)[x] for x in X])

def per_class_f1(y, pred):
    """
    Returns the per class f1 score.
    Todo: make this cleaner.
    """

    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)

    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        f1 = f1_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)

def per_class_prec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        f1 = precision_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)

def per_class_rec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        f1 = recall_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)


def only_scopes(x, tag2idx):
    idx2tag = dict([(i,w) for w, i in tag2idx.items()])
    return [1 if idx2tag[i] in ["B_neg", "I_neg", "B_scope", "I_scope"] else 0 for i in x]

def only_cues(x, tag2idx):
    idx2tag = dict([(i,w) for w, i in tag2idx.items()])
    return [1 if idx2tag[i] in ["B_negcue", "I_negcue", "B_cue", "I_cue"] else 0 for i in x]

def tokenlevel_scope_f1(gold, pred, tag2idx):
    gold = only_scopes(gold, tag2idx)
    pred = only_scopes(pred, tag2idx)
    return f1_score(gold, pred)

def tokenlevel_cue_f1(gold, pred, tag2idx):
    gold = only_cues(gold, tag2idx)
    pred = only_cues(pred, tag2idx)
    return f1_score(gold, pred)

def average_scope_f1(Y, preds, tag2idx):
    f1 = 0
    for y, p in zip(Y, preds):
        f1 += tokenlevel_scope_f1(y, p, tag2idx)
    return f1 / len(Y)

def average_cue_f1(Y, preds, tag2idx):
    f1 = 0
    for y, p in zip(Y, preds):
        f1 += tokenlevel_cue_f1(y, p, tag2idx)
    return f1 / len(Y)

def str2bool(v):
    # Converts a string to a boolean, for parsing command line arguments
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    X, Y, org_X, org_Y, word2id, char2id, task2label2id = get_conll_data("../../data/datasets/en/preprocessed/SFU/filtered_negation_scope.conll",
                                                                   ["negation_scope", "speculation_scope"])

