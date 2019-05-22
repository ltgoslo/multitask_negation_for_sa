from scipy.stats import entropy, kurtosis
from collections import Counter
import numpy as np
import argparse

def import_conll_data(file, field):
    """
    field should be the column number for the relevant annotations
    """

    labels = []
    for line in open(file):
        if not line.startswith("#") and not line == "\n":
            anns = line.strip().split("\t")
            labels.append(anns[field])

    return labels
    
def get_label_distribution(labels):
    c = Counter(labels)
    print("Labels: ", end="")
    print(c)
    return np.array(list(c.values())) / sum(list(c.values()))


def analyze_labels(label_dist):
    e = entropy(label_dist)
    k = kurtosis(label_dist)
    print("Label entropy: {0:.3f}".format(e))
    print("Label kurtosis: {0:.1f}".format(k))
    return e, k


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-field", default=1, type=int)
    args = parser.parse_args()

    labels = import_conll_data(args.file, args.field)
    dist = get_label_distribution(labels)
    e, k = analyze_labels(dist)
