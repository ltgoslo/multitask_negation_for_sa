
Skip to content
Enterprise

    Pull requests
    Issues
    Explore

    @jeremycb

2
2

    0

in5550/2018 Private
Code
Issues 0
Pull requests 0
Projects 0
Wiki
Insights
2018/obligatories/4/tagger.py
8c1d8fc on Nov 23, 2018
@oe oe updates from the group session yesterday
380 lines (345 sloc) 14 KB
#!/usr/bin/env python3

# -*- coding: utf-8; -*-

import argparse;
import copy;
from keras import optimizers;
from keras.callbacks import TensorBoard, EarlyStopping;
from keras.layers \
  import LSTM, Embedding, Dense, concatenate, Dropout, Bidirectional;
from keras.models import Model, Input;
from keras.utils import plot_model;
import numpy as np;
import os;
from pathlib import Path;
import pickle;
from sklearn.metrics import classification_report;
import sys;
import tensorflow as tf;

from convert \
  import read_negations, distribute_negation_instances, write_negations;
from score import starsem_score;


__author__ = "oe"
__version__ = "2018"

np.random.seed(42);
tf.set_random_seed(42);

LENGTH = 100;

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description = "Neural Negation Baseline");
  parser.add_argument('--cores', type = int,
                      default = os.environ.get("SLURM_CPUS_ON_NODE", 4));
  parser.add_argument("--id");
  parser.add_argument("--log", nargs = "?", const = True);
  parser.add_argument("--train");
  parser.add_argument("--filter", action = "store_true");
  parser.add_argument("--cues", action = "store_true");
  parser.add_argument("--test");
  parser.add_argument("--output");
  parser.add_argument("--model");
  parser.add_argument("--mask", action = "store_false");
  parser.add_argument("-wd", type = int, default = 200);
  parser.add_argument("-cd", type = int, default = 50);
  parser.add_argument("-do", type = float, default = 0.1);
  parser.add_argument("-rd", type = float, default = 0.1);
  parser.add_argument("-ru", type = int, default = 200);
  parser.add_argument("-lr", type = float, default = 0.001);
  parser.add_argument("-vs", type = float, default = 0.1);
  parser.add_argument("-bs", type = int, default = 16);
  parser.add_argument("-e", dest = "epochs", type = int, default = 20);
  parser.add_argument("-d", dest = "debug", action = "count");
  parser.add_argument("--score", nargs = "?", const = True);
  arguments = parser.parse_args();

  if not arguments.id:
    arguments.id \
      = ("wd={},cd={},do={},rd={},ru={},lr={},vs={},bs={},e={}"
         "".format(arguments.wd, arguments.cd,
                   arguments.do, arguments.rd, arguments.ru,
                   arguments.lr, arguments.vs, arguments.bs,
                   arguments.epochs));

  if arguments.score:
    if arguments.score == True:
      arguments.score = arguments.id + ".score";
    #
    # we hope for the touch() method to be realized (at least on modern
    # kernels) in a pseudo-atomic manner, such that only one of potentially a
    # large number of concurrent processes will be abel to create a new file.
    #
    try:
      Path(arguments.score).touch(exist_ok = False);
    except FileExistsError:
      print("skipping: {}.".format(arguments.id));
      exit(1);

  if arguments.log:
    sys.stdout = sys.stderr = open(arguments.id + ".log", "w");

  #
  # try to make sure TensorFlow respects our resource allocations
  #
  config = tf.ConfigProto(inter_op_parallelism_threads=arguments.cores,
                          intra_op_parallelism_threads=arguments.cores);
  tf.keras.backend.set_session(tf.Session(config = config));

  train = read_negations(arguments.train);
  n = len(train);
  train = distribute_negation_instances(train, filter = arguments.filter);
  print("{} training sentences distributed into {} sequences."
        "".format(n, len(train)));

  test = read_negations(arguments.test);
  n = len(test);
  test = distribute_negation_instances(test, filter = arguments.filter);
  print("{} evaluation sentences distributed into {} sequences."
        "".format(n, len(test)));

  vocabulary = {"__padding__": 0,
                "__cue__": 1, "__affix__": 2,
                "__?__": 3 };
  names = ["__padding__", "__cue__", "__affix__", "__?__"];
  prefixes = {};
  suffixes = {};

  inputs = np.zeros((len(train), LENGTH), dtype = int);
  cues = np.zeros((len(train), LENGTH), dtype = int);
  #
  # to get started, distinguish (full-token) cues from (sub-token) affixes,
  # in- and out-of-scope (aka ‘true’) tokens, and (of course) padding:
  #
  # when using gold-standard cues, however, we only predict the first three
  # classes: for non-padding inputs the classification problem is binary.
  #
  classes = {"P": 0, "T": 1, "F": 2, "C": 3, "A": 4};
  outputs = np.zeros((len(train), LENGTH,
                      len(classes) - (2 if arguments.cues else 0)),
                     dtype = int);
  n = 0;
  for i, sentence in enumerate(train):
    n += len(sentence["nodes"]);
    for j, node in enumerate(sentence["nodes"]):
      form = node["form"];
      if not form in vocabulary:
        vocabulary[form] = len(vocabulary);
        names.append(form);
      inputs[i, j] = vocabulary[form];
      negation = node.get("negation");
      negation = (negation[0] if negation else None);
      if negation and ("cue" in negation or "scope" in negation):
        if negation.get("cue") == form:
          cues[i, j] = 2; # a word cue
          if arguments.cues:
            outputs[i, j, classes["T"]] = 1;
          else:
            outputs[i, j, classes["C"]] = 1;
        elif "cue" in negation:
          cue = negation["cue"];
          cues[i, j] = 3; # an affix cue
          if form.startswith(cue):
            prefixes[cue] = prefixes.get(cue, 0) + 1;
          elif form.endswith(cue):
            suffixes[cue] = suffixes.get(cue, 0) + 1;
          if arguments.cues:
            outputs[i, j, classes["T"]] = 1;
          else:
            outputs[i, j, classes["A"]] = 1;
        else:
          cues[i, j] = 1; # not a cue
          outputs[i, j, classes["F"]] = 1;

      else:
        cues[i, j] = 1; # not a cue
        outputs[i, j, classes["T"]] = 1;

    for j in range(j + 1, LENGTH):
      #
      # _fix_me_
      # what would happen if we were to use an all-zero class vector to pad?
      #
      outputs[i, j, classes["P"]] = 1;

  if arguments.debug:
    for i, sentence in enumerate(train):
      labels = np.argmax(outputs[i], axis = 1);
      for j, (node, cue, label) in enumerate(zip(sentence["nodes"],
                                                 cues[i], labels)):
        print("{}\t{}\t{}\t{}".format(names[int(inputs[i, j])],
                                      cue, label, outputs[i, j]));
      print();

  if arguments.debug:
    print("training: {} word tokens; {} types; {} classes."
          "".format(n, len(vocabulary),
                    len(classes) - (2 if arguments.cues else 0)));
    for prefix in prefixes:
      print("prefix: ‘{}’ {}".format(prefix, prefixes[prefix]));
    for suffix in suffixes:
      print("suffix: ‘{}’ {}".format(suffix, suffixes[suffix]));

  with open(arguments.id + ".pyc", mode = "w+b") as stream:
    pickle.dump(vocabulary, stream);
    pickle.dump(names, stream);
    pickle.dump(prefixes, stream);
    pickle.dump(suffixes, stream);

  forms_input = Input(shape = (LENGTH,));
  model = Embedding(input_dim = len(vocabulary),
                    output_dim = arguments.wd,
                    mask_zero = arguments.mask)(forms_input);
  if arguments.cues:
    cues_input = Input(shape = (LENGTH,));
    cues_model = Embedding(input_dim = 4, # {0, 1, 2, or 3}
                           output_dim = arguments.cd,
                           mask_zero = arguments.mask)(cues_input);
    model = concatenate([model, cues_model]);
  model = Bidirectional(LSTM(units = arguments.ru, return_sequences = True,
                             dropout = arguments.do,
                             recurrent_dropout = arguments.rd))(model);
  output = Dense(len(classes) - (2 if arguments.cues else 0),
                 activation = "softmax")(model);
  model = Model([forms_input, cues_input] if arguments.cues else forms_input,
                output);
  optimizer = optimizers.Adam(lr = arguments.lr);
  model.compile(optimizer = optimizer, loss = "categorical_crossentropy",
                metrics = ["accuracy"]);
  print(model.summary());

  monitor = EarlyStopping(monitor = "val_acc", min_delta = 0.0001,
                          patience = 3, verbose = 1, mode = "max");
  board = TensorBoard(log_dir = "log/{}".format(arguments.id));

  model.fit(([inputs, cues] if arguments.cues else inputs), outputs,
            validation_split = arguments.vs,
            batch_size = arguments.bs, epochs = arguments.epochs,
            callbacks = [monitor, board],
            verbose = 1);
  if arguments.debug:
    print("model.evaluate() on training: {}"
          "".format(model.evaluate(([inputs, cues] if arguments.cues
                                    else inputs),
                                   outputs, verbose = 1)));

  model.save(arguments.id + ".h5");

  #
  # in a few, rare circumstances, we allow ourselves to re-interpret variable
  # names, as is the case of .inputs. and .outputs. here: now turning our focus
  # to the evaluation data.
  #
  n = 0;
  unknown = 0;
  inputs = np.zeros((len(test), LENGTH), dtype = int);
  cues = np.zeros((len(test), LENGTH), dtype = int);
  golds = np.zeros((len(test), LENGTH,
                    len(classes) - (2 if arguments.cues else 0)),
                   dtype = int);
  for i, sentence in enumerate(test):
    n += len(sentence["nodes"]);
    for j, node in enumerate(sentence["nodes"]):
      form = node["form"];
      if form in vocabulary:
        inputs[i, j] = vocabulary[form];
      else:
        inputs[i, j] = vocabulary["__?__"];
        unknown += 1;
      negation = node.get("negation");
      if negation and len(negation):
        if "cue" in negation[0]:
          if negation[0]["cue"] == form:
            cues[i, j] = 2; # a word cue
            if arguments.cues:
              golds[i, j, classes["T"]] = 1;
            else:
              golds[i, j, classes["C"]] = 1;
          else:
            cues[i, j] = 3; # an affix cue
            if arguments.cues:
              golds[i, j, classes["T"]] = 1;
            else:
              golds[i, j, classes["A"]] = 1;
        elif "scope" in negation[0]:
          cues[i, j] = 1; # not a cue
          golds[i, j, classes["F"]] = 1;
        else:
          cues[i, j] = 1; # not a cue
          golds[i, j, classes["T"]] = 1;

  if arguments.debug:
    print("evaluation: {} word tokens; {} unknown."
          "".format(n, unknown));
    print("model.evaluate() on evaluation: {}"
          "".format(model.evaluate(([inputs, cues] if arguments.cues
                                    else inputs),
                                   golds, verbose = 1)));

  outputs = model.predict(([inputs, cues] if arguments.cues else inputs),
                          verbose = 1);
  tf.keras.backend.clear_session();

  #
  # convert back from ‘categorical’, one-hot encoding and un-pad;
  # while at it, (wastefully :-) produce two flat lists of labels.
  #
  golds = [np.argmax(gold, axis = 1) for gold in golds];
  outputs = [np.argmax(output, axis = 1) for output in outputs];
  labels = [];
  system = [];
  for i, sentence in enumerate(test):
    golds[i] = golds[i][0:len(sentence["nodes"])];
    labels.extend(golds[i]);
    outputs[i] = outputs[i][0:len(sentence["nodes"])];
    system.extend(outputs[i]);

  #
  # call out to SciKit-learn (alas) for a per-class summary
  #
  print(classification_report(labels, system, target_names = classes));

  if arguments.debug:
    for i, sentence in enumerate(test):
      for j, (node, cue, label) in enumerate(zip(sentence["nodes"],
                                                 cues[i], outputs[i])):
        print("{}\t{}\t{}".format(names[int(inputs[i, j])], cue, label));
      print();

  prediction = copy.deepcopy(test);
  for i, sentence in enumerate(prediction):
    #
    # as a kind of wellformedness condition on our negation predictions, only
    # output a negation instance if there is at least one cue.
    #
    if arguments.cues and sentence["negations"] \
       or classes["C"] in outputs[i] or classes["A"] in outputs[i]:
      #
      # _fix_me_
      # if sentence length exceeds the number of labels, add ‘padding’ labels
      #
      for node, label in zip(sentence["nodes"], outputs[i]):
        form = node["form"];
        negation = {};
        if not arguments.cues:
          if label == classes["C"]:
            negation["cue"] = form;
          elif label == classes["A"]:
            for prefix in prefixes:
              if form.startswith(prefix):
                negation["cue"] = prefix;
                negation["scope"] = form[len(prefix):];
                break;
            for suffix in suffixes:
              #
              # _fix_me_
              # i think, we can also see {help}<less>{ly}, which would require
              # a little more subtlety in suffix handling.
              #
              if form.startswith(suffix):
                negation["cue"] = suffix;
                negation["scope"] = form[:-len(suffix)];
                break;
          if label == classes["F"]:
            negation["scope"] = form;
        else:
          if label == classes["F"]:
            negation["scope"] = form;
          gold = node.get("negation");
          if gold and len(gold) and "cue" in gold[0]:
            #
            # we need to be careful and deliberate in using gold-standard cue
            # information to enrich our predictions: copying "cue" values must
            # be fair game, of course; and given the known conventions about
            # affix scopes, picking up the complement sub-string as in-scope
            # for sub-token clues also must be considered kosher.
            #
            cue = gold[0]["cue"];
            negation["cue"] = cue;
            if cue != form:
              if form.startswith(cue):
                negation["scope"] = form[len(cue):];
              elif form.endswith(cue):
                negation["scope"] = form[:-len(cue)];
        node["negation"] = [negation];
      sentence["negations"] = 1;
    else:
      for node in sentence["nodes"]:
        node["negation"] = [];
      sentence["negations"] = 0;

  if arguments.output:
    write_negations(prediction, arguments.output);
  if arguments.score:
    starsem_score(test, prediction, arguments.score);
  exit(0);

