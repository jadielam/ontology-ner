from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import random
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import sys
import json

from datasets import load_windows, load_articles, generate_examples
import model.features as features

# All capitalized constants come from this file

random.seed(42)

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    test_on_corpus(conf)

def test_on_mycorpus(conf):
    """Tests on the corpus set in ARTICLES_FILEPATH.
    Prints a full report, including precision, recall and F1 score per label.
    Args:
        args: Command line arguments as parsed by argparse.ArgumentParser.
    """
    articles_filepath = conf['articles_filepath']
    count_windows_test = conf['count_windows_test']
    window_size = conf['window_size']

    print("Testing on mycorpus (%s)..." % (articles_filepath))
    test_on_articles("test", load_articles(articles_filepath),
                     window_size,
                     nb_append = count_windows_test)

def test_on_articles(identifier, articles, window_size, nb_append=None):
    """Test a trained CRF model on a list of Article objects (annotated text).
    Will print a full classification report by label (f1, precision, recall).
    Args:
        identifier: Identifier of the trained model to be used.
        articles: A list of Article objects or a generator for such a list. May only contain
            one single Article object.
    """
    print("Loading tagger...")
    tagger = pycrfsuite.Tagger()
    tagger.open("train")

    # create feature generators
    # this may take a while
    print("Creating features...")
    feature_generators = features.create_features()

    # create window generator
    print("Loading windows...")
    windows = load_windows(articles, window_size, feature_generators, only_labeled_windows = False)

    # load feature lists and label lists (X, Y)
    # this may take a while
    all_feature_values_lists = []
    correct_label_chains = []
    for fvlist, labels in generate_examples(windows, nb_append=nb_append):
        all_feature_values_lists.append(fvlist)
        correct_label_chains.append(labels)

    # generate predicted chains of labels
    print("Testing on %d windows..." % (len(all_feature_values_lists)))
    predicted_label_chains = [tagger.tag(fvlists) for fvlists in all_feature_values_lists]

    # print classification report (precision, recall, f1)
    print(bio_classification_report(correct_label_chains, predicted_label_chains))

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    Note: This function was copied from
    http://nbviewer.ipython.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
    Args:
        y_true: True labels, list of strings
        y_pred: Predicted labels, list of strings
    Returns:
        classification report as string
    """
    lbin = LabelBinarizer()
    y_true_combined = lbin.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lbin.transform(list(chain.from_iterable(y_pred)))

    #tagset = set(lbin.classes_) - {NO_NE_LABEL}
    tagset = set(lbin.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lbin.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )

# ----------------------

if __name__ == "__main__":
    main()
