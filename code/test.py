from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import random
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import sys
import json

from datasets import load_windows, load_articles, generate_examples, Article, Window
import features.features as features

# All capitalized constants come from this file

random.seed(42)

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    gaz_filepaths = conf.get('gaz_filepaths', None)
    brown_clusters_filepath = conf.get('brown_clusters_filepath', None)
    w2v_clusters_filepath = conf.get('w2v_clusters_filepath', None)
    lda_model_filepath = conf.get('lda_model_filepath', None)
    lda_dictionary_filepath = conf.get('lda_dictionary_filepath', None)
    lda_cache_filepath = conf.get('lda_cache_filepath', None)
    skip_chain_left = conf['skip_chain_left']
    skip_chain_right = conf['skip_chain_right']

    print("Creating tagger... ")
    tagger = pycrfsuite.Tagger()
    tagger.open("train")

    print("Creating feature extractors... ")
    feature_generators = features.create_features(gaz_filepaths, brown_clusters_filepath,
                                    w2v_clusters_filepath, lda_model_filepath, 
                                    lda_dictionary_filepath, lda_cache_filepath, 
                                    verbose = True, lda_window_left_size = 5,
                                    lda_window_right_size = 5)
                                    
    while True:
        query_text = input("Your text: ")
        if query_text == "exit":
            break
        query_text = query_text.lower()
        article = Article(query_text)
        window = Window(article.tokens)
        window.apply_features(feature_generators)

        feature_values_lists = []
        for word_idx in range(len(window.tokens)):
            fvl = window.get_feature_values_list(word_idx, skip_chain_left, skip_chain_right)
            feature_values_lists.append(fvl)
        tagged_sequence = tagger.tag(feature_values_lists)
        print(tagged_sequence)

if __name__ == "__main__":
    main()