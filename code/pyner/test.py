from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import random
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import sys
import json

from pyner.datasets import load_windows, load_articles, generate_examples, Article, Window
import pyner.features.features as features
from pyner.features.gazetteer import Gazetteer

random.seed(42)

def tag_sentence_factory(conf):
    gaz_filepaths = conf.get('gaz_filepaths', None)
    brown_clusters_filepath = conf.get('brown_clusters_filepath', None)
    w2v_clusters_filepath = conf.get('w2v_clusters_filepath', None)
    lda_model_filepath = conf.get('lda_model_filepath', None)
    lda_dictionary_filepath = conf.get('lda_dictionary_filepath', None)
    lda_cache_filepath = conf.get('lda_cache_filepath', None)
    model_output_path = conf['model_output_path']
    skip_chain_left = conf['skip_chain_left']
    skip_chain_right = conf['skip_chain_right']

    print("Creating tagger... ")
    tagger = pycrfsuite.Tagger()
    tagger.open(model_output_path)

    print("Creating feature extractors... ")
    feature_generators = features.create_features(gaz_filepaths, brown_clusters_filepath,
                                    w2v_clusters_filepath, lda_model_filepath, 
                                    lda_dictionary_filepath, lda_cache_filepath, 
                                    verbose = True, lda_window_left_size = 5,
                                    lda_window_right_size = 5)


    gaz_d = {}
    for gaz_type, gaz_filepath in gaz_filepaths.items():
        gaz = Gazetteer(gaz_filepath, type = gaz_type)
        gaz_d[gaz_type] = gaz 

    def tag_sentence(sentence_text):
        sentence_text = sentence_text.lower()
        article = Article(sentence_text)
        window = Window(article.tokens)
        window.apply_features(feature_generators)

        feature_values_lists = []
        for word_idx in range(len(window.tokens)):
            fvl = window.get_feature_values_list(word_idx, skip_chain_left, skip_chain_right)
            feature_values_lists.append(fvl)
        tagged_sequence = tagger.tag(feature_values_lists)
        return tagged_sequence

    return tag_sentence

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    tag_sentence = tag_sentence_factory(conf)

    while True:
        query_text = raw_input("Your text: ")
        if query_text == "exit":
            break
        tagged_sequence = tag_sentence(query_text)
        print("Tagged: {}".format(tagged_sequence))

if __name__ == "__main__":
    main()