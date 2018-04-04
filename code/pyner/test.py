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

def transcribe_text(original_text, output_tags, gaz, tags_to_gaz_type_d):
    original_tokens = original_text.split()
    return original_text


def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
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

    tag_to_gaz_type_d = {
        'B-CHAR': 'characters',
        'I-CHAR': 'characters',
        'B-PARK': 'parks',
        'I-PARK': 'parks',
        'B-ATTR': 'attractions',
        'I-ATTR': 'attractions',
        'B-REST': 'restaurants',
        'I-REST': 'restaurants',
        'B-ENTE': 'entertainment',
        'I-ENTE': 'entertainment',
        'B-RESO': 'resorts',
        'I-RESO': 'resorts'
    } 

    while True:
        query_text = raw_input("Your text: ")
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
        print("Tagged: {}".format(tagged_sequence))
        transcribed_sentence = transcribe_text(query_text, tagged_sequence, gaz_d, tag_to_gaz_type_d)
        print("Transcribed: {}".format(transcribed_sentence))

if __name__ == "__main__":
    main()