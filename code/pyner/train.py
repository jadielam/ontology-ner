'''
Main training file for the CRF.  
This file trains a CRF model and saves it under the filename provided via an 'identifier' command line argument

'''

from __future__ import absolute_import, division, print_function, unicode_literals
import random
import sys
import json
import itertools
import pycrfsuite


from pyner.datasets import load_windows, load_articles, generate_examples
import pyner.features.features as features

random.seed(42)

def main():
    '''
    Main function that reads parameter and starts the training process
    '''
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    train(conf)
    
def train(conf):
    gaz_filepaths = conf.get('gaz_filepaths', None)
    brown_clusters_filepath = conf.get('brown_clusters_filepath', None)
    w2v_clusters_filepath = conf.get('w2v_clusters_filepath', None)
    lda_model_filepath = conf.get('lda_model_filepath', None)
    lda_dictionary_filepath = conf.get('lda_dictionary_filepath', None)
    lda_cache_filepath = conf.get('lda_cache_filepath', None)
    articles_filepath = conf.get('articles_filepath')
    window_size = conf['window_size']
    count_windows_train = conf['count_windows_train']
    count_windows_test = conf['count_windows_test']
    skip_chain_left = conf['skip_chain_left']
    skip_chain_right = conf['skip_chain_right']
    max_iterations = conf.get('max_iterations', None)
    features_to_extract = conf.get('features_to_extract', None)

    print("Creating trainer... ")
    trainer = pycrfsuite.Trainer(verbose = True)

    print("Creating features... ")
    feature_generators = features.create_features(gaz_filepaths, brown_clusters_filepath,
                                    w2v_clusters_filepath, lda_model_filepath, 
                                    lda_dictionary_filepath, lda_cache_filepath, 
                                    verbose = True, lda_window_left_size = 5,
                                    lda_window_right_size = 5, 
                                    features_to_extract = features_to_extract)

    print("Loading articles... ")
    articles = load_articles(articles_filepath)
    articles = list(articles)

    print("Loading windows... ")
    windows = load_windows(articles, window_size,
                            feature_generators, only_labeled_windows = False)
    windows = itertools.islice(windows, count_windows_train + count_windows_test)

    print("Adding example windows (up to max %d)..." % (count_windows_train))
    examples = generate_examples(windows, skip_chain_left, skip_chain_right, nb_append = count_windows_train,
                                    nb_skip = count_windows_test, verbose = True)
    examples = list(examples)

    counter = 0
    for feature_values_lists, labels in examples:
        counter += 1
        trainer.append(feature_values_lists, labels, group = counter % 10)

    print("Training... ")
    if max_iterations is not None and max_iterations > 0:
        trainer.set_params({'max_iterations': max_iterations,
                            'c1': 1.0,
                            'c2': 1e-4,
                            'feature.minfreq': 0 })
    trainer.train("train", holdout = 1)                    

if __name__ == "__main__":
    main()