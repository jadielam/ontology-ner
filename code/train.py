'''
Main training file for the CRF.  
This file trains a CRF model and saves it under the filename provided via an 'identifier' command line argument

'''

from __future__ import absolute_import, division, print_function, unicode_literals
import random
import sys
import json
import pycrfsuite

from datasets import load_windows, load_articles, generate_examples
import features.features as features

random.seed(42)

def main():
    '''
    Main function that reads parameter and starts the training process
    '''
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    train(conf)
    
def train(conf):
    gaz_filepaths = conf['gaz_filepath']
    brown_clusters_filepath = conf['brown_clusters_filepath']
    w2v_clusters_filepath = conf['w2v_clusters_filepath']
    lda_model_filepath = conf['lda_model_filepath']
    lda_dictionary_filepath = conf['lda_dictionary_filepath']
    lda_cache_filepath = conf['lda_cache_filepath']
    articles_filepath = conf['articles_filepath']
    window_size = conf['window_size']
    count_windows_train = conf['count_windows_train']
    count_windows_test = conf['count_windows_test']
    max_iterations = conf.get('max_iterations', None)

    print("Creating trainer... ")
    trainer = pycrfsuite.Trainer(verbose = True)

    print("Creating features... ")
    feature_generators = features.create_features(gaz_filepaths, brown_clusters_filepath,
                                    w2v_clusters_filepath, lda_model_filepath, 
                                    lda_dictionary_filepath, lda_cache_filepath, 
                                    verbose = True, lda_window_left_size = 5,
                                    lda_window_right_size = 5)

    print("Loading windows... ")
    windows = load_windows(load_articles(articles_filepath), window_size,
                            feature_generators, only_labeled_windows = True)

    print("Adding example windows (up to max %d)..." % (count_windows_train))
    examples = generate_examples(windows, nb_append = count_windows_train,
                                    nb_skip = count_windows_test, verbose = True)
    for feature_values_lists, labels in examples:
        trainer.append(feature_values_lists, labels)

    print("Training... ")
    if max_iterations is not None and max_iterations > 0:
        trainer.set_params({'max_iterations': max_iterations})
    trainer.train()                      

if __name__ == "__main__":
    main()