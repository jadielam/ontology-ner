# -*- coding: utf-8 -*-
"""
Contains:
    1. Various classes (feature generators) to convert windows (of words/tokens) to feature values.
       Each feature value is a string, e.g. "starts_with_uppercase=1", "brown_cluster=123".
    2. A method to create all feature generators.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import re
from collections import OrderedDict

def bucketize_minimum_distance(value):
    return int(value * 20)
    
class Cache:
    '''
    In process memory cache. Not thread safe
    '''
    def __init__(self, max_size = 100000):
        self._store = OrderedDict()
        self._max_size = max_size

    def set(self, key, value):
        self._check_limit()
        self._store[key] = value
    
    def get(self, key, default = None):
        data = self._store.get(key)
        if not data:
            return default
        return data
    
    def _check_limit(self):
        if len(self._store) >= self._max_size:
            self._store.popitem(last = False)
    
    def clear(self):
        self._store = OrderedDict()


from pyner.features.brown import BrownClusters
from pyner.features.gazetteer import Gazetteer, AllGazetteer
from pyner.features.lda import LdaWrapper
from pyner.features.pos import PosTagger
from pyner.features.w2v import W2VClusters

# All capitalized constants come from this file

def create_features(gazetteers_data, brown_clusters_filepath, w2v_clusters_filepath,
                    lda_model_filepath, lda_dictionary_filepath, lda_cache_filepath,
                    verbose=True, lda_window_left_size = 5, lda_window_right_size = 5,
                    features_to_extract = None):
    """This method creates all feature generators.
    The feature generators will be used to convert windows of tokens to their string features.
    This function may run for a few minutes.
    Args:
        verbose: Whether to output messages.
    Returns:
        List of feature generators
    """

    def print_if_verbose(msg):
        """This method prints a message only if verbose was set to True, otherwise does nothing.
        Args:
            msg: The message to print.
        """
        if verbose:
            print(msg)

    # Create the gazetteer. The gazetteer will contain all names from ug_names that have a higher
    # frequency among those names than among all unigrams (from ug_all).
    print_if_verbose("Creating gazetteer...")
    gazetteers = []
    for gaz_type, gaz_filepath in gazetteers_data.items():
        gaz = Gazetteer(gaz_filepath, type = gaz_type)
        gazetteers.append(gaz)
    
    allgazetteer = AllGazetteer(gazetteers_data)

    # Load the mapping of word to brown cluster and word to brown cluster bitchain
    #print_if_verbose("Loading brown clusters...")
    #brown = BrownClusters(brown_clusters_filepath)

    # Load the mapping of word to word2vec cluster
    #print_if_verbose("Loading W2V clusters...")
    #w2vc = W2VClusters(w2v_clusters_filepath)

    # Load the wrapper for the gensim LDA
    #print_if_verbose("Loading LDA...")
    #lda = LdaWrapper(lda_model_filepath, lda_dictionary_filepath,
    #                 lda_cache_filepath)

    # Load the wrapper for the stanford POS tagger
    print_if_verbose("Loading POS-Tagger...")
    pos = PosTagger()

    # create feature generators
    result = [
        StartsWithUppercaseFeature(),
        TokenLengthFeature(),
        ContainsDigitsFeature(),
        ContainsPunctuationFeature(),
        OnlyDigitsFeature(),
        OnlyPunctuationFeature(),
        #W2VClusterFeature(w2vc),
        #BrownClusterFeature(brown),
        #BrownClusterBitsFeature(brown),
        WordPatternFeature(),
        PrefixFeature(),
        SuffixFeature(),
        POSTagFeature(pos),
        WordFeature(),
        #LDATopicFeature(lda, lda_window_left_size, lda_window_right_size),
        AllGazetteerMinimumDistanceToken(allgazetteer),
        #AllGazetteerMinimumDistanceEntry(allgazetteer),
        #AllGazetteerClosestEntryType(allgazetteer),
        #AllGazetteerClosestTypeNGram(allgazetteer, 2),
        #AllGazetteerMinimumDistanceNGram(allgazetteer, 2),
        AllGazetteerClosestTokenType(allgazetteer)
    ]# + [
    #    GazetteerOfficialName(gaz) for gaz in gazetteers
    # ] + [
    #    GazetteerSynonym(gaz) for gaz in gazetteers
    #] + [
    #    GazetteerMinimumDistanceOfficialName(gaz) for gaz in gazetteers
    #] + [
    #    GazetteerMinimumDistanceSynonym(gaz) for gaz in gazetteers
    #] + [
    #    GazetteerMinimumDistanceToken(gaz) for gaz in gazetteers
    #] + [
    #    GazetteerClosestToken(gaz) for gaz in gazetteers
    #] + [
    #    GazetteerTokenPosition(gaz) for gaz in gazetteers
    #] + [
    #    GazetteerMinimumDistanceNGram(gaz, 2) for gaz in gazetteers
    #] # + [
    #    GazetteerMinimumDistanceNGram(gaz, 3) for gaz in gazetteers
    #] + [
    #    GazetteerMinimumDistanceNGram(gaz, 4) for gaz in gazetteers
    #]

    return result

class StartsWithUppercaseFeature(object):
    """Generates a feature that describes, whether a given token starts with an uppercase letter."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["swu=%d" % (int(token.word[:1].istitle()))])
        return result

class WordFeature(object):
    def __init__(self):
        pass
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            result.append(["word_feature=%s" % token.word])
        return result

class TokenLengthFeature(object):
    """Generates a feature that describes the character length of a token."""
    def __init__(self, max_length=30):
        """Instantiates a new object of this feature generator.
        Args:
            max_length: The max length to return in the generated features, e.g. if set to 30 you
                will never get a "l=31" result, only "l=30" for a token with length >= 30.
        """
        self.max_length = max_length

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["l=%d" % (min(len(token.word), self.max_length))])
        return result

class ContainsDigitsFeature(object):
    """Generates a feature that describes, whether a token contains any digit."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_digits = re.compile(r'[0-9]+')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            any_digits = self.regexp_contains_digits.search(token.word) is not None
            result.append(["cD=%d" % (int(any_digits))])
        return result

class ContainsPunctuationFeature(object):
    """Generates a feature that describes, whether a token contains any punctuation."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_punctuation = re.compile(r'[\.\,\:\;\(\)\[\]\?\!]+')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            any_punct = self.regexp_contains_punctuation.search(token.word) is not None
            result.append(["cP=%d" % (int(any_punct))])
        return result

class OnlyDigitsFeature(object):
    """Generates a feature that describes, whether a token contains only digits."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_only_digits = re.compile(r'^[0-9]+$')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            only_digits = self.regexp_contains_only_digits.search(token.word) is not None
            result.append(["oD=%d" % (int(only_digits))])
        return result

class OnlyPunctuationFeature(object):
    """Generates a feature that describes, whether a token contains only punctuation."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexp_contains_only_punctuation = re.compile(r'^[\.\,\:\;\(\)\[\]\?\!]+$')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            only_punct = self.regexp_contains_only_punctuation.search(token.word) is not None
            result.append(["oP=%d" % (int(only_punct))])
        return result

class W2VClusterFeature(object):
    """Generates a feature that describes the word2vec cluster of the token."""
    def __init__(self, w2v_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            w2v_clusters: An instance of W2VClusters as defined in w2v.py that can be queried to
                estimate the cluster of a word.
        """
        self.w2v_clusters = w2v_clusters

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["w2v=%d" % (self.token_to_cluster(token))])
        return result

    def token_to_cluster(self, token):
        """Converts a token/word to its cluster index among the word2vec clusters.
        Args:
            token: The token/word to convert.
        Returns:
            cluster index as integer,
            or -1 if it wasn't found among the w2v clusters.
        """
        return self.w2v_clusters.get_cluster_of(token.word, -1)

class BrownClusterFeature(object):
    """Generates a feature that describes the brown cluster of the token."""
    def __init__(self, brown_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            brown_clusters: An instance of BrownClusters as defined in brown.py that can be queried
                to estimate the brown cluster of a word.
        """
        self.brown_clusters = brown_clusters

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["bc=%d" % (self.token_to_cluster(token))])
        return result

    def token_to_cluster(self, token):
        """Converts a token/word to its cluster index among the brown clusters.
        Args:
            token: The token/word to convert.
        Returns:
            cluster index as integer,
            or -1 if it wasn't found among the brown clusters.
        """
        return self.brown_clusters.get_cluster_of(token.word, -1)

class BrownClusterBitsFeature(object):
    """Generates a feature that contains the brown cluster bitchain of the token."""
    def __init__(self, brown_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            brown_clusters: An instance of BrownClusters as defined in brown.py that can be queried
                to estimate the brown cluster bitchain of a word.
        """
        self.brown_clusters = brown_clusters

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["bcb=%s" % (self.token_to_bitchain(token)[0:7])])
        return result

    def token_to_bitchain(self, token):
        """Converts a token/word to its brown cluster bitchain among the brown clusters.
        Args:
            token: The token/word to convert.
        Returns:
            brown cluster bitchain as string,
            or "" (empty string) if it wasn't found among the brown clusters.
        """
        return self.brown_clusters.get_bitchain_of(token.word, "")

class GazetteerOfficialName(object):
    """Generates a feature that describes, whether a token is contained in the gazetteer."""
    def __init__(self, gazetteer):
        """Instantiates a new object of this feature generator.
        Args:
            gazetteer: An instance of Gazetteer as defined in gazetteer.py that can be queried
                to estimate whether a word is contained in an Gazetteer.
        """
        self.g = gazetteer

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["g_official_{}=%d".format(self.g.type) % (int(self.g.contains_as_official_name(token.word)))])
        return result

class GazetteerSynonym(object):
    def __init__(self, gazetteer):
        self.g = gazetteer
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            result.append(["g_synonym_{}=%d".format(self.g.type) % (int(self.g.contains_as_synonym(token.word)))])
        return result

class AllGazetteerMinimumDistanceToken(object):
    def __init__(self, gazetteer):
        self._g = gazetteer
        self._cache = Cache()
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            minimum_distance = self._cache.get(token.word, None)
            if minimum_distance is None:
                minimum_distance = self._g.minimum_distance_to_token(token.word)
                self._cache.set(token.word, minimum_distance)
            result.append(["g_minimum_distance_token=%d" % minimum_distance])
        return result

class AllGazetteerMinimumDistanceEntry(object):
    def __init__(self, gazetteer):
        self._g = gazetteer
        self._cache = Cache()
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            minimum_distance = self._cache.get(token.word, None)
            if minimum_distance is None:
                minimum_distance = self._g.minimum_distance_to_entry(token.word)
                self._cache.set(token.word, minimum_distance)
            result.append(["g_minimum_distance_entry=%d" % minimum_distance])
        return result

class AllGazetteerClosestEntryType(object):
    def __init__(self, gazetteer):
        self._g = gazetteer
        self._cache = Cache()
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            types = self._cache.get(token.word, None)
            if types is None:
                types = self._g.closest_entry_types(token.word)
                self._cache.set(token.word, types)
            result.append(["g_types_entry=%s" % types])
        return result

class AllGazetteerClosestTokenType(object):
    def __init__(self, gazetteer):
        self._g = gazetteer
        self._cache = Cache()
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            types = self._cache.get(token.word, None)
            if types is None:
                types = self._g.closest_token_types(token.word)
                self._cache.set(token.word, types)
            result.append(["g_types_token=%s" % types])
        return result

class AllGazetteerClosestTypeNGram(object):
    def __init__(self, g, ngram):
        self._g = g
        self._ngram = ngram
        self._cache = Cache()
    
    def _find_ngrams(self, input_list, ngram):
        return list(zip(*[input_list[i:] for i in range(ngram)]))

    def convert_window(self, window):
        result = []
        ngrams = self._find_ngrams(window.tokens, self._ngram)
        for token_ngram in ngrams:
            phrase = " ".join([token.word for token in token_ngram])
            closest_types = self._cache.get(phrase, None)
            if closest_types is None:
                closest_types = self._g.closest_entry_types(phrase)
                self._cache.set(phrase, closest_types)
            result.append(["g_{}gram_types=%s".format(self._ngram) % closest_types])
        for _ in range(len(ngrams), len(window.tokens)):
            result.append(["g_{}gram_types=%s".format(self._ngram) % "NONE"])
        return result

class AllGazetteerMinimumDistanceNGram(object):
    def __init__(self, g, ngram):
        self._g = g
        self._ngram = ngram
        self._cache = Cache()
    
    def _find_ngrams(self, input_list, ngram):
        return list(zip(*[input_list[i:] for i in range(ngram)]))
    
    def convert_window(self, window):
        result = []
        ngrams = self._find_ngrams(window.tokens, self._ngram)
        for token_ngram in ngrams:
            phrase = " ".join([token.word for token in token_ngram])
            minimum_distance = self._cache.get(phrase, None)
            if minimum_distance is None:
                minimum_distance = self._g.minimum_distance_to_entry(phrase)
                self._cache.set(phrase, minimum_distance)
            result.append(["g_{}gram_distance=%d".format(self._ngram) % minimum_distance])
        for _ in range(len(ngrams), len(window.tokens)):
            result.append(["g_{}gram_distance=%d".format(self._ngram) % 1.0])
        return result

class GazetteerClosestToken(object):
    def __init__(self, gazetteer):
        self.g = gazetteer
        self.cache = Cache()

    def convert_window(self, window):
        result = []
        for token in window.tokens:
            closest_token = self.cache.get(token.word, None)
            if closest_token is None:
                closest_token = self.g.closest_token(token.word)
                self.cache.set(token.word, closest_token)
            result.append(["g_closest_{}=%s".format(self.g.type) % closest_token])
        return result

class GazetteerTokenPosition(object):
    def __init__(self, gazetteer):
        self.g = gazetteer
        self.cache = Cache()
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            token_position = self.cache.get(token.word, None)
            if token_position is None:
                closest_token = self.g.closest_token(token.word)
                token_position = self.g.token_position_in_name(closest_token)
                self.cache.set(token.word, token_position)

            result.append(["g_token_position_{}=%d".format(self.g.type) % token_position])
        return result
    

class GazetteerMinimumDistanceOfficialName(object):
    def __init__(self, gazetteer):
        self.g = gazetteer
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            result.append(["g_official_distance_{}=%d".format(self.g.type) % self.g.minimum_distance_to_official_name(token.word)])
        return result

class GazetteerMinimumDistanceSynonym(object):
    def __init__(self, gazetteer):
        self.g = gazetteer
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            result.append(["g_synonym_distance_{}=%d".format(self.g.type) % self.g.minimum_distance_to_synonym(token.word)])
            
        return result

class GazetteerMinimumDistanceToken(object):
    def __init__(self, gazetteer):
        self.g = gazetteer
        self.cache = Cache()
    
    def convert_window(self, window):
        result = []
        for token in window.tokens:
            minimum_distance = self.cache.get(token.word, None)
            if minimum_distance is None:
                minimum_distance = self.g.minimum_distance_to_token(token.word)
                minimum_distance = bucketize_minimum_distance(minimum_distance)
                self.cache.set(token.word, minimum_distance)
            result.append(["g_token_distance_{}=%d".format(self.g.type) % minimum_distance])
        
        return result

class GazetteerMinimumDistanceNGram(object):
    def __init__(self, gazetteer, ngram):
        self.g = gazetteer
        self.ngram = ngram
        self.cache = Cache()
    
    def _find_ngrams(self, input_list, ngram):
        return list(zip(*[input_list[i:] for i in range(ngram)]))

    def convert_window(self, window):
        result = []
        ngrams = self._find_ngrams(window.tokens, self.ngram)
        for token_ngram in ngrams:
            phrase = " ".join([token.word for token in token_ngram])
            minimum_distance = self.cache.get(phrase, None)
            if minimum_distance is None:
                minimum_distance = self.g.minimum_distance_to_synonym(phrase)
                minimum_distance = bucketize_minimum_distance(minimum_distance)
                self.cache.set(phrase, minimum_distance)
            result.append(["g_{}gram_{}_distance=%d".format(self.ngram, self.g.type) % minimum_distance])
        for _ in range(len(ngrams), len(window.tokens)):
            result.append(["g_{}gram_{}_distance=%d".format(self.ngram, self.g.type) % bucketize_minimum_distance(1.0)])
        return result

class WordPatternFeature(object):
    """Generates a feature that describes the word pattern of a feature.
    A word pattern is a rough representation of the word, examples:
        original word | word pattern
        ----------------------------
        John          | Aa+
        Washington    | Aa+
        DARPA         | A+
        2055          | 9+
    """
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        # maximum length of tokens after which to simply cut off
        self.max_length = 15
        # if cut off because of maximum length, use this char at the end of the word to signal
        # the cutoff
        self.max_length_char = "~"

        self.normalization = [
            (r"[A-ZÄÖÜ]", "A"),
            (r"[a-zäöüß]", "a"),
            (r"[0-9]", "9"),
            (r"[\.\!\?\,\;]", "."),
            (r"[\(\)\[\]\{\}]", "("),
            (r"[^Aa9\.\(]", "#")
        ]

        # note: we do not map numers to 9+, e.g. years will still be 9999
        self.mappings = [
            (r"[A]{2,}", "A+"),
            (r"[a]{2,}", "a+"),
            (r"[\.]{2,}", ".+"),
            (r"[\(]{2,}", "(+"),
            (r"[#]{2,}", "#+")
        ]

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["wp=%s" % (self.token_to_wordpattern(token))])
        return result

    def token_to_wordpattern(self, token):
        """Converts a token/word to its word pattern.
        Args:
            token: The token/word to convert.
        Returns:
            The word pattern as string.
        """
        normalized = token.word
        for from_regex, to_str in self.normalization:
            normalized = re.sub(from_regex, to_str, normalized)

        wpattern = normalized
        for from_regex, to_str in self.mappings:
            wpattern = re.sub(from_regex, to_str, wpattern)

        if len(wpattern) > self.max_length:
            wpattern = wpattern[0:self.max_length] + self.max_length_char

        return wpattern

class PrefixFeature(object):
    """Generates a feature that describes the prefix (the first three chars) of the word."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            prefix = re.sub(r"[^a-zA-ZäöüÄÖÜß\.\,\!\?]", "#", token.word[0:3])
            result.append(["pf=%s" % (prefix)])
        return result

class SuffixFeature(object):
    """Generates a feature that describes the suffix (the last three chars) of the word."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            suffix = re.sub(r"[^a-zA-ZäöüÄÖÜß\.\,\!\?]", "#", token.word[-3:])
            result.append(["sf=%s" % (suffix)])
        return result

class POSTagFeature(object):
    """Generates a feature that describes the Part Of Speech tag of the word."""
    def __init__(self, pos_tagger):
        """Instantiates a new object of this feature generator.
        Args:
            pos_tagger: An instance of PosTagger as defined in pos.py that can be queried
                to estimate the POS-tag of a word.
        """
        self.pos_tagger = pos_tagger

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        pos_tags = self.default_pos_tag(window)
        result = []
        
        # catch stupid problems with stanford POS tagger and unicode characters
        if len(pos_tags) == len(window.tokens):
            # _ is the word
            for _, pos_tag in pos_tags:
                result.append(["pos=%s" % (pos_tag)])
        else:
            orig_str = "|".join([token.word for token in window.tokens])
            pos_str = "|".join([word for word, _ in pos_tags])
            print("[Info] Stanford POS tagger got sequence of length %d, returned " \
                  "POS-sequence of length %d. This sometimes happens with special unicode " \
                  "characters. Returning empty list instead." % (len(window.tokens), len(pos_tags)))
            print("[Info] Original sequence was:", orig_str)
            print("[Info] Tagged sequence      :", pos_str)

            # fill with empty feature value lists (one empty list per token)
            for _ in range(len(window.tokens)):
                result.append([])

        return result

    def default_pos_tag(self, window):
        """Converts a Window (list of tokens) to their POS tags.
        Args:
            window: Window object containing the token list to POS-tag.
        Returns:
            List of POS tags as strings.
        """
        return self.pos_tagger.tag([token.word for token in window.tokens])

class LDATopicFeature(object):
    """Generates a list of features that contains one or more topics of the window around the
    word."""
    def __init__(self, lda_wrapper, window_left_size, window_right_size, prob_threshold=0.2):
        """Instantiates a new object of this feature generator.
        Args:
            lda_wrapper: An instance of LdaWrapper as defined in models/lda.py that can be queried
                to estimate the LDA topics of a window around a word.
            window_left_size: Size in words/tokens to the left of a word/token to use for the LDA.
            window_right_size: See window_left_size.
            prob_threshold: The probability threshold to use for the topics. If a topic has a
                higher porbability than this threshold, it will be added as a feature,
                e.g. "lda_15=1" if topic 15 has a probability >= 0.2.
        """
        self.lda_wrapper = lda_wrapper
        self.window_left_size = window_left_size
        self.window_right_size = window_right_size
        self.prob_threshold = prob_threshold

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for i, token in enumerate(window.tokens):
            token_features = []
            window_start = max(0, i - self.window_left_size)
            window_end = min(len(window.tokens), i + self.window_right_size + 1)
            window_tokens = window.tokens[window_start:window_end]
            text = " ".join([token.word for token in window_tokens])
            topics = self.get_topics(text)
            for (topic_idx, prob) in topics:
                if prob > self.prob_threshold:
                    token_features.append("lda_%d=%s" % (topic_idx, "1"))
            result.append(token_features)
        return result

    def get_topics(self, text):
        """Converts a small text window (string) to its LDA topics.
        Args:
            text: The small text window to convert (as string).
        Returns:
            List of tuples of form (topic index, probability).
        """
        return self.lda_wrapper.get_topics(text)