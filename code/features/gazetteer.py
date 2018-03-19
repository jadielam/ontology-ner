# -*- coding: utf-8 -*-
"""Class encapsulating a Gazetteer.
A Gazetteer contains a set of words that are names (e.g. names of people)."""
from __future__ import absolute_import, division, print_function, unicode_literals

class Gazetteer(object):
    """Class encapsulating a Gazetteer.
    A Gazetteer contains a set of words that are names (e.g. names of people)."""
    def __init__(self, phrases, type):
        """Initializes the gazetter and fills it from two unigrams list.
        Args:
            unigrams_names: Unigrams object that should contain only names (e.g. only names of
                people).
        """
        self.gazetteer = set()
        self.fill_gazetteer(phrases)

    def clear(self):
        """Resets/empties the Gazetteer."""
        self.gazetteer = set()

    def fill_gazetteer(self, phrases):
        """Fills the Gazetteer automatically from two lists of unigrams (as described in Args).
        After filling the Gazetteer only contains those names which appeared more often
        among the names than among the whole corpus.
        Args:
            unigrams_names: Unigrams object that should contain only names (e.g. only names of
                people).
            unigrams: Unigrams object that should contain all words of the corpus.
        """
        pass

    def contains(self, word):
        """Returns whether the Gazetteer contains the provided word.
        Args:
            word: The word to check.
        Returns:
            True if the word is contained in the Gazetteer, False otherwise.
        """
        return word in self.gazetteer