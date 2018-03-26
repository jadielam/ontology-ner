# -*- coding: utf-8 -*-
"""Class encapsulating a Gazetteer.
A Gazetteer contains a set of words that are names (e.g. names of people)."""


from __future__ import absolute_import, division, print_function, unicode_literals

class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

    def insert(self, word):
        node = self
        for letter in word:
            if letter not in node.children: 
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.word = word

def search(trie, word, max_cost):
    # build first row
    current_row = range(len(word) + 1)

    results = []

    for letter in trie.children:
        search_recursive(trie.children[letter], letter, word, current_row, 
            results, max_cost)
    return results

def search_recursive(node, letter, word, previous_row, results, max_cost):

    columns = len( word ) + 1
    current_row = [ previous_row[0] + 1 ]

    # Build one row for the letter, with a column for each letter in the target
    # word, plus one for the empty string at column 0
    for column in xrange(1, columns):

        insert_cost = current_row[column - 1] + 1
        delete_cost = previous_row[column] + 1

        if word[column - 1] != letter:
            replace_cost = previous_row[column - 1] + 1
        else:                
            replace_cost = previous_row[column - 1]

        current_row.append(min(insert_cost, delete_cost, replace_cost))

    # if the last entry in the row indicates the optimal cost is less than the
    # maximum cost, and there is a word in this trie node, then add it.
    if current_row[-1] <= max_cost and node.word != None:
        results.append((node.word, current_row[-1]))

    # if any entries in the row are less than the maximum cost, then 
    # recursively search each branch of the trie
    if min(current_row ) <= max_cost:
        for letter in node.children:
            search_recursive(node.children[letter], letter, word, current_row, 
                results, max_cost)

class Gazetteer(object):
    """Class encapsulating a Gazetteer.
    A Gazetteer contains a set of words that are names (e.g. names of people)."""
    def __init__(self, file_path, type):
        """Initializes the gazetter and fills it from two unigrams list.
        Args:
            unigrams_names: Unigrams object that should contain only names (e.g. only names of
                people).
        """
        self.type = type
        self.official_names_set = set()
        self.synonyms_set = set()
        self.official_names_trie = TrieNode()
        self.synonyms_trie = TrieNode()
        self.fill_gazetteer(file_path)

    def fill_gazetteer(self, file_path):
        """Fills the gazetteer from a list of entries provided in a 
        text file.  Each line in the file corresponds to an entity. 
        Each line in the file is a list of comma separated values
        The first entry in the list is the original nam
        Args:
            file_path: 
        """

        with open(file_path) as f:
            for line in f:
                entries = line.split(",")
                if len(entries) > 0:
                    self.official_names_set.add(entries[0])
                    self.official_names_trie.insert(entries[0])
                    self.synonyms_set.update(entries[0:])
                    for entry in entries[0:]:
                        self.synonyms_trie.insert(entry)

    def contains_as_official_name(self, phrase):
        """Returns whether the Gazetteer contains the entire phrase or not
        as an original name
        Args:
            word: The word to check.
        Returns:
            True if the word is contained in the Gazetteer, False otherwise.
        """
        return phrase in self.official_names_set
    
    def contains_as_synonym(self, phrase):
        '''
        Returns whether the Gazetteer contains the entire phrase or not
        as a synonym
        '''
        return phrase in self.synonyms_set
    
    def minimum_distance_to_official_name(self, phrase):
        '''
        Returns the minimum Levenshtein distance value from the phrase to 
        any entry in the official_names_list.
        '''
        results = search(self.official_names_trie, phrase, 5)
        if len(results) > 0:
            return results[0][1]
        else:
            return 999
    
    def minimum_distance_to_synonym(self, phrase):
        '''
        Returns the minimum Levenshtein distance value from the phrase to
        any entry in the synonym name list.
        '''
        results = search(self.synonyms_trie, phrase, 5)
        if len(results) > 0:
            return results[0][1]
        else:
            return 999