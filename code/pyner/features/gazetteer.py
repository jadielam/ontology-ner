# -*- coding: utf-8 -*-
"""Class encapsulating a Gazetteer.
A Gazetteer contains a set of words that are names (e.g. names of people)."""


from __future__ import absolute_import, division, print_function, unicode_literals
from array import array
import operator
    
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
    current_row = array('i', range(len(word) + 1))

    results = []
    
    for letter in trie.children:
        stack = [(trie.children[letter], letter, word, current_row, results)]    
        search_stack(stack, max_cost)
    return results

def search_stack(stack, max_cost):
    '''
    Searches recursively for the word in the trie.
    '''
    
    while stack:
        node, letter, word, previous_row, results = stack.pop()
        columns = len(word) + 1
        current_row = array('i', [0]*(len(word) + 1))
        current_row[0] = previous_row[0] + 1

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range(1, columns):
            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if word[column - 1] != letter:
                replace_cost = previous_row[column - 1] + 1
            else:                
                replace_cost = previous_row[column - 1]

            current_row[column] = min(insert_cost, delete_cost, replace_cost)

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if current_row[-1] <= max_cost and node.word != None:
            results.append((node.word, current_row[-1]))

        # if any entries in the row are less than the maximum cost, then 
        # recursively search each branch of the trie
        if min(current_row) <= max_cost:
            for letter in node.children:
                stack.append((node.children[letter], letter, word, current_row, results))

class AllGazetteer(object):
    def __init__(self, type_filepath_dict):
        self._token_types = dict()
        self._entry_types = dict()
        self._entries_trie = TrieNode()
        self._tokens_trie = TrieNode()

        self.fill_gazetteer(type_filepath_dict)
        
    def fill_gazetteer(self, type_filepath_dict):
        """Fills the gazetteer from a list of entries provided in a 
        text file.  Each line in the file corresponds to an entity. 
        Each line in the file is a list of comma separated values
        The first entry in the list is the original nam
        Args:
            file_path: 
        """

        for e_type, file_path in type_filepath_dict.items():
            with open(file_path, "r") as f:
                for line in f:
                    lower_line = line.lower()
                    entries = lower_line.split(",")
                    entries = [a.strip() for a in entries]

                    if len(entries) > 0:
                        for idx, entry in enumerate(entries):
                            # 1. Enter entry into trie and types
                            if not entry in self._entry_types:
                                self._entry_types[entry] = []
                                self._entries_trie.insert(entry)
                            self._entry_types[entry].append(e_type)
                            
                            #2. Enter tokens into type and trie
                            tokens = entry.split()
                            for idx, token in enumerate(tokens):
                                if not token in self._token_types:
                                    self._token_types[token] = []
                                    self._tokens_trie.insert(token)
                                self._token_types[token].append(e_type)
        
        #This sorting will reduce the dimensionality of features later on.
        for token, token_types in self._token_types.items():
            token_types.sort()
        for entry, entry_types in self._entry_types.items():
            entry_types.sort()
    
    def minimum_distance_to_token(self, phrase):
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self._tokens_trie, phrase.lower(), max_distance)
        
        if len(results) > 0:
            minimum_value = min(results, key = operator.itemgetter(1))[1]
            return minimum_value / float(len(phrase))
        else:
            return 1.0
    
    def minimum_distance_to_entry(self, phrase):
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self._entries_trie, phrase.lower(), max_distance)
        
        if len(results) > 0:
            minimum_value = min(results, key = operator.itemgetter(1))[1]
            return minimum_value / float(len(phrase))
        else:
            return 1.0
    
    def closest_entry_types(self, phrase):
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self._entries_trie, phrase.lower(), max_distance)
        
        if len(results) > 0:
            entry = min(results, key = operator.itemgetter(1))[0]
            if entry in self._entry_types:
                types = self._entry_types[entry]
                return "_".join(types)
        
        return "NONE"
    
    def closest_token_types(self, phrase):
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self._tokens_trie, phrase.lower(), max_distance)

        if len(results) > 0:
            token = min(results, key = operator.itemgetter(1))[0]
            if token in self._token_types:
                types = self._token_types[token]
                return "_".join(types)
        
        return "NONE"

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
        self.synonyms_to_official_name = dict()
        self.official_names_set = set()
        self.synonyms_set = set()
        self.official_names_trie = TrieNode()
        self.synonyms_trie = TrieNode()
        self.position_in_name = {}
        self.tokens_trie = TrieNode()
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
                lower_line = line.lower()
                entries = lower_line.split(",")
                entries = [a.strip() for a in entries]

                if len(entries) > 0:
                    self.official_names_set.add(entries[0])
                    self.official_names_trie.insert(entries[0])
                    self.synonyms_set.update(entries[0:])
                    for entry in entries[0:]:
                        self.synonyms_trie.insert(entry)
                    
                    for i in range(len(entries)):
                        official_name = entries[0]
                        entry = entries[i]
                        self.synonyms_to_official_name[entry] = official_name
                        tokens = entry.split()
                        for it, token in enumerate(tokens):
                            self.tokens_trie.insert(token)
                            self.position_in_name[token] = it

    def contains_as_official_name(self, phrase):
        """Returns whether the Gazetteer contains the entire phrase or not
        as an original name
        Args:
            word: The word to check.
        Returns:
            True if the word is contained in the Gazetteer, False otherwise.
        """
        return phrase.lower() in self.official_names_set
    
    def contains_as_synonym(self, phrase):
        '''
        Returns whether the Gazetteer contains the entire phrase or not
        as a synonym
        '''
        return phrase.lower() in self.synonyms_set
    
    def minimum_distance_to_token(self, phrase):
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self.tokens_trie, phrase.lower(), max_distance)

        if len(results) > 0:
            minimum_value = min(results, key = operator.itemgetter(1))[1]
            return minimum_value / float(len(phrase))
        else:
            return 1.0
    
    def minimum_distance_to_official_name(self, phrase):
        '''
        Returns the minimum Levenshtein distance value from the phrase to 
        any entry in the official_names_list.
        '''
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self.official_names_trie, phrase.lower(), max_distance)

        if len(results) > 0:
            minimum_value = min(results, key = operator.itemgetter(1))[1]
            return minimum_value / float(len(phrase))
        else:
            return 1.0
    
    def closest_official_name(self, phrase):
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self.synonyms_trie, phrase.lower(), max_distance)
        if len(results) > 0:
            entry = min(results, key = operator.itemgetter(1))[0]
            return self.synonyms_to_official_name[entry]
        else:
            return "NONE"

    def closest_token(self, phrase):
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self.tokens_trie, phrase.lower(), max_distance)

        if len(results) > 0:
            entry = min(results, key = operator.itemgetter(1))[0]
            return entry
        else:
            "None"
    
    def token_position_in_name(self, token):
        if token in self.position_in_name:
            return self.position_in_name[token]
        return -1

    def minimum_distance_to_synonym(self, phrase):
        '''
        Returns the minimum Levenshtein distance value from the phrase to
        any entry in the synonym name list.
        '''
        distance_percentage = 0.30
        max_distance = max(1, int(len(phrase) * distance_percentage))
        results = search(self.synonyms_trie, phrase.lower(), max_distance)
        if len(results) > 0:
            minimum_distance = min(results, key = operator.itemgetter(1))[1]
            return minimum_distance / float(len(phrase))
        else:
            return 1.0