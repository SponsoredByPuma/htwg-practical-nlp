"""Tweet preprocessing module.

This module contains the TweetProcessor class which is used to preprocess tweets.

ASSIGNMENT-1:
Your job in this assignment is to implement the methods of this class.
Note that you will need to import several modules from the nltk library,
as well as from the Python standard library.
You can find the documentation for the nltk library here: https://www.nltk.org/
You can find the documentation for the Python standard library here: https://docs.python.org/3/library/
Your task is complete when all the tests in the test_preprocessing.py file pass.
You can check if the tests pass by running `make assignment-1` in the terminal.
You can follow the `TODO ASSIGNMENT-1` comments to find the places where you need to implement something.
"""

import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer, TweetTokenizer


class TweetProcessor:
    # TODO ASSIGNMENT-1: Add a `stemmer` attribute to the class

    # TODO ASSIGNMENT-1: Add a `tokenizer` attribute to the class
    #  - text should be lowercased
    #  - handles should be stripped
    #  - the length should be reduced for repeated characters

    @staticmethod
    def remove_urls(tweet: str) -> str:
        """Remove urls from a tweet.

        Args:
            tweet (str): the input tweet

        Returns:
            str: the tweet without urls
        """
        # TODO ASSIGNMENT-1: implement this function
        return re.sub(r"http[s]?://\S+", "", tweet)

    @staticmethod
    def remove_hashtags(tweet: str) -> str:
        """Remove hashtags from a tweet.
        Only the hashtag symbol is removed, the word itself is kept.

        Args:
            tweet (str): the input tweet

        Returns:
            str: the tweet without hashtags symbols
        """
        # TODO ASSIGNMENT-1: implement this function
        return tweet.replace("#", "")

    def tokenize(self, tweet: str) -> list[str]:
        """Tokenizes a tweet using the nltk TweetTokenizer.
        This also lowercases the tweet, removes handles, and reduces the length of repeated characters.

        Args:
            tweet (str): the input tweet

        Returns:
            list[str]: the tokenized tweet
        """
        # TODO ASSIGNMENT-1: implement this function
        # Initialize the TweetTokenizer with handle stripping
        tknzr = TweetTokenizer(strip_handles=True)

        # Lowercase the tweet
        tweet = tweet.lower()

        # Reduce repeated characters to a maximum of 3 (e.g., "looooveee" -> "loooveee")
        tweet = re.sub(r"(.)\1{3,}", r"\1\1\1", tweet)

        # Tokenize the tweet
        tokens = tknzr.tokenize(tweet)

        return tokens

    @staticmethod
    def remove_stopwords(tokens: list[str]) -> list[str]:
        """Removes stopwords from a tweet.

        Only English stopwords are removed.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet without stopwords
        """
        # TODO ASSIGNMENT-1: implement this function
        words = stopwords.words("english")
        returnList = []
        for token in tokens:
            if token in words:
                pass
            else:
                returnList.append(token)
        return returnList

    @staticmethod
    def remove_punctuation(tokens: list[str]) -> list[str]:
        """Removes punctuation from a tweet.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet without punctuation
        """
        # TODO ASSIGNMENT-1: implement this function
        # Patterns to allow specific structures
        returnList = []
        text_pattern = r"(?<!\.)[^\w\s'#@]"
        emoticon_pattern = r"[:;][-~]?[()DpPd]"
        dots_pattern = r"\.{2,}"

        for token in tokens:
            # Check if the token is an emoticon
            if re.match(emoticon_pattern, token):
                returnList.append(token)
            # Check if the token is a sequence of dots
            elif re.match(dots_pattern, token):
                returnList.append(token)
            else:
                # Remove unwanted punctuation from other tokens
                new_word = re.sub(text_pattern, "", token)
                if len(new_word) > 0:
                    returnList.append(new_word)

        return returnList

    def stem(self, tokens: list[str]) -> list[str]:
        """Stems the tokens of a tweet using the nltk PorterStemmer.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet with stemmed tokens
        """
        # TODO ASSIGNMENT-1: implement this function
        ps = PorterStemmer()
        returnList = []
        for token in tokens:
            returnList.append(ps.stem(token))
        return returnList

    def process_tweet(self, tweet: str) -> list[str]:
        """Processes a tweet by removing urls, hashtags, stopwords, punctuation, and stemming the tokens.

        Args:
            tweet (str): the input tweet

        Returns:
            list[str]: the processed tweet
        """
        # TODO ASSIGNMENT-1: implement this function
        without_url = self.remove_urls(tweet=tweet)
        without_hashtags = self.remove_hashtags(tweet=without_url)
        tokenized_tweet = self.tokenize(tweet=without_hashtags)
        without_stopwords = self.remove_stopwords(tokens=tokenized_tweet)
        without_punctuation = self.remove_punctuation(tokens=without_stopwords)
        return self.stem(tokens=without_punctuation)
