from sklearn.base import BaseEstimator, TransformerMixin
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd


class CountPosTagTransformer(BaseEstimator, TransformerMixin):
    """
    Counting the number of some tags at a text.
    The tags could be: verb, adjective, pronoun and noun.
    """

    def __init__(self, tag_to_sum):
        self.tag_to_sum = tag_to_sum

    def sum_types(self, text):
        """Sum the number of a specific tag at a text

        Parameters
        ----------
        text: String
            The text to be analysed

        Returns
        -------
        Int
            The number of tags at a text
        """
        sentence = word_tokenize(text)
        tags_list = pos_tag(sentence)
        tags_total = 0
        for tag in tags_list:
            if self.tag_to_sum in tag[1]:
                tags_total += 1
        return tags_total

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        total_tags = pd.Series(X).apply(self.sum_types)
        return pd.DataFrame(total_tags)
