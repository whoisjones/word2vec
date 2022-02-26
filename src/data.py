import nltk
from nltk.corpus import reuters
import ssl

def download_reuters():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("reuters")

def read_corpus():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    download_reuters()
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    files = reuters.fileids()
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]