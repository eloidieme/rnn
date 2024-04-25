import numpy as np
from pathlib import Path

def make_charmap(book_fname: str = "./data/goblet_book.txt"):
    book_path = Path(book_fname)
    with open(book_path, 'r') as book:
        book_data = book.read()

    word_list = book_data.split()
    chars = [[*word] for word in word_list]
    max_len = max(len(word) for word in chars)
    for wordl in chars:
        while len(wordl) < max_len:
            wordl.append(' ')
    chars = np.array(chars)

    unique_chars = list(np.unique(chars))
    unique_chars.append('\n')
    unique_chars.append('\t')
    K = len(unique_chars)  # dimensionality of the input and output vectors

    char_to_ind = {}
    ind_to_char = {}
    for idx, char in enumerate(unique_chars):
        char_to_ind[char] = idx
        ind_to_char[idx] = char

    return book_data, K, char_to_ind, ind_to_char

def encode_char(char, char_to_ind, K):
    oh = [0]*K
    oh[char_to_ind[char]] = 1
    return oh

