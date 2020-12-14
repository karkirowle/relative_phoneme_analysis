import editdistance
import numpy as np


def levenshtein_distance(ref: str, hyp: str) -> np.ndarray:

    # ref: kitten
    # hyp: sitting

    N = len(hyp)
    M = len(ref)
    distance_matrix = np.zeros((N + 1, M + 1))

    # Fill up the
    distance_matrix[0,:] = list(range(M + 1)) # includes last
    distance_matrix[:,0] = list(range(N + 1)) # includes last

    print(distance_matrix)

    for i in range(1,N+1):
        for j in range(1,M+1):

            string_i = i - 1
            string_j = j - 1
            if ref[string_j] == hyp[string_i]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            distance_matrix[i, j] = np.minimum(
                np.minimum(distance_matrix[i - 1, j] + 1,  distance_matrix[i, j - 1] + 1),
                            distance_matrix[i - 1, j - 1] + substitution_cost) # substitution

    k = np.minimum(N,M)


    diagonal = distance_matrix[list(range(k+1)),list(range(k+1))]

    #idx = np.range
    print(diagonal)

    # kitten
    # sitting

    # k -> s
    # i -> i
    # t -> t
    # t -> t
    # i -> i
    # n -> n
    # g -> *
    return distance_matrix


def calculate_wer(seqs_hat, seqs_true):
    """Calculate sentence-level WER score. (From ESPNET)

    # Note to myself: sensitive to uppercase-lowercase

    :param list seqs_hat: prediction
    :param list seqs_true: reference
    :return: average sentence-level WER score
    :rtype float
    """
    word_eds, word_ref_lens = [], []
    for i, seq_hat_text in enumerate(seqs_hat):
        seq_true_text = seqs_true[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        print(hyp_words)
        print(ref_words)
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        word_ref_lens.append(len(ref_words))
    return float(sum(word_eds)) / sum(word_ref_lens)


if __name__ == '__main__':

    d = levenshtein_distance("kitten","sitting")
    print(d)
    #print(calculate_wer(["this is something"],["cheese this is something"]))

    # Number of reference words: 3
    # Substitution, deletion, insertion
    #print(calculate_wer())