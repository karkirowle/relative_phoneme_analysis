import numpy
import sys
from sklearn.metrics import confusion_matrix


def edit_distance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def get_step_list(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0:
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]


def aligned_print(list, r, h):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.

    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    #print("REF:", end=" ")
    reference_aligned = []
    hypothesis_aligned = []

    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            reference_aligned.append(" ")
            #print(" " * (len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            reference_aligned.append(r[index1])
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            reference_aligned.append(r[index])
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            hypothesis_aligned.append(" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            # Bence: This if was simplified because it is just a space alignement
            hypothesis_aligned.append(h[index2])
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            hypothesis_aligned.append(h[index])
    #print(hypothesis_aligned)
    return reference_aligned, hypothesis_aligned


def align_sequences(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    # build the matrix
    d = edit_distance(r, h)

    # find out the manipulation steps
    manipulations = get_step_list(r, h, d)

    return aligned_print(manipulations, r, h), manipulations


def per_phoneme_per(phoneme: str, reference: list, hypothesis: list, manipulations: list):
    """

    Calculates the per phoneme error rate needs the aligned reference and hypothesis sentences
    :param phoneme:
    :param reference:
    :param hypothesis:
    :param manipulations:
    :return:
    """

    phoneme_idx_ref = [i for i in range(len(reference)) if reference[i] == phoneme]
    phoneme_idx_hyp = [i for i in range(len(hypothesis)) if hypothesis[i] == phoneme]

    n = len(phoneme_idx_ref)

    manipulations_sel = numpy.array(manipulations)[phoneme_idx_ref]
    subs_dels = numpy.sum(numpy.where((manipulations_sel == "s") | (manipulations_sel == "d"), 1, 0))

    manipulations_sel = numpy.array(manipulations)[phoneme_idx_hyp]
    ins = numpy.sum(numpy.where(manipulations_sel == "i", 1, 0))




    # PER: insertion + substitution + deletion / total

    if n == 0:
        raise Exception("Not in hypothesis")
    else:
        result = float(subs_dels + ins) / float(n) * 100

        return result


def afer(phonemes: list, reference: list, hypothesis: list, manipulations: list) -> float:
    """

    Articulatory feature error rate calculation

    :param phonemes:
    :param reference:
    :param hypothesis:
    :param manipulations:
    :return:
    """

    phoneme_idx_ref = [i for i in range(len(reference)) if reference[i] in phonemes]
    phoneme_idx_hyp = [i for i in range(len(hypothesis)) if hypothesis[i] in phonemes]

    n = len(phoneme_idx_ref)

    shared_idx = list(set(phoneme_idx_hyp).intersection(set(phoneme_idx_ref)))
    manipulations = numpy.array(manipulations)

    # Phonemes deemed substitution errors are reconsidered as correct if they both the reference and the hypothesis
    # phoneme is in the same articulatory feature

    # Note that this works because a shared idx will never be included for an insertion/deletion error except if " " is
    # the phoneme, but it should be used with that
    manipulations[shared_idx] = "e"

    manipulations_sel = numpy.array(manipulations)[phoneme_idx_ref]
    subs_dels = numpy.sum(numpy.where((manipulations_sel == "s") | (manipulations_sel == "d"), 1, 0))

    manipulations_sel = numpy.array(manipulations)[phoneme_idx_hyp]
    ins = numpy.sum(numpy.where(manipulations_sel == "i", 1, 0))


    # AFER: insertion + substitution + deletion / total

    if n == 0:
        raise Exception("Not in hypothesis")
    else:
        result = float(subs_dels + ins) / float(n) * 100

        return result


if __name__ == '__main__':

    r = ["t","p", "ax", "h", "ae","ax"]
    h = ["ax", "k", "ae", "t"]

    alignments, manipulations = align_sequences(r, h)
    reference_aligned, hypothesis_aligned = alignments

    # PER: insertion + delete + substitution / total things to guess
    phoneme = "t"
    #per_phoneme_per(phoneme, reference_aligned, hypothesis_aligned, manipulations)
    # Phoneme Error RATE (per phoneme)
    print(reference_aligned)
    print(hypothesis_aligned)
    result = afer(["t","ax"], reference_aligned, hypothesis_aligned, manipulations)
    print(result)
    # Articulatory feature error RATE (per articulatory feature)


