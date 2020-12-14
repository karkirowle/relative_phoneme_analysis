from unittest import TestCase

from per_utils import *


class Test(TestCase):

    def test_afer(self):

        r = ["t", "p", "ax", "h", "ae", "ax"]
        h = ["ax", "k", "ae", "t"]

        alignments, manipulations = align_sequences(r, h)
        reference_aligned, hypothesis_aligned = alignments

        # t, ax, ax in GT (3)
        # t deleted, ax correct, and ax->t is considered correct in the AFR setting
        self.assertEqual((afer(["t","ax"], reference_aligned, hypothesis_aligned, manipulations)),1/3 * 100)

    def test_afer_2(self):

        r = ["t", "ax", "p"]
        h = ["t", "ax", "p"]

        alignments, manipulations = align_sequences(r, h)
        reference_aligned, hypothesis_aligned = alignments

        # t, ax, ax in GT (3)
        # t deleted, ax correct, and ax->t is considered correct in the AFR setting
        self.assertEqual((afer(["p","t","ax"], reference_aligned, hypothesis_aligned, manipulations)),0)
