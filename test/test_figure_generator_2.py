from unittest import TestCase
from figure_generator_2 import *


class Test(TestCase):
    def test_visualise_confusion_matrices(self):
        experiment_1 = [(np.array([[0, 1], [1, 0]]),
                         ["Keksz", "Sajt"]
                         )]
        experiment_2 = [(np.array([[0, 1], [1, 0]]),
                         ["Keksz", "Sajt"]
                         )]

        experiments = list()
        experiments.append(experiment_1)
        experiments.append(experiment_2)
        names = ["Baseline", "Proposed"]

        figure_name = "unit_test"
        success = visualise_confusion_matrices(experiments, names, figure_name)
        self.assertEqual(success, 1)

    def test_moa_to_phonemes(self):
        phonemes = moa_to_phonemes("Plosive")
        self.assertEqual(sorted(["P","T","K","B","D","G"]),sorted(phonemes))