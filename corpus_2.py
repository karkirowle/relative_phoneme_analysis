import itertools
import text
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from per_utils import per_phoneme_per, align_sequences, afer
import os

from typing import List, Tuple


class WERDetails:

    def __init__(self, location: str, skip_calculation=False):
        """
        A class aimed to isolate all functions regarding one WERDetails calculations

        :param location: path to wer_details/per_utt file
        :param skip_calculation: skip the phoneme alignments, will make the class mostly unusable
        """
        self.location = location

        current_dir = os.path.dirname(__file__)
        self.cmudict = text.cmudict.CMUDict(os.path.join(current_dir,"text/cmu_dictionary"))

        # Converter table formatting
        self.converter_table = pd.read_csv(os.path.join(current_dir,"PhoneSet_modified.csv"))
        self.converter_table = self.converter_table[self.converter_table['ARPAbet'].notna()]
        self.converter_table = self.converter_table.fillna(0)

        # We consistently need to use uppercase throughout the code
        self.converter_table = self.converter_table.set_index(self.converter_table["ARPAbet"].str.upper())
        if not skip_calculation:
            self.all_ref_phonemes, self.all_hyp_phonemes, self.manipulations = self.phoneme_alignment()

    def phoneme_alignment(self):
        """
        Extracts the reference and hypothesis phonemes based on the sentences in the Kaldi experiment's WERDetails file
        Then, aligns the reference and hypothesis phonemes using the Levensteihn distance

        Finally, makes sure that all phonemes are represented at least once

        Performing this function enables calculation of confusion matrix and PER, and AFER
        :return:
        """
        ref_sentences, hyp_sentences = self.load_sentences()

        all_ref_phonemes = list()
        all_hyp_phonemes = list()
        all_manipulations = list()

        # Realign everything on the phoneme-level
        for ref_sentence, hyp_sentence in zip(ref_sentences,hyp_sentences):
            clean_ref_sentence = self.clean_non_words(ref_sentence)
            clean_hyp_sentence = self.clean_non_words(hyp_sentence)

            ref_phoneme_list = self.words_to_phoneme(clean_ref_sentence,stress_cleaned=True)
            hyp_phoneme_list = self.words_to_phoneme(clean_hyp_sentence,stress_cleaned=True)
            alignments, manipulations = align_sequences(ref_phoneme_list,hyp_phoneme_list)
            reference_aligned, hypothesis_aligned = alignments

            assert len(reference_aligned) == len(hypothesis_aligned)
            all_ref_phonemes.extend(reference_aligned)
            all_hyp_phonemes.extend(hypothesis_aligned)
            all_manipulations.extend(manipulations)

        # There could be missing letters in hyp/ref sentences, and for the missing letters we should do an extra correct

        hyp_set = set(all_hyp_phonemes)
        ref_set = set(all_ref_phonemes)
        missing_phonemes = hyp_set.symmetric_difference(ref_set)

        for phoneme in missing_phonemes:
            all_ref_phonemes.append(phoneme)
            all_hyp_phonemes.append(phoneme)
            all_manipulations.append("c")

        # Sanity check to make sure that the set of reference phonemes is the same as the set of hyp phonemes
        assert set(all_ref_phonemes) == set(all_hyp_phonemes)

        return all_ref_phonemes, all_hyp_phonemes, all_manipulations

    def raw_confusion_matrix(self) -> Tuple[np.ndarray, list]:
        """
        Calculates the phoneme confusion matrix

        :return:
        """

        labels = sorted(list(set(self.all_ref_phonemes)))
        conf_matrix = confusion_matrix(self.all_ref_phonemes, self.all_hyp_phonemes, labels)

        return conf_matrix, labels

    def moa_confusion_matrix(self) -> Tuple[np.ndarray, list]:
        """
        Obtains confusion matrix for manner of articulation
        :return:
        """

        all_ref_moas = [self.phoneme_to_moa(phoneme) for phoneme in self.all_ref_phonemes]
        all_hyp_moas = [self.phoneme_to_moa(phoneme) for phoneme in self.all_hyp_phonemes]

        labels = sorted(list(set(all_ref_moas)))
        conf_matrix = confusion_matrix(all_ref_moas, all_hyp_moas, labels)

        return conf_matrix, labels

    def poa_confusion_matrix(self) -> Tuple[np.ndarray, list]:

        all_ref_poas = [self.phoneme_to_poa(phoneme) for phoneme in self.all_ref_phonemes]
        all_hyp_poas = [self.phoneme_to_poa(phoneme) for phoneme in self.all_hyp_phonemes]

        labels = sorted(list(set(all_ref_poas)))
        conf_matrix = confusion_matrix(all_ref_poas, all_hyp_poas, labels)

        return conf_matrix, labels

    def per_per_phoneme(self, phoneme: str) -> float:
        """
        Returns per for a given phoneme

        :param phoneme:
        :return:
        """
        assert phoneme in set(self.all_ref_phonemes)
        per = per_phoneme_per(phoneme, self.all_ref_phonemes, self.all_hyp_phonemes, self.manipulations)

        return per

    def afer_per_phoneme(self, phonemes: list) -> float:

        afer_val = afer(phonemes, self.all_ref_phonemes, self.all_hyp_phonemes, self.manipulations)

        return afer_val

    def all_pers(self) -> Tuple[list, List[float]]:
        """
        Returns the PER for each phoneme

        :return:
        """
        labels = sorted(list(set(self.all_ref_phonemes)))
        return labels, [self.per_per_phoneme(phoneme) for phoneme in labels]

    def all_moa_afers(self) -> Tuple[list, List[float]]:
        """
        Calculate Manner of Articulation Articulatory Feature Error Rate
        :return:
        """
        moas = ["Vowel","Plosive","Nasal","Fricative","Affricates","Approximant"]
        return moas, [self.afer_per_phoneme(self.moa_to_phonemes(moa)) for moa in moas]

    def all_poa_afers(self) -> Tuple[list, List[float]]:
        """
        Calculate Place of Articulation Articulatory Feature Error Rate
        :return:
        """
        poas = ["Vowel","Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"]
        return poas, [self.afer_per_phoneme(self.poa_to_phonemes(poa)) for poa in poas]

    def load_sentences(self) -> Tuple[list,list]:
        """
        Loads the references and hypothesis sentences from the WER details file
        :return: two list of sentences
        """

        with open(self.location, 'r') as f:
            all_lines = f.readlines()

        ref_words_sentencewise = [line.split()[2:] for line in all_lines if ("ref" in line)]
        hyp_words_sentencewise = [line.split()[2:] for line in all_lines if ("hyp" in line)]

        return ref_words_sentencewise, hyp_words_sentencewise

    @staticmethod
    def clean_non_words(sentence: list) -> list:
        """
        Clears the WERDetails segments from the *** s
        :param sentence:
        :return:
        """
        words = list(filter(lambda a: a != "***", sentence))

        return words

    @staticmethod
    def arpabet_cleaner(arpabet: str, stress_remove: bool = False) -> List[str]:
        """
        Cleans a string with curly braces and numbers delimited with space to a list of strings containing the ARPABet
        phonemes
        :param arpabet: the ARPAbet string
        :param stress_remove: whether to remove ToBi stress or not
        :return:
        """
        arpabet_wo_braces = arpabet.replace("{", "")
        arpabet_wo_braces = arpabet_wo_braces.replace("}", "")
        arpabet_split = arpabet_wo_braces.split()

        if stress_remove:
            arpabet_split = [arpabet.rstrip('0123456789') for arpabet in arpabet_split]
        return arpabet_split

    def word_to_phoneme(self, word: str, stress_cleaned: bool) -> List[str]:
        return self.arpabet_cleaner(text.get_arpabet(word, self.cmudict), stress_cleaned)

    def words_to_phoneme(self, words: list, stress_cleaned: bool) -> List[str]:
        return list(itertools.chain(*[self.word_to_phoneme(word, stress_cleaned) for word in words
                                      if self.word_in_cmu_dict(word)]))

    def word_in_cmu_dict(self, word) -> bool:
        decoded = text.get_arpabet(word, self.cmudict)
        return "{" in decoded

    def phoneme_to_poa(self, phoneme: str) -> str:
        assert phoneme.upper() == phoneme
        # Edge case for handling insertion/deletion errors
        if phoneme == " ":
            return phoneme

        poas = ["Vowel","Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"]
        poa_filter = self.converter_table.loc[phoneme, poas]
        # If there are multiple, we return the first one
        poa = poa_filter.index[poa_filter == 1].values[0]
        return poa

    def phoneme_to_moa(self, phoneme: str) -> str:

        assert phoneme.upper() == phoneme
        # Edge case for handling insertion/deletion errors
        if phoneme == " ":
            return phoneme
        moas = ["Vowel","Plosive","Nasal","Fricative","Affricates","Approximant"]
        moa_filter = self.converter_table.loc[phoneme, moas]
        # If there are multiple, we return the first one
        moa = moa_filter.index[moa_filter == 1].values[0]

        return moa

    def poa_to_phonemes(self, poa: str) -> list:
        """
        Converts a place of articulation feature to the corresponding list of phonemes

        :param poa:
        :return:
        """
        assert poa in ["Vowel","Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"]

        poa_filter = self.converter_table.loc[:, poa]
        poas = poa_filter.index[poa_filter == 1].values

        return poas

    def moa_to_phonemes(self, moa: str) -> list:
        """
        Converts a manner of articulation feature to the corresponding list of phonemes
        :param moa:
        :return:
        """
        assert moa in ["Vowel","Plosive","Nasal","Fricative","Affricates","Approximant"]

        moa_filter = self.converter_table.loc[:, moa]
        moas = moa_filter.index[moa_filter == 1].values
        return moas


if __name__ == '__main__':

    print("sajt")
    
