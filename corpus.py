import itertools
import text
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional, Tuple


class Corpus():

    def __init__(self, location: str, ops: str):
        self.location = location
        self.ops = ops
        self.cmudict = text.cmudict.CMUDict("./text/cmu_dictionary")

    @staticmethod
    def arpabet_cleaner(arpabet: str, stress_remove: bool = False):
        arpabet_wo_braces = arpabet.replace("{", "")
        arpabet_wo_braces = arpabet_wo_braces.replace("}", "")
        arpabet_split = arpabet_wo_braces.split()

        if stress_remove:
            arpabet_split = [arpabet.rstrip('0123456789') for arpabet in arpabet_split]
        return arpabet_split

    def get_words_for_sentence(self, words: list, star_cleaned: bool = True, stress_cleaned: bool = False) -> list:

        # Flattents the word list
        ref_words = list(itertools.chain(*words))

        # Cleans the word files from starts which are added to mark insertion or deletion errors
        if star_cleaned:
            ref_words_cleaned = [word for word in ref_words if word != "***"]
        else:
            ref_words_cleaned = ref_words

        # Makes it to clean ARPABET, however if returned value is same, we don't have pronounciation except in the case
        # of ***, where we want to keep it for the confusion matrix calculation

        ref_words_arpabet = [self.arpabet_cleaner(text.get_arpabet(word,self.cmudict),stress_cleaned) \
                             for word in ref_words_cleaned \
                             if (text.get_arpabet(word,self.cmudict) != word) | (word == "***")]
        if star_cleaned:
            ref_arpabet = list(itertools.chain(*ref_words_arpabet))
        else:
            ref_arpabet = ref_words_arpabet

        return ref_arpabet



    @staticmethod
    def get_phoneme_and_count_from_sentence(intentions_list, realisations_list):
        """
        Gets phoneme and the number of phonemes for the reference and hypothesis sentences
        It has to see both because there could be non-overlapping phonemes in the two phoneme sets
        """

        intention_counter = Counter(intentions_list)
        realisation_counter = Counter(realisations_list)

        phonemes_1 = [phoneme for phoneme in intention_counter.keys()]
        phonemes_2 = [phoneme for phoneme in realisation_counter.keys()]

        # We have to convert back to list because pandas doesn't like it
        phonemes = list(set(phonemes_1).union(phonemes_2))

        intentions = [intention_counter[phoneme] for phoneme in phonemes]
        realisations = [realisation_counter[phoneme] for phoneme in phonemes]
        return phonemes, intentions, realisations

    def create_dataframe(self, phoneme, intentions, realisations):
        d = {"phoneme": phoneme, "intentions": intentions, "realisations": realisations}
        df = pd.DataFrame(data=d)

        df["diff"] = df["intentions"] - df["realisations"]
        df["norm"] = np.round(df["diff"] / df["intentions"], decimals=2)

        return df

    @staticmethod
    def get_recording_id_of_utterance(utterance: str) -> str:
        speaker_id = utterance[2:4]
        return speaker_id

    def extract_new_speakers(self, speaker_id_list: Optional[list], all_lines: list):

        if speaker_id_list is not None:
            new_speakers = [self.get_recording_id_of_utterance(line) for line in all_lines if ("ref" in line) &
                            (self.get_recording_id_of_utterance(line) not in speaker_id_list)]
        else:
            new_speakers = [self.get_recording_id_of_utterance(line) for line in all_lines if ("ref" in line)]

        new_speakers = set(new_speakers)
        return new_speakers

    def get_phoneme_set(self, word_list: list) -> list:
        """
        Takes an unflattened arpabetised word list of lists and returns all the phoneme occurences

        :param word_list: unflattened word list
        :return:
        """
        word_list = list(itertools.chain(*word_list))
        word_list = list(set(word_list))
        return word_list

    def confusion_dataframe(self, reference_words, hypothesis_words):

        phoneme_set_1 = self.get_phoneme_set(reference_words)
        phoneme_set_2 = self.get_phoneme_set(hypothesis_words)

        # Create a confusion matrix dataframe, adding insertion and deletion error column
        phoneme_set_1.remove("***")
        phoneme_set_2.remove("***")
        phoneme_set_1 = sorted(phoneme_set_1)
        phoneme_set_2 = sorted(phoneme_set_2)
        phoneme_set_1.extend(["insertion"])
        phoneme_set_2.extend(["deletion"])

        confusion_matrix = pd.DataFrame(0, index=phoneme_set_1, columns=phoneme_set_2)
        with open(self.ops, 'r') as f:
            all_lines = f.readlines()

        for line in all_lines:
            error_type, reference, hypothesis, occurence = line.split()

            # Skipping condition: the skipping condition is without numbers because it can happen that erroneously
            # a number is left in the hypothesis side. The stress mark remover will return an empty string in that case
            # because it cannot be found in the dictionary and it's not checked for that problem
            if (self.arpabet_cleaner(text.get_arpabet(reference, self.cmudict))[0] == reference)  \
                    & (reference != "***"):
                continue
            if (self.arpabet_cleaner(text.get_arpabet(hypothesis, self.cmudict))[0] == hypothesis) \
                    & (hypothesis != "***"):
                continue

            occurence = int(occurence)
            if error_type == "correct":
                # In the case of correct, we just add the right phonemes to the diagonal
                phonemes = self.arpabet_cleaner(text.get_arpabet(reference, self.cmudict), True)

                for phoneme in phonemes:
                    confusion_matrix.loc[[phoneme], [phoneme]] += occurence

            elif error_type == "substitution":
                # Substitution error is the most difficult case, except if the lengths match, then we just insert into
                # the right column row

                phonemes_ref = self.arpabet_cleaner(text.get_arpabet(reference, self.cmudict), True)
                phonemes_hyp = self.arpabet_cleaner(text.get_arpabet(hypothesis, self.cmudict), True)

                # We should enumerate from the perspective of the longer

                if len(phonemes_ref) >= len(phonemes_hyp):
                    for i,phoneme_ref in enumerate(phonemes_ref):
                        if i >= len(phonemes_hyp):
                            confusion_matrix.loc[[phoneme_ref], ["deletion"]] += occurence
                        else:
                            confusion_matrix.loc[[phoneme_ref], [phonemes_hyp[i]]] += occurence
                else:
                    for i, phoneme_hyp in enumerate(phonemes_hyp):
                        if i >= len(phonemes_ref):
                            confusion_matrix.loc[["insertion"], [phoneme_hyp]] += occurence
                        else:
                            confusion_matrix.loc[[phonemes_ref[i]], [phoneme_hyp]] += occurence

            elif error_type == "insertion":
                # Insertion error means that the hypothesis needs to be decoded
                phonemes = self.arpabet_cleaner(text.get_arpabet(hypothesis, self.cmudict),True)

                for phoneme in phonemes:
                    confusion_matrix.loc[["insertion"], [phoneme]] += occurence

            elif error_type == "deletion":
                # Insertion error means that the hypothesis needs to be decoded
                phonemes = self.arpabet_cleaner(text.get_arpabet(reference, self.cmudict),True)

                for phoneme in phonemes:
                    confusion_matrix.loc[[phoneme], ["deletion"]] += occurence

        return confusion_matrix

    def get_confusion_matrix(self) -> pd.DataFrame:

        with open(self.location, 'r') as f:
            all_lines = f.readlines()

        ref_words_sentencewise = [line.split()[2:] for line in all_lines if ("ref" in line)]
        hyp_words_sentencewise = [line.split()[2:] for line in all_lines if ("hyp" in line)]

        reference_words = self.get_words_for_sentence(ref_words_sentencewise,star_cleaned=False,stress_cleaned=True)
        hypothesis_words = self.get_words_for_sentence(hyp_words_sentencewise,star_cleaned=False,stress_cleaned=True)

        confusion_dataframe = self.confusion_dataframe(reference_words,hypothesis_words)

        return confusion_dataframe

    def get_dataframe(self, recordings_to_exclude: Optional[list] = None, no_stress: bool = False) -> pd.DataFrame:
        """
        Extract the phoneme dataframes based on the Kaldi experiments wer_detail and per_utt
        :param recordings_to_exclude: ID of recordings to NOT include
        :param no_stress: if True, ignores linguistic stress
        :return:
        """

        with open(self.location, 'r') as f:
            all_lines = f.readlines()

        if recordings_to_exclude is None:
            ref_words_sentencewise = [line.split()[2:] for line in all_lines if ("ref" in line)]
            hyp_words_sentencewise = [line.split()[2:] for line in all_lines if ("hyp" in line)]
        else:
            ref_words_sentencewise = [line.split()[2:] for line in all_lines if ("ref" in line) &
                                      (self.get_recording_id_of_utterance(line) not in recordings_to_exclude)]
            hyp_words_sentencewise = [line.split()[2:] for line in all_lines if ("hyp" in line) &
                                      (self.get_recording_id_of_utterance(line) not in recordings_to_exclude)]

        reference_words = self.get_words_for_sentence(ref_words_sentencewise)
        hypothesis_words = self.get_words_for_sentence(hyp_words_sentencewise)

        # Sentence list with clean words goes in here: not sure what would be a better name, I recognise its confusing
        phoneme, intentions, realisations = self.get_phoneme_and_count_from_sentence(reference_words, hypothesis_words)

        df = self.create_dataframe(phoneme,intentions,realisations)

        if no_stress:
            df = self.clean_tobi_stress(df)
            df = df.rename(columns={'ARPAbet': 'phoneme'})

        new_speakers = self.extract_new_speakers(recordings_to_exclude, all_lines)

        return df, new_speakers

    @staticmethod
    def clean_tobi_stress(df: pd.DataFrame):
        # First we drop the linguistic stress features
        df = df.rename(columns={'phoneme': 'ARPAbet'})

        df['ARPAbet'] = df['ARPAbet'].str.lower()

        # Reconcatenating non-digit characters
        df["ARPAbet"] = df["ARPAbet"].apply(lambda s: ''.join([i for i in s if not i.isdigit()]))

        # Finally, duplicate rows have to be summed
        df = df.groupby("ARPAbet").sum().reset_index()
        return df

    @staticmethod
    def convert_phoneme_to_articulatory(df: pd.DataFrame, conversion_table: pd.DataFrame, feature_list: list):
        """
        Converts a phoneme count dataframe to an articulatory feature count dataframe
        :param df: phoneme count dataframe in ARPAbet format
        :param conversion_table: the conversion table including the mappings to the articulory features
        :param feature_list: the list of articulatory features (Plosives, etc.) to be included

        :return: the articulatory dataframe
        """
        # Select relevant columns for the conversion and merge them in a bigger table
        columns_to_select = ["ARPAbet"] + feature_list
        conversion_table = conversion_table[columns_to_select]
        conversion_table = pd.merge(df, conversion_table, how="left", on=['ARPAbet'])

        articulatory_df = pd.DataFrame(columns=["intentions", "realisations"], index=feature_list)
        num_features = len(feature_list)

        for i in range(num_features):
            feature = feature_list[i]
            matching_phonemes = conversion_table[conversion_table[feature] == 1]

            # Adds new row to articulatory df which groups phones by features and then sums them
            articulatory_df.iloc[i,:] = matching_phonemes.groupby(feature)[
                ["intentions", "realisations"]].sum().values

        articulatory_df["diff"] = articulatory_df["intentions"] - articulatory_df["realisations"]
        articulatory_df["norm"] = articulatory_df["diff"] / articulatory_df["intentions"]

        return articulatory_df

    def convert_phoneme_df_to_articulatory_df(self, phoneme_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts an ARPAbet phoneme-based table to an articulatory dataframe

        :param phoneme_df:
        :return:
        """

        converter_table = pd.read_csv("PhoneSet_modified.csv")

        converter_table = converter_table[converter_table['ARPAbet'].notna()]
        converter_table = converter_table.fillna(0)

        moa_list = ["Vowel","Plosive","Nasal","Fricative","Affricates","Approximant"]
        poa_list = ["Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"]

        poa_df = self.convert_phoneme_to_articulatory(phoneme_df, converter_table, poa_list)
        moa_df = self.convert_phoneme_to_articulatory(phoneme_df, converter_table, moa_list)

        return poa_df, moa_df

    def get_articulatory_dataframe(self, recordings_to_exclude: Optional[list] = None):

        df, new_speakers = self.get_dataframe(recordings_to_exclude=recordings_to_exclude)
        df = self.clean_tobi_stress(df)

        poa_df, moa_df = self.convert_phoneme_df_to_articulatory_df(df)

        return poa_df, moa_df, new_speakers

    def articulatory_confusion_sub(self, moa_list: list, converter_table: pd.DataFrame, confusion_frame: pd.DataFrame) -> pd.DataFrame:

        moa_df = pd.DataFrame(columns=moa_list, index=moa_list)

        for moa_a in moa_list:
            a_phonemes = list(converter_table[converter_table[moa_a] == 1]["ARPAbet"].values)
            a_phonemes = [phoneme for phoneme in a_phonemes if phoneme in confusion_frame.columns]
            for moa_b in moa_list:
                b_phonemes = list(converter_table[converter_table[moa_b] == 1]["ARPAbet"].values)
                b_phonemes = [phoneme for phoneme in b_phonemes if phoneme in confusion_frame.columns]

                moa_df.loc[moa_a, moa_b] = confusion_frame.loc[a_phonemes, b_phonemes].sum(axis=1).sum(axis=0)

        return moa_df

    def get_articulatory_confusion(self) -> Tuple[pd.DataFrame,pd.DataFrame]:

        confusion_frame = self.get_confusion_matrix()

        converter_table = pd.read_csv("PhoneSet_modified.csv")
        converter_table = converter_table[converter_table['ARPAbet'].notna()]
        converter_table = converter_table.fillna(0)
        converter_table['ARPAbet'] = converter_table['ARPAbet'].str.upper()

        moa_list = ["Vowel","Plosive","Nasal","Fricative","Affricates","Approximant"]
        poa_list = ["Vowel","Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"]

        moa_df = self.articulatory_confusion_sub(moa_list, converter_table, confusion_frame)
        poa_df = self.articulatory_confusion_sub(poa_list, converter_table, confusion_frame)

        return poa_df, moa_df