import itertools
import text
import pandas as pd
import numpy as np
from collections import Counter


class Corpus():

    def __init__(self, location):
        self.location = location
        self.cmudict = text.cmudict.CMUDict("/home/boomkin/repos/mellotron/data/cmu_dictionary")

    @staticmethod
    def arpabet_cleaner(arpabet):
        arpabet_wo_braces = arpabet.replace("{", "")
        arpabet_wo_braces = arpabet_wo_braces.replace("}", "")
        arpabet_split = arpabet_wo_braces.split()

        return arpabet_split

    def get_words_for_sentence(self, words) -> list:
        ref_words = list(itertools.chain(*words))

        # Cleans the word files from "misalignment" stars
        ref_words_cleaned = [word for word in ref_words if word != "***"]

        # Makes it to clean ARPABET
        ref_words_arpabet = [self.arpabet_cleaner(text.get_arpabet(word,self.cmudict)) \
                             for word in ref_words_cleaned if text.get_arpabet(word,self.cmudict) != word]

        ref_arpabet = list(itertools.chain(*ref_words_arpabet))

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

    def get_dataframe(self) -> pd.DataFrame:

        with open(self.location, 'r') as f:
            all_lines = f.readlines()

        ref_words_sentencewise = [line.split()[2:] for line in all_lines if ("ref" in line)]
        hyp_words_sentencewise = [line.split()[2:] for line in all_lines if ("hyp" in line)]

        reference_words = self.get_words_for_sentence(ref_words_sentencewise)
        hypothesis_words = self.get_words_for_sentence(hyp_words_sentencewise)

        # Sentence list with clean words goes in here: not sure what would be a better name, I recognise its confusing
        phoneme, intentions, realisations = self.get_phoneme_and_count_from_sentence(reference_words, hypothesis_words)

        df = self.create_dataframe(phoneme,intentions,realisations)

        return df

    @staticmethod
    def clean_tobi_stress(df):
        # First we drop the linguistic stress features
        df = df.rename(columns={'phoneme': 'ARPAbet'})
        df['ARPAbet'] = df['ARPAbet'].str.lower()

        df["ARPAbet"] = df["ARPAbet"].apply(lambda s: ''.join([i for i in s if not i.isdigit()]))
        return df

    @staticmethod
    def convert_phoneme_to_articulatory(df: pd.DataFrame, conversion_table: pd.DataFrame, feature_list: list):
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

    def get_articulatory_dataframe(self):

        df = self.clean_tobi_stress(self.get_dataframe())

        converter_table = pd.read_csv("PhoneSet_modified.csv")

        converter_table = converter_table[converter_table['ARPAbet'].notna()]
        converter_table = converter_table.fillna(0)

        moa_list = ["Vowel","Plosive","Nasal","Fricative","Affricates","Approximant"]
        poa_list = ["Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"]

        poa_df = self.convert_phoneme_to_articulatory(df, converter_table, poa_list)
        moa_df = self.convert_phoneme_to_articulatory(df, converter_table, moa_list)

        return poa_df, moa_df
