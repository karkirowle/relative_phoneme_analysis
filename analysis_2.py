from corpus import Corpus
from figure_generator import phoneme_barplot, articulatory_barplot

corpus_1 = Corpus("/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_44.32_train/wer_details/per_utt")
corpus_2 = Corpus("/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_49.04_fmllr_train/wer_details/per_utt")

poa_1, moa_1 = corpus_1.get_articulatory_dataframe()
poa_2, moa_2 = corpus_2.get_articulatory_dataframe()


phoneme_1 = corpus_1.get_dataframe()
phoneme_2 = corpus_2.get_dataframe()




phoneme_barplot([phoneme_1,phoneme_2],phoneme_2,["arch1","arch2"],500)
articulatory_barplot([moa_1,moa_2],["arch1","arch2"],"sajt")