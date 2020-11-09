import pandas as pd
import numpy as np
import text
import text.cmudict
from collections import Counter
import itertools

import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def read_words_to_tuple(txt):

    df = pd.read_csv(txt,delim_whitespace=True,header=None,names=["errortype","word1","word2","count"])

    df_sub = df.loc[df['errortype'] == "substitution"]

    return df_sub["word1"].values,df_sub["word2"].values,df_sub["count"].values


def arpabet_cleaner(arpabet):

    arpabet_wo_braces = arpabet.replace("{","")
    arpabet_wo_braces = arpabet_wo_braces.replace("}","")
    arpabet_split = arpabet_wo_braces.split()

    return arpabet_split

def articulatory_plotter(df_list: list, label_list: list, filename: str):


    x = np.arange(len(df_list[0]))

    fig_fontsize = 6 # 7
    width = 0.13
    steps= len(df_list)
    # TODO: This logic works for this case only :( Probably if you do odd = width*2 it works for extended cases
    width_logic = np.linspace(start=-width*2,stop=width,num=steps)


    xdim = 3.14
    ydim = 3.14 * 0.62
    fig =plt.figure(num=None, figsize=(xdim, ydim), dpi=100, facecolor='w', edgecolor='k')
    #plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    for i in range(len(df_list)):
        plt.bar(x + width/2 + width_logic[i],df_list[i]["norm"],width,label=label_list[i])


    #plt.xticks(rotation=45)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed',which="major")


    ax.axhline(y=0, color="black",linewidth=0.8)
    ax.set_xticks(x)

    ax.set_xticklabels(df_list[0].index.values + " (" + df_list[0]["intentions"].astype(str) + ")", rotation=45, fontsize=fig_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fig_fontsize, pad=1)

    plt.ylabel("Relative articulatory error",fontsize=fig_fontsize,labelpad=-4)
    lg = plt.legend(fontsize=fig_fontsize)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    fig.tight_layout(pad=0)
    fig.set_size_inches(xdim, ydim)
    plt.savefig("figures/" + filename + ".pdf",bbox_inches='tight',pad_inches = 0.005)



def articulatory_features(df: pd.DataFrame):
    # First we drop the linguistic stress features
    df = df.rename(columns={'phoneme': 'ARPAbet'})
    df['ARPAbet'] = df['ARPAbet'].str.lower()
    df["ARPAbet"] = df["ARPAbet"].apply(lambda s: ''.join([i for i in s if not i.isdigit()]))

    converter_table = pd.read_csv("PhoneSet_modified.csv")

    converter_table = converter_table[converter_table['ARPAbet'].notna()]
    converter_table = converter_table.fillna(0)







    # Plosive?
    moa_list = ["ARPAbet","Vowel","Plosive","Nasal","Fricative","Affricates","Approximant"]
    poa_list = ["ARPAbet","Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"]
    manner_of_articulation = converter_table[moa_list]
    place_of_articulation = converter_table[poa_list]

    # We want to get a table for each, let's start with voicing because that's straighforward

    def something_that_is_terrible(df: pd.DataFrame, pma: pd.DataFrame, p_list: list):
        result = pd.merge(df, pma, how="left", on=['ARPAbet'])
        v_df = pd.DataFrame(columns=["intentions", "realisations"], index=p_list[1:])
        for i in range(1, len(p_list)):
            temp = result[result[p_list[i]] == 1]
            v_df.iloc[i-1,:] = temp.groupby(p_list[i])[["intentions", "realisations"]].sum().values


        v_df["diff"] = v_df["intentions"] - v_df["realisations"]
        v_df["norm"] = v_df["diff"] / v_df["intentions"]

        return v_df

    poa_df = something_that_is_terrible(df, place_of_articulation, poa_list)
    moa_df = something_that_is_terrible(df, manner_of_articulation, moa_list)

    return poa_df, moa_df





def corpus_plotter(df_list: list, legend_list: list):

    # There is a combination of pointer complexity and alignment here so let me explain


    # The first line is a pass by value. We only need the referenc_df for the filtering (i.e) we know what ot filter
    # in other arrays
    reference_df = df_list[3]
    df_list[3] = df_list[3][(df_list[3]["intentions"] >= 500) & (df_list[3]["phoneme"] != "HH")].sort_values(by=["norm"],
                                                                                                 ascending=False)

    for i in [0,1,2]:
        df = df_list[i]

        df = df[reference_df["intentions"] >= 500]

        df = df.set_index("phoneme")
        df = df.reindex(index=df_list[3]['phoneme'])
        #df = df.reindex(index=df_list[3]['phoneme'])
        df = df.reset_index()
        df_list[i] = df

    print("hosszok")
    [print(len(item)) for item in df_list]
    x = np.arange(len(df_list[3]))
    width = 0.20
    steps = len(df_list)
    width_logic = np.linspace(start=-width*2,stop=width,num=steps)

    width_fig = 6.6 # 6.29
    height_fig = 3.14 * 0.60
    fig = plt.figure(num=None, figsize=(width_fig,height_fig), dpi=100, facecolor='w', edgecolor='k')
    for i in range(len(df_list)):
        df = df_list[i]
        print(legend_list[i])
        plt.bar(x + width/2 + width_logic[i],df["norm"],width,label=legend_list[i])

    plt.grid()
    ax = plt.gca()

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)

    ax.set_xticklabels(["/" + x.lower() + "/" for x in df_list[3]["phoneme"].values], rotation=45,
                       fontsize=7)


    ax.tick_params(axis='both', which='major', labelsize=7)

    plt.xlim([-0.6,len(df_list[3])-0.4])

    plt.ylabel("Relative phoneme error", fontsize=7,labelpad=-4)
    lg = plt.legend(fontsize=7)
    plt.tight_layout()
    fig.tight_layout()
    fig.set_size_inches(width_fig, height_fig)
    plt.savefig("figures/phoneme_error.pdf",bbox_inches='tight',pad_inches = 0.005)
    #plt.show()

def full_corpus(location: str) -> pd.DataFrame:

    #location="/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_49.04_fmllr_train/wer_details/per_utt"
    cmudict = text.cmudict.CMUDict("/home/boomkin/repos/mellotron/data/cmu_dictionary")

    with open(location, 'r') as f:
        all_lines = f.readlines()

    ref_words_sentence_wise = [line.split()[2:] for line in all_lines if ("ref" in line)]
    hyp_words_sentence_wise = [line.split()[2:] for line in all_lines if ("hyp" in line)]

    ref_words = list(itertools.chain(*ref_words_sentence_wise))
    hyp_words = list(itertools.chain(*hyp_words_sentence_wise))
    ref_words_cleaned = [word for word in ref_words if word != "***"]
    hyp_words_cleaned = [word for word in hyp_words if word != "***"]

    ref_words_arpabet = [arpabet_cleaner(text.get_arpabet(word,cmudict)) for word in ref_words_cleaned if text.get_arpabet(word,cmudict) != word]
    hyp_words_arpabet = [arpabet_cleaner(text.get_arpabet(word,cmudict)) for word in hyp_words_cleaned if text.get_arpabet(word,cmudict) != word]

    ref_arpabet = list(itertools.chain(*ref_words_arpabet))
    hyp_arpabet = list(itertools.chain(*hyp_words_arpabet))

    intention_counter = Counter(ref_arpabet)
    realisation_counter = Counter(hyp_arpabet)

    keys = [key for key in intention_counter.keys()]
    intentions = [intention_counter[key] for key in keys]
    realisations = [realisation_counter[key] for key in keys]


    d = {"phoneme": keys, "intentions": intentions, "realisations": realisations}
    df = pd.DataFrame(data=d)

    df["diff"] = df["intentions"] - df["realisations"]
    df["norm"] = np.round(df["diff"] / df["intentions"],decimals=2)

    return df
    #df.to_csv("error_analysis_train_fmllr.csv")

def figure_producer():
    filename = "backup/error_analysis_train.csv"
    word1,word2,count = read_words_to_tuple("/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_44.32_train/wer_details/ops")
    cmudict = text.cmudict.CMUDict("/home/boomkin/repos/mellotron/data/cmu_dictionary")

    exclusion_count = 0
    arpabet_intention_list = []
    arpabet_realisation_list = []

    for i in range(len(word1)):
        #print(word2[i])

        arpabet_word1 = text.get_arpabet(word1[i],cmudict)
        arpabet_word2 = text.get_arpabet(word2[i],cmudict)
        if (arpabet_word1 == word1[i]) or (arpabet_word2 == word2[i]):
            exclusion_count += count[i]
        else:
            arpabet_intention_list.append(arpabet_cleaner(arpabet_word1) * count[i])
            arpabet_realisation_list.append(arpabet_cleaner(arpabet_word2) * count[i])

    arpabet_intention_list = list(itertools.chain(*arpabet_intention_list))
    arpabet_realisation_list = list(itertools.chain(*arpabet_realisation_list))

    intention_counter = Counter(arpabet_intention_list)
    realisation_counter = Counter(arpabet_realisation_list)

    keys = [key for key in intention_counter.keys()]
    print(keys)
    intentions = [intention_counter[key] for key in keys]
    realisations = [realisation_counter[key] for key in keys]
    d = {"phoneme": keys, "intentions": intentions, "realisations": realisations}
    df = pd.DataFrame(data=d)

    df.to_csv(filename)
    #for key in intention_counter.keys():
    #    print(key,end="\t")
    #    print(intention_counter[key],end="\t")
    #    print(realisation_counter[key])


    #print(Counter(arpabet_intention_list).keys())
    #print(Counter(arpabet_intention_list).values())
    #print(Counter(arpabet_realisation_list).keys())
    #print(Counter(arpabet_realisation_list).values())
    ratio_excluded = exclusion_count / np.sum(count)
    print(ratio_excluded)


def top_word_error(paths,top=5):

    for i,path in enumerate(paths):
        df = pd.read_csv(path,delim_whitespace=True,names=["errortype","word1","word2","count_" + str(i)])
        #names=["type","word","word","num"])
        df = df[df["errortype"] == "substitution"][:top]
        print(df)

def summary_of_error_types(paths):

    for path in paths:
        df = pd.read_csv(path,delim_whitespace=True,header=None)

        print(path)
        print("correct: ", len(df[df[0] == "correct"]))
        print("substitution: ", len(df[df[0] == "substitution"]))
        print("insertion: ", len(df[df[0] == "deletion"]))
        print("deletion: ", len(df[df[0] == "insertion"]))




if __name__ == '__main__':

    # Top word error analysis
    path=["/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_44.32_train/wer_details/ops",
          "/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_49.04_fmllr_train/wer_details/ops",
          "/home/boomkin/Downloads/second_last_asr/scoring_kaldi_53.62_fbank_pitch_train/wer_details/ops",
          "/home/boomkin/Downloads/second_last_asr/scoring_kaldi_78.57_baseline_not_trained_on_oralcancer_train/wer_details/ops"]

    top_word_error(path)


    location_1="/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_44.32_train/wer_details/per_utt"
    df_1 = full_corpus(location_1)
    poa_1, moa_1 = articulatory_features(df_1)
    location_2 = "/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_49.04_fmllr_train/wer_details/per_utt"
    df_2 = full_corpus(location_2)
    poa_2, moa_2 = articulatory_features(df_2)
    location_3 = "/home/boomkin/Downloads/second_last_asr/scoring_kaldi_53.62_fbank_pitch_train/wer_details/per_utt"
    df_3 = full_corpus(location_3)
    poa_3, moa_3 = articulatory_features(df_3)
    location_4 = "/home/boomkin/Downloads/second_last_asr/scoring_kaldi_78.57_baseline_not_trained_on_oralcancer_train/wer_details/per_utt"
    df_4 = full_corpus(location_4)
    poa_4, moa_4 = articulatory_features(df_4)
    #corpus_plotter([df_2,df_3,df_1,df_4],["fMLLR", "baseline 2", "baseline 1", "w/o retraining"])
    corpus_plotter([df_4,df_1,df_3,df_2],["Baseline", "DNN AM retraining", "Baseline + OC", "fMLLR for AM retraining"])

    #articulatory_plotter([poa_1, poa_2,poa_3,poa_4],["AM retrained","fmllr retrained","pitch baseline retrained(?)","baseline"],"poa_train")
    #articulatory_plotter([moa_1, moa_2,moa_3,moa_4],["AM retrained","fmllr retrained","pitch baseline","baseline"],"moa_train")


    location_1="/home/boomkin/Downloads/second_last_asr/scoring_kaldi_54.13_baseline_not_trained_on_oralcancer_test/wer_details/per_utt"
    df_1 = full_corpus(location_1)
    df_1.to_csv("error_analysis_baseline.csv")
    poa_1, moa_1 = articulatory_features(df_1)
    location_2 = "/home/boomkin/Downloads/second_last_asr/scoring_kaldi_49.95_fbank_pitch_test/wer_details/per_utt"
    df_2 = full_corpus(location_2)
    df_2.to_csv("error_analysis_baseline_oc.csv")
    poa_2, moa_2 = articulatory_features(df_2)
    location_3 = "/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_48.55_test/wer_details/per_utt"
    df_2 = full_corpus(location_3)
    df_2.to_csv("error_analysis_dnn_am_retraining.csv")
    poa_3, moa_3 = articulatory_features(df_2)
    location_4 = "/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_44.15_fmllr_test/wer_details/per_utt"
    df_2 = full_corpus(location_4)
    df_2.to_csv("error_analysis_fmllr.csv")

    poa_4, moa_4 = articulatory_features(df_2)
    #
    articulatory_plotter([poa_1, poa_3, poa_2, poa_4],["Baseline", "DNN AM retraining", "Baseline + OC", "fMLLR for AM retraining"],"poa_0429")
    articulatory_plotter([moa_1, moa_3, moa_2, moa_4],["Baseline", "DNN AM retraining", "Baseline + OC", "fMLLR for AM retraining"],"moa_0429")
    #
    paths = [str.replace(location_1,"per_utt","ops"),
             str.replace(location_2,"per_utt","ops"),
             str.replace(location_3,"per_utt","ops"),
             str.replace(location_4,"per_utt","ops")]

    summary_of_error_types(paths)
    #articulatory_plotter([v_1, v_2, v_3, v_4], ["not retrained", "pitch retrained", "AM retrained", "fMLLR"])
    #corpus_plotter(df_1,df_2)
    #figure_producer()