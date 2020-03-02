import pandas as pd

def read_words_to_tuple(txt):

    df = pd.read_csv(txt,delim_whitespace=True,header=None,names=["errortype","word1","word2","count"])

    df_sub = df.loc[df['errortype'] == "substitution"]


    return df_sub["word1"].values,df_sub["count"].values

def figure_producer():

    word, count = read_words_to_tuple("/home/boomkin/Downloads/last_ASR/TUD_male_fem/scoring_kaldi_44.32_train/wer_details/ops")

    for i in range(len(word)):
        print(word[i])
        print(count[i])

if __name__ == '__main__':
    figure_producer()