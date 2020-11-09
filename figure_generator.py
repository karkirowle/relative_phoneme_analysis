import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# Sets to a nicer font type
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def phoneme_barplot(df_list: list, reference_df: pd.DataFrame, legend_list: list, number_of_occurence: int):

    """
    Creates a bar plot with the phonemes occuring in the dataset and their intentions vs realisations

    :param df_list:
    :param reference_df:
    :param legend_list: the legend list should contain the identifier of the experiment, should be the same size as df_list
    :param number_of_occurence:
    :return:
    """

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.60,
        "dpi": 100,
        "width": 0.20, # the width of the bars
        "fontsize": 7
    }

    assert len(df_list) == len(legend_list)

    # More than number of occcurences intensions + sorted by decreasing relative phoneme error rate
    reference_df_common = reference_df[reference_df["intentions"] >= number_of_occurence]
    reference_df_phoneme = reference_df_common.sort_values(by=["norm"],ascending=False)["phoneme"]

    # The different dataframes can have different phoneme sets, so we use the reference phonemes to filter them
    filtered_df_list = list()

    for df in df_list:

        df = df[df["phoneme"].isin(reference_df_phoneme)]
        df = df.set_index("phoneme")
        df = df.reindex(index=reference_df_phoneme)
        df = df.reset_index()
        filtered_df_list.append(df)

    num_phonemes = len(reference_df_phoneme)
    num_architectures = len(filtered_df_list)



    fig = plt.figure(num=None, figsize=(format_dict["width_fig"],format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    # Each architecture is plotted separately
    for i in range(len(filtered_df_list)):
        df = filtered_df_list[i]

        # For each architecture we position the barplots so there is equal amount if the number of categories are
        # equal, otherwise we put the extra on the right
        x = np.arange(num_phonemes)

        left_max = num_architectures // 2
        right_max = np.ceil(num_architectures / 2)
        width_cp = format_dict["width"]/ 2

        width_logic = np.linspace(start=-(width_cp) * left_max, stop=(width_cp) * right_max, num=num_architectures) \
                      + format_dict["width"] / 2

        plt.bar(x + width_logic[i],df["norm"],format_dict["width"],label=legend_list[i])

    plt.grid()
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)

    ax.set_xticklabels(["/" + x.lower() + "/" for x in reference_df["phoneme"].values], rotation=45, fontsize=format_dict["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=format_dict["fontsize"])

    #plt.xlim([-0.6,len(reference_df)-0.4])

    plt.ylabel("Relative phoneme error", fontsize=format_dict["fontsize"],labelpad=-4)
    plt.legend(fontsize=format_dict["fontsize"])
    plt.tight_layout()
    fig.tight_layout()
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    plt.show()
    #plt.savefig("figures/phoneme_error_2.pdf",bbox_inches='tight',pad_inches = 0.005)


def articulatory_barplot(df_list: list, label_list: list, filename: str):

    format_dict = {
        "width_fig": 3.14,
        "height_fig": 3.14 * 0.62,
        "dpi": 100,
        "width": 0.13, # the width of the bars
        "fontsize": 7
    }

    x = np.arange(len(df_list[0]))

    fig_fontsize = 6 # 7
    num_architectures = len(df_list)
    # TODO: This logic works for this case only :( Probably if you do odd = width*2 it works for extended cases
    width_logic = np.linspace(start=-format_dict["width"]*2,stop=format_dict["width"],num=num_architectures)

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                    dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    for i in range(num_architectures):
        plt.bar(x + format_dict["width"]/2 + width_logic[i],df_list[i]["norm"],format_dict["width"],label=label_list[i])

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed',which="major")

    ax.axhline(y=0, color="black",linewidth=0.8)
    ax.set_xticks(x)

    ax.set_xticklabels(df_list[0].index.values + " (" + df_list[0]["intentions"].astype(str) + ")", rotation=45, fontsize=fig_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fig_fontsize, pad=1)

    plt.ylabel("Relative articulatory error",fontsize=fig_fontsize)
    plt.legend(fontsize=fig_fontsize)
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    fig.tight_layout(pad=0)
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])
    plt.show()
    #plt.savefig("figures/" + filename + ".pdf",bbox_inches='tight',pad_inches = 0.005)
