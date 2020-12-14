import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# Sets to a nicer font type
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def phoneme_barplot(df_list: list, reference_df: pd.DataFrame, legend_list: list, number_of_occurence: int, single_fold: bool,
                    top_n_phonemes: int):

    """
    Creates a bar plot with the phonemes occuring in the dataset and their intentions vs realisations

    :param df_list:
    :param reference_df:
    :param legend_list: the legend list should contain the identifier of the experiment, should be the same size as df_list
    :param number_of_occurence: filters the phoneme list by the number of (mean) occurence of a phoneme
    :param top_n_phonemes: an additional parameter applied on top of number of (mean) occurences to show top n phonemes
    :return:
    """

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.60,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    assert len(df_list) == len(legend_list)

    # More than number of occcurences intensions + sorted by decreasing relative phoneme error rate

    if single_fold:
        reference_df_common = reference_df[reference_df["intentions"] >= number_of_occurence]
        reference_df_phoneme = reference_df_common.sort_values(by=["norm"], ascending=False)["phoneme"]
    else:
        reference_df_common = reference_df[reference_df["mean"] >= number_of_occurence]
        reference_df_phoneme = reference_df_common.sort_values(by=["norm_mean"],ascending=False)["phoneme"]

    print("In total ", len(reference_df_phoneme), " phonemes and ", top_n_phonemes/len(reference_df_phoneme)*100, "% is shown")
    reference_df_phoneme = reference_df_phoneme[:top_n_phonemes]

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
        x = np.arange(num_phonemes*2.5,step=2.5)

        left_max = num_architectures // 2
        right_max = np.ceil(num_architectures / 2)
        width_cp = format_dict["width"] / 2


        width_logic = np.arange(start= -format_dict["width"] * left_max, stop=format_dict["width"] * right_max,
                                step=format_dict["width"]) + width_cp

        legend_props = {"elinewidth": 0.5}

        if single_fold:
            plt.bar(x + width_logic[i],
                    df["norm"],
                    format_dict["width"],
                    label=legend_list[i],
                    capsize=format_dict["capsize"],
                    error_kw=legend_props)
        else:
            plt.bar(x + width_logic[i],
                    df["norm_mean"],
                    format_dict["width"],
                    label=legend_list[i],
                    capsize=format_dict["capsize"],
                    yerr=df["norm_std"],
                    error_kw=legend_props)

    plt.grid()
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)

    ax.set_xticklabels(["/" + x.lower() + "/" for x in reference_df_phoneme.values], rotation=45, fontsize=format_dict["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=format_dict["fontsize"])

    # plt.ylim([-0.2,0.6])
    plt.ylabel("RPER", fontsize=format_dict["fontsize"])

    plt.legend(fontsize=format_dict["legend_fontsize"],
               loc='upper center', bbox_to_anchor=(0.5, +1.3),
               fancybox=True, shadow=True, ncol=num_architectures)

    plt.tight_layout()
    fig.tight_layout()
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    plt.savefig("figures/phoneme_error_2020_11_10.pdf",bbox_inches='tight',pad_inches = 0.005)


def articulatory_barplot(df_list: list, label_list: list, filename: str, single_fold: bool, poa: bool):

    format_dict = {
        "width_fig": 3.14,
        "height_fig": 3.14 * 0.8, # 0.62
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
        legend_props = {"elinewidth": 0.5}
        if single_fold:
            plt.bar(x + format_dict["width"]/2 + width_logic[i],
                    df_list[i]["norm"],format_dict["width"],
                    label=label_list[i],
                    capsize=1.0, error_kw=legend_props)
        else:
            plt.bar(x + format_dict["width"]/2 + width_logic[i],
                    df_list[i]["norm_mean"],format_dict["width"],
                    label=label_list[i],yerr=df_list[i]["norm_std"],
                    capsize=1.0, error_kw=legend_props)

    #plt.ylim([-0.5,0.5])
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed',which="major")
    if poa:
        ax.set_yticks([-0.8, -0.6,-0.4,-0.2,+0.2,0.4,0.6])
    ax.axhline(y=0, color="black",linewidth=0.8)
    ax.set_xticks(x)

    if single_fold:
        ax.set_xticklabels(df_list[0]["phoneme"] + " (" + df_list[0]["intentions"].astype(str) + ")", rotation=45, fontsize=fig_fontsize)
    else:
        ax.set_xticklabels(df_list[0]["phoneme"] + " (" + df_list[0]["mean"].astype(str) + ")", rotation=45, fontsize=fig_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fig_fontsize, pad=1)

    plt.ylabel("RAFR",fontsize=fig_fontsize)
    #plt.legend(fontsize=fig_fontsize)

    plt.legend(fontsize=6,
               loc='upper center', bbox_to_anchor=(0.5, +1.35),
               fancybox=True, shadow=True, ncol=(num_architectures//2))

    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    fig.tight_layout(pad=0)
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])
    #plt.show()
    plt.savefig("figures/" + filename + ".pdf",bbox_inches='tight',pad_inches = 0.005)


def visualise_confusion_matrices(confusion_matrices: list, ignore_last: bool, model_list: list, filename: str) -> None:
    """

    Methods which accepts either a list of pd.Dataframes containing the confusion matrices
    or a list of strings which contain the path to them
    :param confusion_matrices:

    :return:
    """

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.60,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    if isinstance(confusion_matrices[0],str):
        for i in range(len(confusion_matrices)):
            confusion_matrices[i] = pd.read_csv(confusion_matrices[i], index_col=0)
    elif isinstance(confusion_matrices[0],pd.DataFrame):
        pass
    else:
        raise Exception

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    num_confusion_matrices = len(confusion_matrices)
    for i,confusion_matrix in enumerate(confusion_matrices):
        plt.subplot(1,num_confusion_matrices,i+1)

        # We ignore insertion and deletion errors for now
        if ignore_last:
            confusion_matrix = confusion_matrix.iloc[:-1,:-1]

        if i != 0:
            confusion_matrix = (confusion_matrix.div(confusion_matrix.sum(axis=1),axis=0) - \
                               confusion_matrices[0].div(confusion_matrices[0].sum(axis=1),axis=0)) * 100
        else:
            confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100

        im = plt.imshow(confusion_matrix,cmap="RdYlGn")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)



        if i == 0:
            plt.clim(-100, 100)
        else:
            plt.clim(-10,10)

        # This is needed so the numbers are right aligned
        for t in cbar.ax.get_yticklabels():
            t.set_horizontalalignment('right')
            # This (all-time function name competition winner) sets the padding from the colorbar
            t.set_x(4.5)

        if i == 0:
            ax = plt.gca()
            # We want to show all ticks...
            ax.set_xticks(np.arange(len(confusion_matrix.columns.values)))
            ax.set_yticks(np.arange(len(confusion_matrix.index.values)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(confusion_matrix.columns.values)
            ax.set_yticklabels(confusion_matrix.index.values)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=5)
            plt.setp(ax.get_yticklabels(), fontsize=6)
        else:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False)  # labels along the bottom edge are off

        # Setting an extra explanatory label for the percentages
        if i == 4:
            plt.ylabel("% (relative) accuracy",fontsize=7,labelpad=27)
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
        plt.title(model_list[i],fontsize=7)

    #plt.tight_layout(pad=0)
    #fig.tight_layout()
    plt.subplots_adjust(wspace=0.40)
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])
    plt.savefig("figures/" + filename + ".pdf",bbox_inches='tight',pad_inches = 0.005)

    plt.show()