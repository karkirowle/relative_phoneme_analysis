import os
from corpus_2 import Corpus
import pickle
import time
import numpy as np
from figure_generator_2 import phoneme_barplot, articulatory_barplot, visualise_confusion_matrices
partition = "test"

preprocessing = False

if preprocessing:
    per = list()
    poa_afer = list()
    moa_afer = list()
    poa_cm = list()
    moa_cm = list()
    for i, experiment in enumerate((os.listdir("experiments"))):

        per_per_experiment = list()
        poa_afer_per_experiment = list()
        moa_afer_per_experiment = list()
        poa_cm_per_experiment = list()
        moa_cm_per_experiment = list()

        if i == 0:
            counts_holder = np.zeros((5,40))
        for fold in range(1,6):
            t = time.time()
            test_folder = [folder for folder in os.listdir(os.path.join("./experiments/",experiment,str(fold))) if partition in folder]
            wer_details = os.path.join("./experiments/",experiment,str(fold),test_folder[0],"wer_details","per_utt")

            corpus = Corpus(wer_details)

            per_per_experiment.append(corpus.all_pers())
            poa_afer_per_experiment.append(corpus.all_poa_afers())
            moa_afer_per_experiment.append(corpus.all_moa_afers())

            poa_cm_per_experiment.append(corpus.poa_confusion_matrix())
            moa_cm_per_experiment.append(corpus.moa_confusion_matrix())
            s = time.time() - t
            print("Fold took", s, "seconds")
            if i == 0:
                unique, counts = np.unique(corpus.all_ref_phonemes, return_counts=True)
                counts_holder[fold - 1,:] = counts
        per.append(per_per_experiment)
        poa_afer.append(poa_afer_per_experiment)
        moa_afer.append(moa_afer_per_experiment)
        poa_cm.append(poa_cm_per_experiment)
        moa_cm.append(moa_cm_per_experiment)

    with open('objs.pkl', 'wb') as file:
        pickle.dump([per, poa_afer, moa_afer, poa_cm, moa_cm, counts_holder], file)
else:
    with open('objs.pkl', 'rb') as file:
        per, poa_afer, moa_afer, poa_cm, moa_cm, counts_holder = pickle.load(file)
        #phoneme_barplot(per, counts_holder, ["Baseline","Baseline plus OC","DNN AM retraining","FHVAE","fMLLR"])
        #articulatory_barplot(moa_afer,["Baseline","Baseline plus OC","DNN AM retraining","FHVAE","fMLLR"],
        #                     ["Vowel", "Plosive", "Nasal", "Fricative", "Affricates", "Approximant"])
        #articulatory_barplot(poa_afer,["Baseline","Baseline plus OC","DNN AM retraining","FHVAE","fMLLR"],
        #                     ["Vowel","Bilabial","Labiodental","Dental","Alveolar","Postalveolar","Palatal","Velar","Glottal"])
        visualise_confusion_matrices(poa_cm,["Baseline","Baseline plus OC","DNN AM retraining","FHVAE","fMLLR"])
