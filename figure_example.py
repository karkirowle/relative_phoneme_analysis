import time
from corpus import WERDetails
from utils import HParam
#partition = "test"
#number_of_phonemes = 40
preprocessing = True

import numpy as np
import pandas as pd

if preprocessing:
    config = HParam("configs/dutch.yaml")
    wer_details = WERDetails("experiments/jasmin_example/scoring_kaldi/wer_details/per_utt", skip_calculation=False,
                             config=config)

    #phoneme, other = wer_details.all_poa_afers()

    t = time.time()
    moa_mat = wer_details.moa_confusion_matrix()
    s = time.time()
    print(s-t, "secs")
    poa_mat = wer_details.poa_confusion_matrix()
    k = time.time()
    print(k-s, "secs")
    print(poa_mat)
    print()

    #df = pd.DataFrame(data=other, index=phoneme)
