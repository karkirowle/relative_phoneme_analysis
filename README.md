
## Phoneme and articulatory analysis framework for Kaldi ASR

This work is the reproduction of the figures in the paper (TODO: insert name)

The code in this repository can be used to PER and AFER based on a Kaldi wer details file.

## What does it do
The WER and the PER is calculated based on the Levensteihn distance. We replicate the PER calculation and
extend this to articulatoy features, for which we also provide a conversion mapping.

### Requirements
Use the conda environment.yml file provided
sklearn version 0.25 will deprecate labels argument


### Instructions

Create the conda environment from environment yml

Run analysis.py

### Adaptation for your own datasets

Put your wer_details experiments in the same format as the other experiments into the folder experiments.
Adapt the script analysis.py to your own use case, i.e legend title should represent your own ASR architectures.


### TODO

* Create environment.yml file
* Intentions
