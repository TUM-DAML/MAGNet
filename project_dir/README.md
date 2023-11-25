## Project Directory

This directory should have to following structure:

```
.
├── data                     # every model has its own directory for preprocessed data / vocabulary
│   ├── MAGNET
│   ├── MOLER
│   ├── JTVAE
│   │   ├── zinc             # we additionally sepearate each dataset
│   │   │   └── vocab.txt
│   │   │   └── train.pkl
│   │   │   └── ...
├── smiles_files             # here we store the raw smiles files
│   ├── zinc
|   │   └── train.txt
|   │   └── val.txt
|   │   └── test.txt
│   ├── qm9
|   │   └── train.txt
|   │   └── val.txt
|   │   └── test.txt
├── wb_logs                   # here we store Weights & Biases log files and checkpoints
```