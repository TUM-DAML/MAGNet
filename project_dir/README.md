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
├── model_ckpts              # every model has its own directory for model weights for inference
│   ├── MAGNET
│   ├── MOLER
│   ├── JTVAE
│   │   ├── zinc             # we additionally sepearate each dataset
│   │   │   └── ckpt1.pkl
│   │   │   └── ckpt1.pkl
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

Currently, when you want to perform inference on a model, you have to move its trained weights from `wb_logs/xyz` into `model_ckpts/xyz`. This is rather inconvenient and we will probably change this in the future.