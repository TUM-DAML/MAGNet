from rdkit import Chem

ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]


INFERENCE_HPS = dict(zinc=dict(min_size=10, max_size=40, sample_threshold=0.5, batch_size=32))

FLOW_NODE_PARAMS = dict(atol=1e-4, rtol=1e-4, solver="dopri5", sensitivity="adjoint", num_steps=200)

FLOW_TRAIN_HPS = dict(lr_sch_decay=0.99, lr=0.001, n_datapoints=250000, gradient_clip=100, patience=13)
