from rdkit import Chem

ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "B", "Se"]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]


FLOW_NODE_PARAMS = dict(atol=1e-4, rtol=1e-4, solver="dopri5", sensitivity="adjoint", num_steps=200)

FLOW_TRAIN_HPS = dict(lr_sch_decay=0.99, lr=0.001, gradient_clip=100, patience=13)
