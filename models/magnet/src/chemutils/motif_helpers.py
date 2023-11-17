import torch


def get_atom_pos_encoding(atom_idx, motif_idx):
    # only works in one dimensions
    assert atom_idx.dim() == 1
    multiplicity, counter = [0], 1
    for i, element in enumerate(atom_idx[1:]):
        # still in same atom class and in same motif (second condition catches i.e. two C rings after each other)
        if element == atom_idx[i] and motif_idx[i + 1] == motif_idx[i]:
            multiplicity.append(counter)
            counter += 1
        # next atom class
        else:
            multiplicity.append(0)
            counter = 1
    assert len(multiplicity) == atom_idx.size(0)
    return torch.Tensor(multiplicity)


def get_motif_pos_encoding(motif_idx, unique):
    multiplicity = [0]
    assert motif_idx.dim() == 1
    assert unique.dim() == 1
    counter = 0
    for i, element in enumerate(motif_idx[1:]):
        # still in same motif, possibly increase
        if element == motif_idx[i]:
            # same motif but different hash -> increat
            if unique[i] != unique[i + 1]:
                counter += 1
            multiplicity.append(counter)
        # next motif, start again
        else:
            multiplicity.append(0)
            counter = 0
    assert len(multiplicity) == motif_idx.size(0)
    return torch.Tensor(multiplicity)


def atom_multiset_to_counts(multiset, num_nodes, num_atoms):
    dev = multiset.device
    multiset = torch.split(multiset, num_nodes)
    atom_counts = torch.zeros(len(num_nodes), num_atoms).to(dev)
    for i, atom_idx in enumerate(multiset):
        vals, counts = torch.unique(atom_idx, return_counts=True)
        atom_counts[i, vals.long()] += counts.to(dev)
    return atom_counts.to(dev)


def atom_counts_to_multiset(counts):
    # expects two-dim tensor, first dim batch_size, second dim num_xatoms
    atom_classes = torch.arange(counts.size(-1)).to(counts.device)
    atom_classes = atom_classes.repeat(counts.size(0))
    atom_idx = atom_classes.repeat_interleave(counts.flatten())
    return atom_idx


def block_diag_fill(diag_elements, fill_value, zero_value, non_zero_value):
    """
    Takes 0/1 input and places it on diagonal, all other elements in matrix are 1
    """
    # encode zeros (we want to keep those) as special character
    # graph decoding expects this later
    for de in diag_elements:
        de[de == 0] = zero_value
        if non_zero_value is not None:
            de[de > 0] = non_zero_value
    adjacency = torch.block_diag(*diag_elements)
    # recode entries, turn off-diagonal elements into ones (we need otherwise fully connected graph)
    adjacency[adjacency == 0] = fill_value
    return adjacency
