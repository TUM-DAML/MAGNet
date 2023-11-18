import torch
import torch.nn.functional as F


def transformer_forward(
    transformer_nn,
    memory_nn,
    num_nodes,
    memory,
    tgt_input,
    max_pos,
    tgt_key_padding_mask=None,
    num_nodes_memory=None,
):
    if num_nodes_memory is None:
        num_nodes_memory = num_nodes
    if max_pos > 1:
        positionals = torch.cat([torch.arange(n) for n in num_nodes_memory]).long()
        one_hot_pos = F.one_hot(positionals, num_classes=max_pos).to(memory.device)
        memory = torch.cat([memory, one_hot_pos], dim=1)
        memory = memory_nn(memory)
        if isinstance(num_nodes_memory, torch.Tensor):
            num_nodes_memory = num_nodes_memory.tolist()
        assert isinstance(num_nodes_memory, list)
        memory = torch.split(memory, num_nodes_memory, dim=0)
        memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True)
        memory_key_padding_mask = (memory == 0).all(-1)
    else:
        memory = memory_nn(memory).unsqueeze(1)
        memory_key_padding_mask = None
    tgt_key_padding_mask = tgt_key_padding_mask if tgt_key_padding_mask is not None else memory_key_padding_mask
    _sz = tgt_input.size(1)
    tgt_mask = torch.triu(torch.full((_sz, _sz), float("-inf"), device=memory.device), diagonal=1)
    transformer_output = transformer_nn(
        tgt_input,
        memory,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    return transformer_output
