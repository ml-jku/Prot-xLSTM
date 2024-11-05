# Original code from ProtMamba under Apache License 2.0.
#
# Modifications made by Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 
#   - Add option to pass input state for generation
#   - Add functions to generate sequences with xlstm

import numpy as np
import torch
from protxlstm.mamba_utils_generation  import (
    InferenceParams,
    GenerationMixin,
    GreedySearchDecoderOnlyOutput,
    modify_logits_for_top_p_filtering, 
    modify_logits_for_min_p_filtering, 
    modify_logit_for_repetition_penalty,
    SampleDecoderOnlyOutput,
    update_graph_cache
)

from protxlstm.utils import AA_TO_ID, decode_sequence

def sample_safe(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)

            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(
                    torch.softmax(logits_top, dim=-1), num_samples=1
                ).squeeze(dim=-1),
            ]
        else:
            if min_p > 0.0:
                logits_top = logits.clone()
                max_prob = logits_top[..., 0].item()
                min_prob = max_prob * min_p
                modify_logits_for_min_p_filtering(logits_top, min_p)
                if temperature != 1.0:
                    logits_top /= temperature
                return torch.multinomial(
                    torch.softmax(logits_top, dim=-1), num_samples=1
                ).squeeze(dim=-1)
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(
                torch.softmax(logits_top, dim=-1), num_samples=1
            ).squeeze(dim=-1)


@torch.inference_mode()
def decode_safe(
    input_ids,
    position_ids,
    seq_position_ids,
    is_fim,
    model,
    max_length,
    state=None,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    cg=False,
    enable_timing=False,
    streamer = None,
    chunk_chunk_size = 2**15,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        is_fim: dictionary with mask indices and associated position indices
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(
            max_seqlen=max_length, max_batch_size=batch_size
        )

    def get_logits(input_ids, position_ids, seq_position_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                seq_position_ids=seq_position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids,
                position_ids,
                inference_params.seqlen_offset,
                seq_position_ids=seq_position_ids,
            ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits
    
    def get_xlstm_logits_step(input_ids, position_ids, seq_position_ids, state):

        if not input_ids.shape[1] == 1:

            for i in range(input_ids.shape[1]):
                if position_ids != None:
                    token_position_ids = position_ids[:,i:(i+1)]
                else:
                    token_position_ids = None
                if seq_position_ids != None:
                    token_seq_position_ids = seq_position_ids[:,i:(i+1)]
                else:
                    token_seq_position_ids = None
                logits, state = model.step(input_ids[:,i:(i+1)], state, position_ids=token_position_ids, seq_position_ids=token_seq_position_ids)
            
        else:
            logits, state = model.step(input_ids, state, position_ids=position_ids, seq_position_ids=seq_position_ids)

        logits = logits.squeeze(dim=1)
        if vocab_size is not None:
            logits = logits[..., :vocab_size]

        return logits, state
    
    def get_xlstm_logits_chunkwise(input_ids, position_ids, seq_position_ids, chunk_chunk_size=2**15, state=None):

        assert model.config.config_dataclass.mlstm_block.mlstm.backend == "chunkwise_variable"

        for chunk in range(input_ids.shape[1]//chunk_chunk_size+1):

            start_idx = chunk*chunk_chunk_size
            end_idx = min((chunk+1)*chunk_chunk_size, input_ids.shape[1])

            if start_idx == end_idx:
                pass
            
            else:
                input_ids_chunk = input_ids[:, start_idx:end_idx]

                if not position_ids == None:
                    position_ids_chunk = position_ids[:, start_idx:end_idx]
                else:
                    position_ids_chunk = None

                if not seq_position_ids == None:
                    seq_position_ids_chunk = seq_position_ids[:, start_idx:end_idx]
                else:
                    seq_position_ids_chunk = None   

                outputs = model(input_ids_chunk, position_ids=position_ids_chunk, seq_position_ids=seq_position_ids_chunk, state=state)
                logits, state = outputs.logits, outputs.state

        logits = logits[:,-1,:]
        logits = logits.squeeze(dim=1)
        if vocab_size is not None:
            logits = logits[..., :vocab_size]

        return logits, state    
           
    def sample_tokens(logits, inference_params):
        if (
            teacher_outputs is None
            or teacher_output_len <= inference_params.seqlen_offset
        ):
            token = sample_safe(
                logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature
            )
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def get_fim_position_id(
        last_position_ids, sampled_tokens, is_fim, repeat_next=False
    ):
        if type(is_fim) is dict:
            val = int(last_position_ids) + 1
            should_repeat_next = False
            if is_fim and int(sampled_tokens) in is_fim:
                val = is_fim[int(sampled_tokens)]
                should_repeat_next = True
            elif repeat_next:
                val = int(last_position_ids)
            return torch.full_like(last_position_ids, fill_value=val), should_repeat_next
        else:
            t = [get_fim_position_id(last_position_ids_, sampled_tokens_, is_fim_dict, repeat_next) for
                 (last_position_ids_, sampled_tokens_, is_fim_dict) in
                 zip(last_position_ids, sampled_tokens, is_fim)]
            return torch.stack([t_[0] for t_ in t], dim=0), t[0][1]

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).any():
            if current_token.shape[1] > 1:
                raise NotImplementedError("Batched eos_token_id not supported")
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        start.record()
    scores, sequences = [], [input_ids]
    new_position_ids, new_seq_position_ids = [position_ids], [seq_position_ids]
    sequences_cat = input_ids
    repeat_next = False
    if position_ids.shape[0] > 1:
        raise NotImplementedError("Batched generation with position_ids not supported")
    
    encode_context=True
    while not should_stop(sequences[-1], inference_params):

        from protxlstm.models.xlstm import xLSTMLMHeadModel
        if isinstance(model, xLSTMLMHeadModel):
            if encode_context:
                with torch.no_grad():
                    logits, state = get_xlstm_logits_chunkwise(sequences[-1], new_position_ids[-1], new_seq_position_ids[-1], state=state, chunk_chunk_size=chunk_chunk_size)
                encode_context = False
            else:
                logits, state = get_xlstm_logits_step(sequences[-1], new_position_ids[-1], new_seq_position_ids[-1], state=state)
        else:
            logits = get_logits(sequences[-1], new_position_ids[-1], new_seq_position_ids[-1], inference_params)

        scores.append(logits)
    
        inference_params.seqlen_offset += sequences[-1].shape[1]
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(scores[-1], inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(
                scores[-1].clone(), sequences_cat, repetition_penalty
            )
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)
        # Update position_ids
        if position_ids is not None:
            last_position_ids, repeat_next = get_fim_position_id(
                new_position_ids[-1][:, -1:], sampled_tokens, is_fim, repeat_next
            )
            new_position_ids.append(last_position_ids)
        # Update seq_position_ids
        if seq_position_ids is not None:
            new_seq_position_ids.append(new_seq_position_ids[-1][:, -1:])

        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    output_cls = (
        GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    )
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class GenerationMixinSafe(GenerationMixin):

    def generate(
        self,
        input_ids,
        position_ids,
        seq_position_ids,
        is_fim=None,
        state=None,
        max_length=1,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        chunk_chunk_size=2**15,
        **kwargs,
    ):

        output = decode_safe(
            input_ids,
            position_ids,
            seq_position_ids,
            is_fim,
            self,
            max_length,
            state=state,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            chunk_chunk_size=chunk_chunk_size,
            **kwargs,
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


def generate_sequence(model, tokens, position_ids=None, seq_position_ids=None, state=None, is_fim=False, max_length=2000, temperature=1., top_p=0.0, top_k=1,
                      return_dict_in_generate=False, output_scores=False, eos_token_id=AA_TO_ID["<cls>"], device="cuda", chunk_chunk_size=2**15):
    """Generating, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p. We assume that all sequences in the same batch have the same length.
    """
    input_ids = tokens.to(device)
    position_ids = position_ids.to(device) if position_ids is not None else None
    seq_position_ids = seq_position_ids.to(device) if seq_position_ids is not None else None
    # generate sequence
    out = model.generate(input_ids=input_ids,
                         position_ids=position_ids,
                         seq_position_ids=seq_position_ids,
                         is_fim=is_fim,
                         state=state,
                         max_length=max_length,
                         temperature=temperature,
                         top_p=top_p,
                         top_k=top_k,
                         return_dict_in_generate=return_dict_in_generate,
                         output_scores=output_scores,
                         eos_token_id=eos_token_id,
                         chunk_chunk_size=chunk_chunk_size,
                         )
    sequences = out.sequences
    dic = {"input": [decode_sequence(seq) for seq in sequences[:, :input_ids.shape[-1]].cpu().numpy()],
            "generated": [decode_sequence(seq) for seq in sequences[:, input_ids.shape[-1]:].cpu().numpy()],
            "input_tokens": [seq for seq in sequences[:, :input_ids.shape[-1]].cpu().numpy()],
            "generated_tokens": [seq for seq in sequences[:, input_ids.shape[-1]:].cpu().numpy()]}
    if output_scores:
        dic["scores"] = np.array([el.to(torch.float32).cpu().numpy() for el in out.scores]).transpose(1, 0, 2)
    return dic






