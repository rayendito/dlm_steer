import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steers=None, attention_mask=None, steer_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    if steer_mask is not None:
        steer_mask = torch.cat([torch.zeros(prompt.shape[1]).to(model.device), steer_mask])
        
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, steers=steers, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, steers=steers, attention_mask=attention_mask, steer_mask=steer_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

@torch.no_grad()
def resteer(
    model, prompt, steers, resteer_idx,
    attention_mask=None, refine_steps=5, resteer_pad=3, remask_per_refine=10,
    temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False
):
    # instead of appending, remask and append where necessary
    resteer_mask = torch.zeros(prompt.shape[0], prompt.shape[1], dtype=torch.long).to(model.device)
    for item in resteer_idx:
        if len(item) == 2:  # range
            start, end = item
            resteer_mask[0, start:end+1] = 1
        else:  # single index
            resteer_mask[0, item[0]] = 1
    prompt[resteer_mask == 1] = mask_id

    # now expand prompt + mask by inserting extra mask tokens after each remasked span
    new_prompt_parts = []
    new_mask_parts = []

    i = 0
    L = prompt.shape[1]
    while i < L:
        # check whether this position starts a remasked run
        if resteer_mask[0, i] == 1:
            j = i
            while j + 1 < L and resteer_mask[0, j + 1] == 1:
                j += 1

            # keep the remasked span itself
            new_prompt_parts.append(prompt[:, i:j+1])
            new_mask_parts.append(resteer_mask[:, i:j+1])

            # insert extra pad masks after the span
            pad_tokens = torch.full(
                (prompt.shape[0], resteer_pad),
                mask_id,
                dtype=prompt.dtype,
                device=prompt.device,
            )
            pad_mask = torch.ones(
                (prompt.shape[0], resteer_pad),
                dtype=resteer_mask.dtype,
                device=resteer_mask.device,
            )

            new_prompt_parts.append(pad_tokens)
            new_mask_parts.append(pad_mask)

            i = j + 1
        else:
            new_prompt_parts.append(prompt[:, i:i+1])
            new_mask_parts.append(resteer_mask[:, i:i+1])
            i += 1
    
    x = torch.cat(new_prompt_parts, dim=1).to(model.device)
    resteer_mask = torch.cat(new_mask_parts, dim=1).to(model.device)
    prompt_index = (x != mask_id).to(model.device)
    attention_mask = torch.ones(prompt.shape, dtype=attention_mask.dtype).to(model.device)
    
    x = x.clone()
    # STEER ONCE
    logits = model(x, steers=steers, attention_mask=attention_mask).logits
    if logits_eos_inf:
        logits[:, :, 126081] = -torch.inf
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    x[resteer_mask == 1] = x0[resteer_mask == 1]
    refine_evolution = [x.clone()]

    # REFINE N TIMES
    for _ in range(refine_steps):
        logits = model(x, attention_mask=attention_mask).logits
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x, -1)), -1) # b, l
            probs = (1.0 - x_p)
            transfer_index = (torch.rand_like(probs) < probs)
        elif remasking == 'random':
            x_p = torch.rand((x.shape[0], x.shape[1]), device=x.device)
        else:
            raise NotImplementedError(remasking)
        
        # REMASKING
        transfer_index = transfer_index & (resteer_mask == 1)
        x[transfer_index] = mask_id

        # REDEMASKING
        logits = model(x, attention_mask=attention_mask).logits
        logits_with_noise = add_gumbel_noise(logits, temperature=1)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # recompute x0 each step
        x[transfer_index] = x0[transfer_index]
        refine_evolution.append(x.clone())

    return refine_evolution

@torch.no_grad()
def resteer_v2(
    model,
    tokenized_inputs,
    steer_vectors,
    resteer_steps,
    refill_steps,
    mask_id=126336,
    strategy="low_confidence"
):
    for i_step in range(resteer_steps):
        out = model(
            tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            output_hidden_states=True,
        )
        logits = out.logits  # [B, T, V]
        x = tokenized_inputs["input_ids"].clone()
        attention_mask = tokenized_inputs["attention_mask"]

        # [B, T], 1 = token should be re-steered
        to_resteer_mask = identify_to_steer(
            out,
            attention_mask=attention_mask,
            steer_vectors=steer_vectors,
            temperature=0.0001,
        )
        
        # change indices in x back to mask_id according to to_resteer_mask
        to_resteer_mask = to_resteer_mask & attention_mask.bool()
        x[to_resteer_mask] = mask_id

        # usual diffusion step, with steering
        refill_mask = to_resteer_mask.clone()
        num_transfer_tokens = get_num_transfer_tokens(refill_mask, refill_steps)
        for refill_step in range(refill_steps):
            still_masked = (x == mask_id) & refill_mask
            # if done
            if not still_masked.any():
                break
            logits = model(x, steers=steer_vectors, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if strategy == "low_confidence":
                probs = torch.softmax(logits, dim=-1)
                scores = torch.gather(
                    probs,
                    dim=-1,
                    index=x0.unsqueeze(-1),
                ).squeeze(-1)
            elif strategy == "random":
                scores = torch.rand(x.shape, device=x.device)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            scores = torch.where(
                still_masked,
                scores,
                torch.full_like(scores, -torch.inf),
            )

            transfer_index = torch.zeros_like(still_masked)

            for b in range(x.shape[0]):
                k = int(num_transfer_tokens[b, refill_step].item())

                if k > 0:
                    k = min(k, int(still_masked[b].sum().item()))
                    _, idx = torch.topk(scores[b], k=k)
                    transfer_index[b, idx] = True

            x[transfer_index] = x0[transfer_index]
    return x
        

@torch.no_grad()
def identify_to_steer(out, attention_mask, steer_vectors, tokenizer=None, temperature=0.1):
    sims = []
    for steer_idx, svector in steer_vectors.items():
        h = out.hidden_states[steer_idx] # [B, T, D]
        sim = F.cosine_similarity(
            h,
            svector.view(1, 1, -1), # [1, 1, D]
            dim=-1
        ) # [B, T]
        sims.append(sim)
    cosines_avg = torch.stack(sims, dim=0).mean(dim=0)   # [B, T] (average over steer layers)

    # 1. sample indices
    probs = torch.sigmoid(-cosines_avg / temperature)
    probs = probs.masked_fill(attention_mask == 0, 0.0)
    mask = (torch.rand_like(probs) < probs)
    return mask