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
def resteer_v2(
    model,
    tokenized_inputs,
    steer_vectors,
    resteer_steps,
    refill_steps,
    sampling_temp = 1,
    identify_temp = 0.5,
    mask_id=126336,
    strategy="low_confidence",
    alpha_decay=False
):
    all_results = []
    for i_step in range(resteer_steps):
        resteer_step_result = {
            "resteer_step" : i_step,
            "before" : tokenized_inputs["input_ids"].clone().detach().cpu()
        }
        out = model(
            tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            output_hidden_states=True,
        )
        logits = out.logits  # [B, T, V]
        x = tokenized_inputs["input_ids"].clone()
        attention_mask = tokenized_inputs["attention_mask"]

        # [B, T], 1 = token should be re-steered
        mask_probs, to_resteer_mask = identify_to_steer(
            out,
            attention_mask=attention_mask,
            steer_vectors=steer_vectors,
            temperature=identify_temp,
        )
        
        # change indices in x back to mask_id according to to_resteer_mask
        to_resteer_mask = to_resteer_mask & attention_mask.bool()
        resteer_step_result["mask_probs"] = mask_probs.clone().detach().cpu()
        resteer_step_result["steer_mask"] = to_resteer_mask.clone().detach().cpu()
        x[to_resteer_mask] = mask_id

        # usual diffusion step, with steering
        refill_mask = to_resteer_mask.clone()
        num_transfer_tokens = get_num_transfer_tokens(refill_mask, refill_steps)
        
        for refill_step in range(refill_steps):
            if (alpha_decay):
                alpha_scale = 1.0 - 0.9 * (refill_step / max(refill_steps - 1, 1))
                refill_steer_vectors = {
                    layer: alpha_scale * vec
                    for layer, vec in steer_vectors.items()
                }
            else:
                refill_steer_vectors = steer_vectors
            still_masked = (x == mask_id) & refill_mask
            # if done
            if not still_masked.any():
                break
            logits = model(x, steers=refill_steer_vectors, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=sampling_temp)
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
        tokenized_inputs["input_ids"] = x.detach()
        resteer_step_result["after"] = x.clone().detach().cpu()
        all_results.append(resteer_step_result)
    return all_results

@torch.no_grad()
def resteer_v2_val(
    model,
    tokenized_inputs,
    steer_vectors,
    mask_id=126336,
    identify_temperature=0.0001,
):
    """
    Val-style one-shot variant of ``resteer_v2``: identify positions → mask them → a single
    steered forward pass → argmax at all masked positions (no multi-step refill loop).
    """
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    out = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    x = input_ids.clone()
    _, to_resteer_mask = identify_to_steer(
        out,
        attention_mask=attention_mask,
        steer_vectors=steer_vectors,
        temperature=identify_temperature,
    )
    to_resteer_mask = to_resteer_mask & attention_mask.bool()
    x[to_resteer_mask] = mask_id

    logits = model(x, steers=steer_vectors, attention_mask=attention_mask).logits
    logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    masked = x == mask_id
    x[masked] = x0[masked]
    return x


@torch.no_grad()
def identify_to_steer(out, attention_mask, steer_vectors, tokenizer=None, temperature=0.0):
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
    return probs, mask