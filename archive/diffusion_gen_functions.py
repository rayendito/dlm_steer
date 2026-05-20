import torch
import numpy as np
import torch.nn.functional as F

ALPHA = 0.3
device = "cuda"

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

@torch.no_grad()
def generate(model, prompt, steer_vectors, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='random', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
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

    outputs_x = []
    for steer_idx, steer_vector in enumerate(steer_vectors):
        v = steer_vector.to(device)
        v = v / (v.norm() + 1e-8)
        v_ = v.to(device=device, dtype=torch.bfloat16)

        # only applying to last token
        # t[:, -1, :] += ALPHA * v_
        # return t

        def steer_hidden(t):
            return t + ALPHA * v_.view(1, 1, -1)

        # STEER REGISTER HOOKS
        if steer_idx < 32:
            def input_hook_fn(module, inp):
                h = inp[0]
                h = steer_hidden(h)
                return (h,) + inp[1:]
            str_block = model.model.transformer.blocks[steer_idx]
            str_handle = str_block.register_forward_pre_hook(input_hook_fn)
        else:
            def output_hook_fn(module, inp, out):
                # out can be tensor or tuple; handle both
                if isinstance(out, tuple):
                    h = out[0]
                    h = steer_hidden(h)
                    return (h,) + out[1:]
                else:
                    return steer_hidden(out)
            str_block = model.model.transformer.ln_f
            str_handle = str_block.register_forward_hook(output_hook_fn)

        try:
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
                        logits = model(x_, attention_mask=attention_mask_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x, attention_mask=attention_mask).logits

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
        finally:
            str_handle.remove()
        outputs_x.append(x)
    return outputs_x
