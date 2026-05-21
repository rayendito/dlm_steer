import torch

def get_steer_vectors(model, tokenizer, texts, device = "cuda", modelname = "diffusion"):
    if (modelname == "diffusion"):
        steer_vectors = [[] for _ in range(len(model.model.transformer.blocks) + 1)]
    elif (modelname == "ar"):
        steer_vectors = [[] for _ in range(len(model.model.layers) + 1)]
    else:
        raise NotImplementedError()
    
    for t in texts:
        inputs = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False,
        ).to(device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        for i, h in enumerate(out.hidden_states):
            if modelname == "diffusion":
                pooled = h.mean(dim=1)[0] # mean over tokens → [d]
            elif modelname == "ar":
                pooled = h[:, -1, :][0] # get last token
            else:
                raise NotImplementedError()
            steer_vectors[i].append(pooled.float())

    # now average across texts for each layer
    steer_vectors = [
        torch.stack(layer_vecs)
        for layer_vecs in steer_vectors
    ]
    return torch.stack(steer_vectors)

def l2_normalize(v, eps=1e-12):
    return v / (v.norm(p=2) + eps)
