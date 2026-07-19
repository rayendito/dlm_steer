import torch

from . import helpers


@torch.no_grad()
def timpa_ar(
    instruction,
    text,
    model=None,
    tokenizer=None,
    completion_fn=None,
    use_chat_template=True,
    max_new_tokens=256,
    temperature=0.0,
    generation_kwargs=None,
    api_kwargs=None,
):
    """Rewrite text directly with an autoregressive model or API callback.

    For a local model, ``instruction`` is rendered as a system message and the
    source text as a user message. For an API, ``completion_fn`` is called once
    per text as ``completion_fn(messages, max_new_tokens=..., temperature=...,
    **api_kwargs)`` and must return the rewritten string.

    Exactly one backend must be selected: either provide both ``model`` and
    ``tokenizer``, or provide ``completion_fn``. The return value is always a
    list containing one rewritten string per input text.
    """
    texts = helpers._as_text_list(text)
    instructions = helpers._as_prompt_list(
        instruction,
        len(texts),
        name="instruction",
    )
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("temperature must be greater than or equal to zero.")
    if generation_kwargs is not None and not isinstance(generation_kwargs, dict):
        raise TypeError("generation_kwargs must be a dictionary or None.")
    if api_kwargs is not None and not isinstance(api_kwargs, dict):
        raise TypeError("api_kwargs must be a dictionary or None.")

    uses_local_model = model is not None or tokenizer is not None
    if completion_fn is not None and uses_local_model:
        raise ValueError(
            "Provide either model/tokenizer or completion_fn, not both backends."
        )
    if completion_fn is None and (model is None or tokenizer is None):
        raise ValueError(
            "Provide both model and tokenizer, or provide a completion_fn."
        )
    if completion_fn is not None and not callable(completion_fn):
        raise TypeError("completion_fn must be callable.")

    rewritten_texts = []
    for prompt, item in zip(instructions, texts):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": item},
        ]
        if completion_fn is not None:
            request_kwargs = dict(api_kwargs or {})
            request_kwargs.setdefault("max_new_tokens", max_new_tokens)
            request_kwargs.setdefault("temperature", float(temperature))
            result = completion_fn(
                messages,
                **request_kwargs,
            )
            if not isinstance(result, str):
                raise TypeError("completion_fn must return a string.")
            rewritten_texts.append(result)
            continue

        if use_chat_template:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(
                    "use_chat_template=True requires a tokenizer with a chat template."
                )
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(input_ids, dict):
                input_ids = input_ids["input_ids"]
        else:
            rendered = (
                f"Instruction: {prompt}\n\n"
                f"Text: {item}\n\n"
                "Rewritten text:"
            )
            input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"]

        input_ids = input_ids.to(helpers._model_device(model))
        attention_mask = torch.ones_like(input_ids)
        local_generation_kwargs = dict(generation_kwargs or {})
        local_generation_kwargs["max_new_tokens"] = max_new_tokens
        local_generation_kwargs["do_sample"] = temperature > 0
        if temperature > 0:
            local_generation_kwargs["temperature"] = float(temperature)
        else:
            local_generation_kwargs.pop("temperature", None)
        if "pad_token_id" not in local_generation_kwargs:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                local_generation_kwargs["pad_token_id"] = pad_token_id

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **local_generation_kwargs,
        )
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        continuation = generated[0, input_ids.shape[1]:]
        rewritten_texts.append(
            tokenizer.decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return rewritten_texts


@torch.no_grad()
def timpa_probabilistic(
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    steer,
    text,
    use_chat_template=True,
    base_assistant_prompt="You are a helpful assistant",
    temperature=1.0,
    margin=0.001,
    generator=None,
    refill_steps=32,
    sampling_temperature=0.0,
    refill_strategy="low_confidence",
    detection_strategy="model",
    random_mask_probability=0.5,
):
    """Detect with AR likelihood changes, then refill with LLaDA.

    Set ``detection_strategy="random"`` for the random-detection baseline. It
    assigns every non-padding token ``random_mask_probability``, then samples
    one decision per word so partial-word masking remains impossible.

    Returns ``(tokenized_text, masking_probs, masked_positions,
    regenerated_texts)``.
    """
    if detection_strategy not in {"model", "random"}:
        raise ValueError("detection_strategy must be 'model' or 'random'.")

    texts = helpers._as_text_list(text)
    steer_prompts = helpers._as_prompt_list(steer, len(texts), name="steer")
    if detection_strategy == "model":
        tokenized_text, masking_probs, masked_positions = (
            helpers._probabilistic_token_detection(
                tokenizer=tokenizer,
                identifier_model=identifier_model,
                identifier_tokenizer=identifier_tokenizer,
                steer=steer_prompts,
                texts=texts,
                base_assistant_prompt=base_assistant_prompt,
                temperature=temperature,
                margin=margin,
                use_chat_template=use_chat_template,
                generator=generator,
            )
        )
    else:
        tokenized_text, masking_probs, masked_positions = (
            helpers._random_token_detection(
                tokenizer=tokenizer,
                texts=texts,
                probability=random_mask_probability,
                device=helpers._model_device(model),
                generator=generator,
            )
        )

    regenerated_texts = helpers.regenerate_masked_text(
        model=model,
        tokenizer=tokenizer,
        steer=steer_prompts,
        text=texts,
        masked_positions=masked_positions,
        response_attention_mask=tokenized_text.get("attention_mask"),
        use_chat_template=use_chat_template,
        refill_steps=refill_steps,
        sampling_temperature=sampling_temperature,
        refill_strategy=refill_strategy,
    )
    return tokenized_text, masking_probs, masked_positions, regenerated_texts


@torch.no_grad()
def timpa_steer(
    model,
    tokenizer,
    steer_vectors,
    text,
    refill_steps=32,
    sampling_temperature=1.0,
    temperature=1.0,
    generator=None,
    refill_strategy="low_confidence",
    system_prompt="You are a helpful assistant",
    use_chat_template=True,
    steer_mode="project_out",
    alpha=1.0,
    margin=0.05,
    detection_strategy="model",
    random_mask_probability=0.5,
):
    """Detect with activation similarity, then refill with intervention.

    ``steer_mode="add"`` applies each direction at its selected layer;
    ``steer_mode="project_out"`` removes the selected direction at every
    transformer layer. Set ``detection_strategy="random"`` to retain the same
    steered refill while replacing cosine detection with random word masks.

    Returns ``(tokenized_text, masking_probs, masked_positions,
    regenerated_texts)``.
    """
    if detection_strategy not in {"model", "random"}:
        raise ValueError("detection_strategy must be 'model' or 'random'.")
    if not isinstance(alpha, (int, float)) or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("alpha must be a finite number.")
    if alpha < 0:
        raise ValueError("alpha must be greater than or equal to zero.")

    texts = helpers._as_text_list(text)
    system_prompts = helpers._as_prompt_list(
        system_prompt,
        len(texts),
        name="system_prompt",
    )
    prepared_vectors, intervention_vectors = helpers._prepare_steer_vectors(
        model,
        steer_vectors,
        steer_mode,
    )

    if detection_strategy == "model":
        tokenized_text, masking_probs, masked_positions = (
            helpers._steering_token_detection(
                model=model,
                tokenizer=tokenizer,
                prepared_vectors=prepared_vectors,
                system_prompts=system_prompts,
                texts=texts,
                use_chat_template=use_chat_template,
                temperature=temperature,
                margin=margin,
                generator=generator,
            )
        )
    else:
        tokenized_text, masking_probs, masked_positions = (
            helpers._random_token_detection(
                tokenizer=tokenizer,
                texts=texts,
                probability=random_mask_probability,
                device=helpers._model_device(model),
                generator=generator,
            )
        )

    regenerated_texts = helpers.regenerate_masked_text(
        model=model,
        tokenizer=tokenizer,
        steer=system_prompts,
        text=texts,
        masked_positions=masked_positions,
        response_attention_mask=tokenized_text.get("attention_mask"),
        use_chat_template=use_chat_template,
        refill_steps=refill_steps,
        sampling_temperature=sampling_temperature,
        refill_strategy=refill_strategy,
        steer_vectors=intervention_vectors,
        steer_mode=steer_mode,
        alpha=alpha,
    )
    return tokenized_text, masking_probs, masked_positions, regenerated_texts


@torch.no_grad()
def timpa_hybrid(
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    steer_vectors,
    steer,
    text,
    use_chat_template=True,
    base_assistant_prompt="You are a helpful assistant",
    system_prompt=None,
    temperature=1.0,
    margin=0.001,
    generator=None,
    refill_steps=32,
    sampling_temperature=1.0,
    refill_strategy="low_confidence",
    alpha=1.0,
):
    """Detect with AR likelihood changes and refill with additive steering.

    ``steer`` and ``base_assistant_prompt`` control only probabilistic token
    detection. The LLaDA refill is conditioned on ``system_prompt`` (or a neutral
    helpful-assistant prompt when it is omitted) and receives ``steer_vectors``
    through the additive residual-stream intervention, keeping the two roles
    explicit.

    Returns ``(tokenized_text, masking_probs, masked_positions,
    regenerated_texts)``.
    """
    if not isinstance(alpha, (int, float)) or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("alpha must be a finite number.")
    if alpha < 0:
        raise ValueError("alpha must be greater than or equal to zero.")

    texts = helpers._as_text_list(text)
    refill_prompt = (
        "You are a helpful assistant" if system_prompt is None else system_prompt
    )
    system_prompts = helpers._as_prompt_list(
        refill_prompt,
        len(texts),
        name="system_prompt",
    )
    _, intervention_vectors = helpers._prepare_steer_vectors(
        model,
        steer_vectors,
        steer_mode="add",
    )
    tokenized_text, masking_probs, masked_positions = (
        helpers._probabilistic_token_detection(
            tokenizer=tokenizer,
            identifier_model=identifier_model,
            identifier_tokenizer=identifier_tokenizer,
            steer=steer,
            texts=texts,
            base_assistant_prompt=base_assistant_prompt,
            temperature=temperature,
            margin=margin,
            use_chat_template=use_chat_template,
            generator=generator,
        )
    )
    regenerated_texts = helpers.regenerate_masked_text(
        model=model,
        tokenizer=tokenizer,
        steer=system_prompts,
        text=texts,
        masked_positions=masked_positions,
        response_attention_mask=tokenized_text.get("attention_mask"),
        use_chat_template=use_chat_template,
        refill_steps=refill_steps,
        sampling_temperature=sampling_temperature,
        refill_strategy=refill_strategy,
        steer_vectors=intervention_vectors,
        steer_mode="add",
        alpha=alpha,
    )
    return tokenized_text, masking_probs, masked_positions, regenerated_texts
