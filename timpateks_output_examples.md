# TIMPA Output Examples

These examples show the exact output schema produced by `run.py`. The rewritten
sentences are illustrative placeholders, not outputs from completed model runs.

General rules:

- One invocation creates one CSV and one JSON file with the same stem.
- The default output root is `timpateks_results/`; each run is written below
  `{dataset}/{technique}/`.
- The filename is `{dataset}_{technique}_{experiment_id}`.
- `--run-name` becomes the experiment ID. Without it, the ID is a UTC timestamp
  plus an eight-character random suffix.
- `direction` is always the target direction, not the source direction.
- The current pipeline performs one TIMPA pass, so it writes only `after1`.
- `picked_tokens` is a JSON array stored inside the CSV cell. Each record contains
  the unpadded, zero-based token `position`, exact `token_id`, vocabulary `token`,
  and readable decoded `text` fragment.
- AR baselines do not select replacement tokens, so their `picked_tokens` value
  is `[]`.
- IMDb and catdog put both directions in the same CSV.
- ELIFive and Dolly reuse each base input for all three target directions.
- `timpa_steer` and `timpa_hybrid` are currently unsupported for ELIFive and
  Dolly.

## IMDb

Representative experiment: additive `timpa_steer` with one example in each
direction.

Files:

```text
timpateks_results/imdb/timpa_steer_add/imdb_timpa_steer_add_demo-imdb-add.csv
timpateks_results/imdb/timpa_steer_add/imdb_timpa_steer_add_demo-imdb-add.json
```

CSV:

```csv
direction,before,after1,picked_tokens
negative,"Amazing movie. Beautiful scenery and great acting. Very poetic. Highly recommend.","The scenery is attractive, but the weak writing and lifeless pacing make the film difficult to recommend.","[{""position"":0,""token_id"":56197,""token"":""Amazing"",""text"":""Amazing""}]"
positive,"Bad acting. Lame story. Terrible effects. Horrible dialogue.","Despite its rough edges, the film has an energetic charm and enough entertaining moments to make it worth watching.","[{""position"":0,""token_id"":26317,""token"":""Bad"",""text"":""Bad""}]"
```

The first row is positive -> negative, while the second is negative -> positive.

JSON:

```json
{
  "arguments": {
    "alpha": 600.0,
    "api_base_url": null,
    "api_key_env": "OPENAI_API_KEY",
    "api_model_id": null,
    "ar_backend": "local",
    "ar_model_id": "Qwen/Qwen2.5-14B-Instruct",
    "batch_size": 1,
    "dataset": "imdb",
    "detection": "model",
    "dlm_model_id": "GSAI-ML/LLaDA-8B-Instruct",
    "margin": 0.05,
    "max_new_tokens": 2048,
    "max_samples": 1,
    "method": "timpa_steer",
    "output_dir": "timpateks_results",
    "random_mask_probability": 0.5,
    "refill_steps": 32,
    "refill_strategy": "low_confidence",
    "run_name": "demo-imdb-add",
    "sampling_temperature": null,
    "seed": 42,
    "source_layer": 23,
    "split": "train",
    "steer": "add",
    "steer_layers": [16, 25, 31],
    "temperature": 0.1,
    "token_position": -4
  },
  "created_at": "2026-07-20T12:00:00+00:00",
  "csv_file": "imdb_timpa_steer_add_demo-imdb-add.csv",
  "direction_counts": {
    "negative": 1,
    "positive": 1
  },
  "effective_method_parameters": {
    "margin": 0.05,
    "sampling_temperature": 1.0,
    "temperature": 0.1
  },
  "experiment_id": "demo-imdb-add",
  "experiment_name": "imdb_timpa_steer_add_demo-imdb-add",
  "num_results": 2,
  "result_columns": [
    "direction",
    "before",
    "after1",
    "picked_tokens"
  ],
  "technique": "timpa_steer_add"
}
```

## CatDog

Representative experiment: `timpa_hybrid` with one example in each direction.

Files:

```text
timpateks_results/catdog/timpa_hybrid/catdog_timpa_hybrid_demo-catdog-hybrid.csv
timpateks_results/catdog/timpa_hybrid/catdog_timpa_hybrid_demo-catdog-hybrid.json
```

CSV:

```csv
direction,before,after1,picked_tokens
dog,"The cat's fur rippled in the wind as she raced across the open field.","The dog's fur rippled in the wind as it raced eagerly across the open field.","[{""position"":1,""token_id"":7748,""token"":""Ġcat"",""text"":"" cat""}]"
cat,"The dog's wagging tail greeted its owner after a long day of work.","The cat greeted its owner with a soft purr after a long day of work.","[{""position"":1,""token_id"":7339,""token"":""Ġdog"",""text"":"" dog""}]"
```

The first row is cat -> dog, while the second is dog -> cat. Hybrid detection
compares the contrasting prompts, but regeneration uses the neutral system
prompt and the target-minus-source additive vector.

JSON:

```json
{
  "arguments": {
    "alpha": 600.0,
    "api_base_url": null,
    "api_key_env": "OPENAI_API_KEY",
    "api_model_id": null,
    "ar_backend": "local",
    "ar_model_id": "Qwen/Qwen2.5-14B-Instruct",
    "batch_size": 1,
    "dataset": "catdog",
    "detection": "model",
    "dlm_model_id": "GSAI-ML/LLaDA-8B-Instruct",
    "margin": 0.001,
    "max_new_tokens": 2048,
    "max_samples": 1,
    "method": "timpa_hybrid",
    "output_dir": "timpateks_results",
    "random_mask_probability": 0.5,
    "refill_steps": 32,
    "refill_strategy": "low_confidence",
    "run_name": "demo-catdog-hybrid",
    "sampling_temperature": 1.0,
    "seed": 42,
    "source_layer": 23,
    "split": "train",
    "steer": "add",
    "steer_layers": [16, 25, 31],
    "temperature": 0.25,
    "token_position": -4
  },
  "created_at": "2026-07-20T12:05:00+00:00",
  "csv_file": "catdog_timpa_hybrid_demo-catdog-hybrid.csv",
  "direction_counts": {
    "cat": 1,
    "dog": 1
  },
  "effective_method_parameters": {
    "margin": 0.001,
    "sampling_temperature": 1.0,
    "temperature": 0.25
  },
  "experiment_id": "demo-catdog-hybrid",
  "experiment_name": "catdog_timpa_hybrid_demo-catdog-hybrid",
  "num_results": 2,
  "result_columns": [
    "direction",
    "before",
    "after1",
    "picked_tokens"
  ],
  "technique": "timpa_hybrid"
}
```

## ELIFive

Representative experiment: `timpa_probabilistic` with one base example reused
for all three reading levels.

Files:

```text
timpateks_results/elifive/timpa_probabilistic/elifive_timpa_probabilistic_demo-elifive-prob.csv
timpateks_results/elifive/timpa_probabilistic/elifive_timpa_probabilistic_demo-elifive-prob.json
```

CSV:

```csv
direction,before,after1,picked_tokens
5yo,"A limit describes the value a function approaches as its input nears a point.","A limit tells us what number a math rule is getting closer and closer to.","[{""position"":6,""token_id"":1399,""token"":""Ġfunction"",""text"":"" function""}]"
highschool,"A limit describes the value a function approaches as its input nears a point.","A limit is the value a function approaches as the input gets increasingly close to a particular point.","[{""position"":11,""token_id"":676,""token"":""Ġne"",""text"":"" ne""},{""position"":12,""token_id"":1536,""token"":""ars"",""text"":""ars""}]"
phd,"A limit describes the value a function approaches as its input nears a point.","A limit characterizes the asymptotic value of a function in a punctured neighborhood of a specified point.","[{""position"":2,""token_id"":15415,""token"":""Ġdescribes"",""text"":"" describes""}]"
```

All three rows start from the same base text. With ten source examples, the real
CSV contains thirty rows: ten for each target direction.

JSON:

```json
{
  "arguments": {
    "alpha": 1.0,
    "api_base_url": null,
    "api_key_env": "OPENAI_API_KEY",
    "api_model_id": null,
    "ar_backend": "local",
    "ar_model_id": "Qwen/Qwen2.5-14B-Instruct",
    "batch_size": 1,
    "dataset": "elifive",
    "detection": "model",
    "dlm_model_id": "GSAI-ML/LLaDA-8B-Instruct",
    "margin": 0.001,
    "max_new_tokens": 2048,
    "max_samples": 1,
    "method": "timpa_probabilistic",
    "output_dir": "timpateks_results",
    "random_mask_probability": 0.5,
    "refill_steps": 32,
    "refill_strategy": "low_confidence",
    "run_name": "demo-elifive-prob",
    "sampling_temperature": null,
    "seed": 42,
    "source_layer": 23,
    "split": "train",
    "steer": null,
    "steer_layers": [16, 25, 31],
    "temperature": 0.25,
    "token_position": -4
  },
  "created_at": "2026-07-20T12:10:00+00:00",
  "csv_file": "elifive_timpa_probabilistic_demo-elifive-prob.csv",
  "direction_counts": {
    "5yo": 1,
    "highschool": 1,
    "phd": 1
  },
  "effective_method_parameters": {
    "margin": 0.001,
    "sampling_temperature": 0.0,
    "temperature": 0.25
  },
  "experiment_id": "demo-elifive-prob",
  "experiment_name": "elifive_timpa_probabilistic_demo-elifive-prob",
  "num_results": 3,
  "result_columns": [
    "direction",
    "before",
    "after1",
    "picked_tokens"
  ],
  "technique": "timpa_probabilistic"
}
```

## Dolly Sample

Representative experiment: the local `timpa_ar` baseline with one base example
reused for all three target styles.

Files:

```text
timpateks_results/dolly_sample/ar/dolly_sample_ar_demo-dolly-ar.csv
timpateks_results/dolly_sample/ar/dolly_sample_ar_demo-dolly-ar.json
```

CSV:

```csv
direction,before,after1,picked_tokens
pirate,"daffodil, rose, lily, daisy, violet, jasmine","Arrr, daffodil, rose, lily, daisy, violet, and jasmine be the flowers on our list.",[]
mean,"daffodil, rose, lily, daisy, violet, jasmine","It is just a predictable list of flowers: daffodil, rose, lily, daisy, violet, and jasmine.",[]
flirty,"daffodil, rose, lily, daisy, violet, jasmine","Daffodil, rose, lily, daisy, violet, and jasmine - though none of them is quite as lovely as you.",[]
```

JSON:

```json
{
  "arguments": {
    "alpha": 1.0,
    "api_base_url": null,
    "api_key_env": "OPENAI_API_KEY",
    "api_model_id": null,
    "ar_backend": "local",
    "ar_model_id": "Qwen/Qwen2.5-14B-Instruct",
    "batch_size": 1,
    "dataset": "dolly_sample",
    "detection": "model",
    "dlm_model_id": "GSAI-ML/LLaDA-8B-Instruct",
    "margin": null,
    "max_new_tokens": 2048,
    "max_samples": 1,
    "method": "timpa_ar",
    "output_dir": "timpateks_results",
    "random_mask_probability": 0.5,
    "refill_steps": 32,
    "refill_strategy": "low_confidence",
    "run_name": "demo-dolly-ar",
    "sampling_temperature": null,
    "seed": 42,
    "source_layer": 23,
    "split": "train",
    "steer": null,
    "steer_layers": [16, 25, 31],
    "temperature": null,
    "token_position": -4
  },
  "created_at": "2026-07-20T12:15:00+00:00",
  "csv_file": "dolly_sample_ar_demo-dolly-ar.csv",
  "direction_counts": {
    "flirty": 1,
    "mean": 1,
    "pirate": 1
  },
  "effective_method_parameters": {
    "temperature": 0.0
  },
  "experiment_id": "demo-dolly-ar",
  "experiment_name": "dolly_sample_ar_demo-dolly-ar",
  "num_results": 3,
  "result_columns": [
    "direction",
    "before",
    "after1",
    "picked_tokens"
  ],
  "technique": "ar"
}
```

## Scaling With More Inputs

`--max-samples` applies per source direction or per target style:

| Dataset | `--max-samples 10` | CSV rows |
| --- | ---: | ---: |
| IMDb | 10 positive + 10 negative source reviews | 20 |
| CatDog | 10 cat + 10 dog source sentences | 20 |
| ELIFive | 10 base texts x 3 target levels | 30 |
| Dolly Sample | 10 base texts x 3 target styles | 30 |

Every row has the same four columns: `direction`, `before`, `after1`, and
`picked_tokens`.
