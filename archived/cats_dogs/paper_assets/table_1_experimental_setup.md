| Decision | Choice | Reason |
| --- | --- | --- |
| Evaluation split | benchmarks/cats_dogs/train.csv | Matches steering sweep eval set |
| Evaluation size | 2100 | All available train rows; apples-to-apples baseline |
| Steering vector data | n=10 validation examples/class | Cheap vector setting used by final sweeps |
| Steering layer / alpha | layer 32, alpha 100 | Best available cats/dogs steering setup |
| Ablation variables | k, refill u, sampling temp, identify temp | Greedy search under compute limits |
| Sentence length | short/medium/long 1/3 quantile bins | Analysis only, not optimized |
| Classifier score | Qwen raw next-token P(cat/dog) | Matches existing sentiment scoring convention |
| Fluency score | Qwen perplexity | Lower is better; inverted before harmonic mean |
| Baseline | Instruction rewrite with masked diffusion continuation | Direct prompt alternative to activation steering |
