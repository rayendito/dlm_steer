# IMDB

This paired sentiment dataset is sampled from
[`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb).
Each CSV row contains one review from each sentiment:

- `text1`: positive review
- `text2`: negative review

The repository's validation file is exposed as the `train` split by
`timpa_load_rows`, while the test file is exposed as `test`.

## Splits

- `val.csv`: 10 pairs sampled from the upstream IMDB `train` split
- `test.csv`: 100 pairs sampled from the upstream IMDB `test` split

Using different upstream splits prevents validation and test examples from
overlapping. Sampling was performed without replacement using seed `20250725`.

## Length Stratification

Review length is measured in whitespace-separated words after cleaning. Each
sentiment is sampled independently across five word-count ranges:

| Word range | Validation per sentiment | Test per sentiment |
| --- | ---: | ---: |
| 20-50 | 2 | 20 |
| 51-100 | 2 | 20 |
| 101-150 | 2 | 20 |
| 151-200 | 2 | 20 |
| 201-300 | 2 | 20 |

This produces 10 positive and 10 negative validation reviews, plus 100
positive and 100 negative test reviews. No sampled review exceeds 300 words.

## Cleaning

HTML break tags such as `<br>`, `<br/>`, and `<br />` are converted to
newlines. Leading and trailing whitespace is removed from every line. The
central dataset loader applies the same normalization when reading the CSVs.
