# Dolly Sample

These splits contain neutral assistant responses selected from
`databricks/databricks-dolly-15k` revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` with seed `20260722`.

- `val.csv` contains 10 curated responses for readable smoke tests.
- `test.csv` contains a seeded sample of 100 responses, disjoint from validation.
- `id` is the one-based JSONL line number in the source revision.
- Only `open_qa` and `general_qa` responses between 45 and 140 words were
  eligible.
- Lists, multiline responses, URLs, first-person narration, and source prompts
  requesting a particular persona or writing style were excluded.

The resulting text is intended to represent a neutral assistant response whose
content can be preserved while TIMPA changes its style to pirate, mean, or
flirty.
