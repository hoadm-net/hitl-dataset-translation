# HITL Dataset Translation

A human-in-the-loop pipeline for translating Text-to-SQL benchmarks into low-resource languages, with Vietnamese as the primary target.

## Motivation

Large-scale Text-to-SQL datasets such as Spider and BIRD are available only in English, limiting research in low-resource language settings. Fully manual translation is prohibitively expensive, while fully automated translation via LLMs sacrifices semantic quality. This project proposes a human-in-the-loop approach: a small human-translated seed is used as few-shot context to guide LLM-based translation of the remaining data, followed by fine-tuning an open-source model to reduce API dependency.

## Datasets

| Dataset | Paper | Split |
|---|---|---|
| [Spider](https://yale-lily.github.io/spider) | Yu et al., EMNLP 2018 | train (8,659) / dev (1,034) / test (2,147) |
| [BIRD](https://bird-bench.github.io/) | Li et al., NeurIPS 2023 | train (9,428) / dev (1,534) |

See [`docs/spider.md`](docs/spider.md) and [`docs/bird.md`](docs/bird.md) for full dataset documentation.

## Pipeline (planned)

| Phase | Description |
|---|---|
| Phase 1 | Data extraction and preparation |
| Phase 2 | Manual translation — gold seed construction |
| Phase 3 | LLM-based translation with few-shot prompting |
| Phase 4 | Dataset assembly and translation model fine-tuning |
| Phase 5 | Downstream Text-to-SQL evaluation (EN vs VI) |

## Research Questions

1. Does human-in-the-loop seed translation improve quality over zero-shot LLM translation?
2. What is the minimum seed ratio (% of data) needed to bootstrap high-quality translation?
3. Can fine-tuning an open-source model on the translated data reduce dependence on commercial APIs?

## Requirements

- Python 3.10+
- GPU with CUDA (Phases 4 and 5)

## License

Spider is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). BIRD follows its [original license](https://bird-bench.github.io/). Vietnamese translations produced by this project are provided for research purposes only.
