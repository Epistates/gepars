# gepa

[![Crates.io](https://img.shields.io/crates/v/gepa.svg)](https://crates.io/crates/gepa)
[![docs.rs](https://docs.rs/gepa/badge.svg)](https://docs.rs/gepa)
[![CI](https://github.com/nicholasjpaterno/gepa/actions/workflows/ci.yml/badge.svg)](https://github.com/nicholasjpaterno/gepa/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade Rust implementation of **GEPA** (Genetic-Pareto Prompt
Optimization), the algorithm described in
["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/pdf/2507.19457)
(ICLR 2026 Oral).  GEPA evolves prompt candidates through LLM-guided reflective
mutation and Pareto-front selection, consistently outperforming GRPO with far
fewer evaluations.

## Key results

- **+6% average improvement** over GRPO across standard benchmarks.
- **Up to 35x fewer rollouts** than conventional RL-based prompt tuning.
- Multi-objective Pareto tracking preserves solution diversity while focusing
  budget on hard examples.

## Quick start

Add the dependency:

```toml
[dependencies]
gepa = "0.1"
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"
```

Implement `GEPAAdapter` for your task, then call `optimize`:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use gepa::{
    optimize, Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset,
    LMConfig, OptimizeConfig, StopConditionConfig, Result,
};
use gepa::core::data_loader::VecLoader;

// --- Data types -----------------------------------------------------------

#[derive(Clone)]
struct Example {
    question: String,
    answer: String,
}

// --- Adapter --------------------------------------------------------------

struct QAAdapter;

#[async_trait]
impl GEPAAdapter<Example, (), String> for QAAdapter {
    async fn evaluate(
        &self,
        batch: &[Example],
        candidate: &Candidate,
        _capture_traces: bool,
    ) -> Result<EvaluationBatch<(), String>> {
        let prompt_template = candidate
            .get("instructions")
            .map(String::as_str)
            .unwrap_or("Answer the following question:");

        let mut outputs = Vec::with_capacity(batch.len());
        let mut scores = Vec::with_capacity(batch.len());

        for example in batch {
            // Replace with your actual LLM call.
            let response = call_your_llm(prompt_template, &example.question).await;
            let score = if response.trim() == example.answer.trim() { 1.0 } else { 0.0 };
            outputs.push(response);
            scores.push(score);
        }

        Ok(EvaluationBatch::new(outputs, scores))
    }

    async fn make_reflective_dataset(
        &self,
        candidate: &Candidate,
        eval_batch: &EvaluationBatch<(), String>,
        components: &[String],
    ) -> Result<ReflectiveDataset> {
        // Return an empty dataset; the engine will skip reflection this round.
        Ok(components.iter().map(|k| (k.clone(), vec![])).collect())
    }
}

// --- Main -----------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let train = vec![
        Example { question: "2+2".into(), answer: "4".into() },
        Example { question: "3+3".into(), answer: "6".into() },
    ];
    let val = vec![
        Example { question: "4+4".into(), answer: "8".into() },
    ];

    let mut seed = HashMap::new();
    seed.insert("instructions".into(), "Answer the following question:".into());

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let config = OptimizeConfig::new(
        seed,
        Arc::new(VecLoader::new(train)),
        Arc::new(VecLoader::new(val)),
        Arc::new(QAAdapter),
        LMConfig {
            model: "gpt-4o-mini".into(),
            api_key,
            base_url: "https://api.openai.com".into(),
            ..LMConfig::default()
        },
    );

    let result = optimize(config).await?;

    println!("Explored {} candidates", result.num_candidates());
    if let Ok(best) = result.best_candidate() {
        println!("Best instructions: {}", best["instructions"]);
    }

    Ok(())
}

async fn call_your_llm(_prompt: &str, _question: &str) -> String {
    "stub".into()
}
```

A complete, runnable example is provided in [`examples/quickstart.rs`](examples/quickstart.rs).

## Features

- **Per-instance Pareto frontier** — tracks which candidate performs best on
  each individual validation example, mirroring Algorithm 2 from the paper.
- **Reflective mutation** — uses an LLM to analyse failure cases and propose
  improved instruction text (Appendix C prompt template included).
- **System-aware merge** — periodically merges complementary Pareto candidates
  by combining high-performing per-instance components (Algorithm 4).
- **Provider-agnostic** — any server that speaks the OpenAI
  `/v1/chat/completions` protocol is supported out of the box via `reqwest`.
- **Pluggable strategies** — swap candidate selectors (`Pareto`, `CurrentBest`,
  `EpsilonGreedy`), component selectors (`RoundRobin`, `All`), and frontier
  types (`Instance`, `Objective`, `Hybrid`, `Cartesian`) without touching the
  engine.
- **Observable** — structured `tracing` logs and a typed callback system
  (`GEPACallback`) for custom metrics, checkpointing, and early-stopping hooks.
- **Serialisable results** — `GEPAResult` round-trips through JSON; schema
  versioning guards against incompatible future formats.
- **No unsafe code** — `unsafe_code = "forbid"` is enforced at the crate level.

## Architecture

```
gepa
├── api.rs          optimize() entry point, OptimizeConfig, LMConfig
├── lm.rs           LanguageModel trait, OpenAICompatibleLM
├── error.rs        GEPAError, Result
├── core/
│   ├── adapter.rs  GEPAAdapter trait, Candidate, EvaluationBatch
│   ├── engine.rs   GEPAEngine — the main optimisation loop
│   ├── result.rs   GEPAResult — immutable run snapshot
│   ├── state.rs    GEPAState, FrontierType, Pareto bookkeeping
│   ├── data_loader.rs  DataLoader, VecLoader
│   └── callbacks.rs    GEPACallback, event structs
├── proposer/
│   ├── reflective_mutation.rs  LLM-guided mutation (Algorithm 3)
│   └── merge.rs                system-aware merge (Algorithm 4)
├── strategies/
│   ├── candidate_selector.rs   Pareto / CurrentBest / EpsilonGreedy
│   ├── component_selector.rs   RoundRobin / All
│   ├── batch_sampler.rs        EpochShuffledSampler
│   └── eval_policy.rs          FullEvalPolicy
└── utils/
    ├── stop_condition.rs       MaxMetricCallsStopper, TimeoutStopper, …
    └── pareto.rs               Pareto utilities
```

The central abstraction is `GEPAAdapter`:

```
Your code  ──[GEPAAdapter]──  GEPAEngine  ──[LanguageModel]──  LLM API
```

The engine is responsible for all Pareto bookkeeping, candidate selection,
budget tracking, and mutation orchestration.  Your adapter handles only two
things: evaluating a batch of examples and, optionally, building a structured
reflective dataset for the mutation LM.

## Configuration

`OptimizeConfig::new` accepts five required arguments and exposes every other
option as a public field with sensible defaults.

### Required

| Field | Type | Description |
|---|---|---|
| `seed_candidate` | `HashMap<String, String>` | Starting prompt(s) by component name |
| `trainset` | `Arc<dyn DataLoader<Id, Item>>` | Training split |
| `valset` | `Arc<dyn DataLoader<Id, Item>>` | Validation split (Pareto tracking) |
| `adapter` | `Arc<dyn GEPAAdapter<Item, T, RO>>` | Your evaluation logic |
| `lm_config` | `LMConfig` | Reflection LM settings |

### Stop condition (`StopConditionConfig`)

| Field | Default | Description |
|---|---|---|
| `max_metric_calls` | `Some(500)` | Budget in adapter `evaluate()` calls |
| `max_iterations` | `None` | Hard iteration cap |
| `timeout` | `None` | Wall-clock limit (`std::time::Duration`) |

All active conditions are combined with `OR` — the first to fire stops the run.

### Strategy knobs

| Field | Default | Description |
|---|---|---|
| `candidate_selector` | `CandidateSelectorKind::Pareto` | How to pick a base candidate |
| `component_selector` | `ComponentSelectorKind::RoundRobin` | Which prompt components to mutate |
| `minibatch_size` | `16` | Training examples per iteration |
| `frontier_type` | `FrontierType::Instance` | Pareto tracking strategy |
| `use_merge` | `true` | Enable system-aware merge |
| `max_merge_invocations` | `20` | Merge budget across the run |

### LM settings (`LMConfig`)

| Field | Default | Description |
|---|---|---|
| `model` | `"gpt-4o-mini"` | Model identifier |
| `api_key` | `""` | Bearer token (`""` for local / unauthenticated servers) |
| `base_url` | `"https://api.openai.com"` | API base URL (no trailing slash) |
| `temperature` | `Some(1.0)` | Sampling temperature |
| `max_tokens` | `Some(4096)` | Max tokens for reflection outputs |
| `max_retries` | `3` | HTTP retries with exponential back-off |

## Supported LLM providers

`OpenAICompatibleLM` calls the standard `/v1/chat/completions` endpoint.
Point `base_url` at any compatible server:

| Provider | `base_url` |
|---|---|
| OpenAI | `https://api.openai.com` |
| Anthropic (OpenAI shim) | `https://api.anthropic.com` |
| Ollama | `http://localhost:11434` |
| LMStudio | `http://localhost:1234` |
| vLLM | `http://localhost:8000` |
| Any OpenAI-compatible | your endpoint |

Pass `api_key: ""` for unauthenticated local servers.

## Examples

```bash
# Quickstart — sentiment classification with a mock scorer
cargo run --example quickstart

# Custom adapter — multi-component prompt with merge enabled
cargo run --example custom_adapter

# Live API (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... cargo run --example quickstart -- --live
```

## References

- Paper: [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/pdf/2507.19457) (ICLR 2026 Oral)
- Python reference implementation: [gepa-py](https://github.com/epistates/gepa)
- API documentation: [docs.rs/gepa](https://docs.rs/gepa)
- Original GEPA repo: https://github.com/gepa-ai/gepa

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{gepa2026,
  title     = {{GEPA}: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Oral presentation},
  url       = {https://arxiv.org/pdf/2507.19457},
}
```
