# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-29

Initial public release of the Rust implementation of the GEPA prompt
optimization algorithm.

### Added

#### Core Algorithm (Paper Algorithms 1-4)
- `GEPAEngine` implementing Algorithm 1: the main reflective evolutionary
  optimization loop with two-phase acceptance (minibatch check then full
  validation evaluation).
- `ReflectiveMutationProposer` implementing Algorithm 3: LLM-based reflective
  prompt mutation using execution traces and structured feedback.
- `MergeProposer` implementing Algorithm 4: system-aware crossover via
  common-ancestor detection and per-component merge logic.
- Per-instance Pareto frontier tracking (`GEPAState`) with four frontier
  strategies: `Instance`, `Objective`, `Hybrid`, and `Cartesian`.
- Algorithm 2 Pareto-based candidate selection with dominated-program removal
  and frequency-weighted sampling (`utils::pareto`).

#### Adapter Pattern
- `GEPAAdapter` async trait for bridging arbitrary AI systems to GEPA.
- `EvaluationBatch` for returning scores, outputs, trajectories, and
  multi-objective metrics from adapter evaluations.
- Optional `propose_new_texts` hook for adapter-driven proposal override.

#### Strategies (Pluggable Components)
- Candidate selectors: `ParetoCandidateSelector` (default),
  `CurrentBestSelector`, `EpsilonGreedySelector`, `TopKParetoCandidateSelector`.
- Component selectors: `RoundRobinSelector` (default, per-candidate counters),
  `AllComponentSelector`.
- Batch sampler: `EpochShuffledSampler` with deterministic seeded shuffling
  and epoch-boundary detection.
- Evaluation policy: `FullEvalPolicy` with coverage-aware tie-breaking.
- Instruction proposal: verbatim Appendix C meta-prompt from the GEPA paper,
  with backtick-block output extraction.

#### Language Model Client
- `OpenAICompatibleLM` supporting any OpenAI-compatible endpoint (OpenAI,
  Anthropic, Ollama, LMStudio, vLLM).
- Exponential-backoff retry with HTTP 429/5xx awareness.
- Streaming and non-streaming completion paths.
- API key redaction in `Debug` output and error body truncation (256 bytes).

#### State Management
- `GEPAState` with per-instance scoring, ancestry tracking, round-robin
  component counters, and evaluation caching.
- `EvaluationCache` with SHA-256-keyed `(candidate, example_id)` lookups,
  wired into the engine's full-valset evaluation path.
- `full_program_trace` for per-iteration debugging metadata.
- `best_outputs_valset` for tracking the best rollout output per validation
  example.
- JSON serialization/deserialization with schema versioning and atomic writes.
- Auto-save to `run_dir` and resume-from-disk on engine startup.

#### Stop Conditions
- `MaxMetricCallsStopper`, `MaxIterationsStopper`, `TimeoutStopper`.
- `FileStopper` (sentinel file), `SignalStopper` (atomic flag for signal
  handlers), `NoImprovementStopper` (patience-based early stopping).
- `CompositeStopper` with `Any`/`All` modes.

#### Callbacks and Tracking
- `GEPACallback` trait with 18 lifecycle event types covering optimization
  start/end, iteration lifecycle, candidate acceptance/rejection, merge
  events, Pareto front updates, and budget tracking.
- `CompositeCallback` for multi-listener dispatch.
- `ExperimentTracker` trait with `NoopTracker`, `WandbTracker` (stub), and
  `MlflowTracker` (stub) behind feature flags.

#### Configuration
- `OptimizeConfig` with sensible defaults matching the paper (minibatch
  size 3, Pareto selection, round-robin components, seed 0).
- `PromptTemplateConfig` supporting single or per-component prompt templates.
- `StopConditionConfig` for declarative budget/timeout/iteration limits.
- `LMConfig` for LM endpoint configuration.

#### Results
- `GEPAResult` frozen snapshot with all candidates, per-instance validation
  scores, Pareto frontier, ancestry lineage, and optional best outputs.
- `best_candidate_str()` convenience for single-component workflows.
- JSON serialization with schema versioning.

#### Data Loading
- `DataLoader` trait with `VecLoader` in-memory implementation.
- `IndexMap`-based `Candidate` type preserving insertion order.

#### Documentation
- Comprehensive README with quick start, architecture overview, and
  configuration reference.
- Two runnable examples (`quickstart`, `custom_adapter`).
- `CLAUDE.md` project guide for AI-assisted development.

### Notes

- This is a Rust port of the official Python GEPA reference implementation
  at <https://github.com/gepa-ai/gepa>, verified against the paper's
  Algorithms 1-4 and the reference's data structures.
- 162 tests, zero clippy warnings, zero unsafe code.
- Requires Rust 2024 edition.
