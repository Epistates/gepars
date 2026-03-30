#![allow(clippy::all, clippy::pedantic, unused_mut)]
//! # Custom Adapter — Multi-component Prompt with Merge and Frontier Variants
//!
//! This example shows more advanced GEPA usage:
//!
//!  - Implementing `GEPAAdapter` for a task with **two prompt components**
//!    (`"system_prompt"` and `"refiner_prompt"`).
//!  - Capturing **trajectories** from `evaluate()` so `make_reflective_dataset`
//!    can build structured per-component feedback.
//!  - Enabling the **merge proposer** (`use_merge: true`) which fuses
//!    complementary Pareto candidates.
//!  - Switching the **frontier type** to `FrontierType::Objective` for
//!    named-metric Pareto tracking.
//!  - Using `EpsilonGreedySelector` for exploration-exploitation balancing.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example custom_adapter
//! OPENAI_API_KEY=sk-... cargo run --example custom_adapter
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use gepa::core::data_loader::VecLoader;
use gepa::core::state::FrontierType;
use gepa::{
    Candidate, CandidateSelectorKind, ComponentSelectorKind, EvaluationBatch, GEPAAdapter,
    LMConfig, OptimizeConfig, ReflectiveDataset, Result, StopConditionConfig, optimize,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A code-generation example: given a natural-language description, produce
/// a Python snippet.
#[derive(Clone)]
struct CodeExample {
    /// Natural-language description of the task.
    description: String,
    /// Expected Python code (used for scoring).
    expected_code: String,
}

impl CodeExample {
    fn new(description: impl Into<String>, expected_code: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            expected_code: expected_code.into(),
        }
    }
}

/// Per-example execution trace — records what each component contributed.
#[derive(Clone)]
#[allow(dead_code)] // fields used in make_reflective_dataset via serde
struct CodeTrace {
    /// The exact system prompt used.
    system_prompt_used: String,
    /// The exact refiner prompt used.
    refiner_prompt_used: String,
    /// The raw LLM output before refinement.
    raw_output: String,
    /// The final refined output.
    refined_output: String,
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// Code-generation adapter with two optimisable components:
///
/// - `"system_prompt"` — the high-level coding-assistant persona.
/// - `"refiner_prompt"` — a second-pass prompt that cleans up the initial code.
///
/// In production both components would drive separate LLM calls.  Here the
/// second pass is simulated deterministically.
struct CodeGenAdapter;

#[async_trait]
impl GEPAAdapter<CodeExample, CodeTrace, String> for CodeGenAdapter {
    async fn evaluate(
        &self,
        batch: &[CodeExample],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> Result<EvaluationBatch<CodeTrace, String>> {
        let system_prompt = candidate
            .get("system_prompt")
            .map(String::as_str)
            .unwrap_or("You are a Python expert. Write clean, idiomatic code.");

        let refiner_prompt = candidate
            .get("refiner_prompt")
            .map(String::as_str)
            .unwrap_or("Review the code above for correctness and style.");

        let mut outputs = Vec::with_capacity(batch.len());
        let mut scores = Vec::with_capacity(batch.len());
        let mut traces: Vec<CodeTrace> = Vec::with_capacity(batch.len());
        let mut objective_scores: Vec<HashMap<String, f64>> = Vec::with_capacity(batch.len());

        for example in batch {
            // --- Mock generation -------------------------------------------
            // Replace with real LLM calls in production.
            let raw = mock_generate(system_prompt, &example.description);
            let refined = mock_refine(refiner_prompt, &raw);
            // ---------------------------------------------------------------

            // Compute two independent objectives:
            // 1. correctness — does the output contain the expected tokens?
            // 2. brevity     — prefer shorter code (inverse length penalty)
            let correctness = compute_correctness(&refined, &example.expected_code);
            let brevity = compute_brevity(&refined);

            // The aggregate score the engine uses for Pareto ranking.
            let aggregate = 0.7 * correctness + 0.3 * brevity;

            let mut obj = HashMap::new();
            obj.insert("correctness".into(), correctness);
            obj.insert("brevity".into(), brevity);

            outputs.push(refined.clone());
            scores.push(aggregate);
            objective_scores.push(obj);

            if capture_traces {
                traces.push(CodeTrace {
                    system_prompt_used: system_prompt.to_string(),
                    refiner_prompt_used: refiner_prompt.to_string(),
                    raw_output: raw,
                    refined_output: refined,
                });
            }
        }

        let mut batch_result =
            EvaluationBatch::new(outputs, scores).with_objective_scores(objective_scores);

        if capture_traces {
            batch_result = batch_result.with_trajectories(traces);
        }

        Ok(batch_result)
    }

    /// Build per-component reflective datasets.
    ///
    /// For `"system_prompt"` we include failures where `correctness` was low.
    /// For `"refiner_prompt"` we include cases where `brevity` was low — i.e.,
    /// the refiner failed to tighten the code.
    async fn make_reflective_dataset(
        &self,
        candidate: &Candidate,
        eval_batch: &EvaluationBatch<CodeTrace, String>,
        components: &[String],
    ) -> Result<ReflectiveDataset> {
        let traces = match &eval_batch.trajectories {
            Some(t) => t,
            None => {
                // Traces were not captured — return empty records to skip this
                // reflection round gracefully.
                return Ok(components.iter().map(|k| (k.clone(), vec![])).collect());
            }
        };

        let objectives = eval_batch.objective_scores.as_deref().unwrap_or(&[]);

        let mut dataset: ReflectiveDataset = HashMap::new();

        for component in components {
            let records: Vec<serde_json::Value> = traces
                .iter()
                .zip(eval_batch.outputs.iter())
                .zip(objectives.iter())
                .filter_map(|((trace, output), obj)| {
                    match component.as_str() {
                        "system_prompt" => {
                            // Include examples where correctness was below threshold.
                            let correctness = obj.get("correctness").copied().unwrap_or(0.0);
                            if correctness < 0.7 {
                                Some(json!({
                                    "Inputs": {
                                        "system_prompt": trace.system_prompt_used,
                                        "task_description": "(see raw_output context)",
                                        "refined": trace.refined_output,
                                    },
                                    "Generated Outputs": trace.raw_output,
                                    "Feedback": format!(
                                        "Correctness score was {correctness:.2}. \
                                         The system prompt should be more precise about \
                                         the expected code structure."
                                    ),
                                }))
                            } else {
                                None
                            }
                        }
                        "refiner_prompt" => {
                            // Include examples where brevity was below threshold.
                            let brevity = obj.get("brevity").copied().unwrap_or(0.0);
                            if brevity < 0.5 {
                                Some(json!({
                                    "Inputs": {
                                        "refiner_prompt": trace.refiner_prompt_used,
                                        "unrefined_output": trace.raw_output,
                                    },
                                    "Generated Outputs": output,
                                    "Feedback": format!(
                                        "Brevity score was {brevity:.2}. \
                                         The refiner prompt should explicitly ask for \
                                         removing redundant lines and boilerplate."
                                    ),
                                }))
                            } else {
                                None
                            }
                        }
                        _ => {
                            // Generic fallback for any additional component.
                            Some(json!({
                                "Inputs": { "component": component },
                                "Generated Outputs": output,
                                "Feedback": "Score below target. Revise instructions.",
                            }))
                        }
                    }
                })
                .collect();

            dataset.insert(component.clone(), records);
        }

        // Suppress unused-variable warning for `candidate`.
        let _ = candidate;

        Ok(dataset)
    }
}

// ---------------------------------------------------------------------------
// Scoring helpers (mock implementations)
// ---------------------------------------------------------------------------

fn mock_generate(system_prompt: &str, description: &str) -> String {
    // Longer, more verbose output when the system prompt is generic.
    let verbose = !system_prompt.to_lowercase().contains("idiomatic");
    if verbose {
        format!(
            "# generated code\n# description: {description}\ndef solve():\n    pass\n\nif __name__ == '__main__':\n    solve()\n"
        )
    } else {
        format!("def solve():\n    # {description}\n    pass\n")
    }
}

fn mock_refine(refiner_prompt: &str, code: &str) -> String {
    // Simulate the refiner stripping boilerplate if explicitly asked.
    if refiner_prompt.to_lowercase().contains("boilerplate") {
        code.lines()
            .filter(|l| !l.starts_with("if __name__"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        code.to_string()
    }
}

fn compute_correctness(output: &str, expected: &str) -> f64 {
    // Token-overlap proxy for correctness.
    let out_tokens: std::collections::HashSet<&str> = output.split_whitespace().collect();
    let exp_tokens: std::collections::HashSet<&str> = expected.split_whitespace().collect();
    if exp_tokens.is_empty() {
        return 1.0;
    }
    let overlap = out_tokens.intersection(&exp_tokens).count();
    overlap as f64 / exp_tokens.len() as f64
}

fn compute_brevity(output: &str) -> f64 {
    // Normalised inverse line count — fewer lines → higher score.
    let lines = output.lines().count().max(1);
    (1.0_f64 / lines as f64).min(1.0)
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

fn make_train() -> Vec<CodeExample> {
    vec![
        CodeExample::new(
            "Return the sum of a list of integers.",
            "def total(nums):\n    return sum(nums)\n",
        ),
        CodeExample::new("Reverse a string.", "def reverse(s):\n    return s[::-1]\n"),
        CodeExample::new(
            "Check whether a number is prime.",
            "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\n",
        ),
        CodeExample::new(
            "Flatten a nested list one level.",
            "def flatten(lst):\n    return [x for sub in lst for x in sub]\n",
        ),
    ]
}

fn make_val() -> Vec<CodeExample> {
    vec![
        CodeExample::new(
            "Count word frequencies in a string.",
            "from collections import Counter\ndef word_freq(text):\n    return Counter(text.split())\n",
        ),
        CodeExample::new(
            "Remove duplicates from a list preserving order.",
            "def dedup(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]\n",
        ),
    ]
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("gepa=info".parse().unwrap()),
        )
        .init();

    // ── Multi-component seed candidate ───────────────────────────────────────
    //
    // GEPA can optimise multiple prompt components simultaneously.  List every
    // component that should be subject to mutation.

    let mut seed = Candidate::new();
    seed.insert(
        "system_prompt".into(),
        "You are a helpful assistant. Write Python code to solve the task.".into(),
    );
    seed.insert(
        "refiner_prompt".into(),
        "Review the code above and improve it for correctness and readability.".into(),
    );

    // ── Data loaders ─────────────────────────────────────────────────────────

    let trainset = Arc::new(VecLoader::new(make_train()));
    let valset = Arc::new(VecLoader::new(make_val()));

    // ── LM configuration ─────────────────────────────────────────────────────

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let (base_url, model) = if api_key.is_empty() {
        eprintln!(
            "OPENAI_API_KEY not set — mutation LM calls will not produce new candidates.\n\
             The example still exercises the full adapter / Pareto logic."
        );
        (
            "http://localhost:19999".to_string(),
            "stub-model".to_string(),
        )
    } else {
        (
            "https://api.openai.com".to_string(),
            "gpt-4o-mini".to_string(),
        )
    };

    // ── OptimizeConfig ───────────────────────────────────────────────────────

    let mut config = OptimizeConfig::new(
        seed,
        trainset,
        valset,
        Arc::new(CodeGenAdapter),
        LMConfig {
            model,
            api_key,
            base_url,
            temperature: Some(0.9),
            max_tokens: Some(2048),
            max_retries: 2,
        },
    );

    // Stop after 80 evaluate() calls.
    config.stop_condition = StopConditionConfig {
        max_metric_calls: Some(80),
        max_iterations: None,
        timeout: None,
    };

    // ---- Candidate selector ------------------------------------------------
    //
    // `EpsilonGreedy` exploits the current best candidate with probability
    // 1 - epsilon and explores a random frontier candidate otherwise.  This
    // is useful early in search when the frontier is sparse.
    config.candidate_selector = CandidateSelectorKind::EpsilonGreedy;
    config.epsilon = 0.2;

    // ---- Component selector ------------------------------------------------
    //
    // `RoundRobin` advances one component per iteration:
    //   iteration 0 → "system_prompt"
    //   iteration 1 → "refiner_prompt"
    //   iteration 2 → "system_prompt", …
    //
    // Switch to `All` to mutate every component simultaneously (more expensive
    // per iteration but fewer iterations needed for compound improvements).
    config.component_selector = ComponentSelectorKind::RoundRobin;

    // ---- Merge proposer ----------------------------------------------------
    //
    // When enabled, GEPA periodically identifies two Pareto candidates that
    // excel on non-overlapping validation examples and combines their
    // best-performing components into a new candidate (Algorithm 4).
    config.use_merge = true;
    config.max_merge_invocations = 5;
    config.val_overlap_floor = 1;

    // ---- Frontier type -----------------------------------------------------
    //
    // `FrontierType::Objective` maintains a separate Pareto front for each
    // named objective returned in `EvaluationBatch::objective_scores`.
    // Use `FrontierType::Instance` (default) for per-example tracking, or
    // `FrontierType::Hybrid` for both simultaneously.
    config.frontier_type = FrontierType::Objective;

    config.minibatch_size = 3;

    // ── Run optimisation ─────────────────────────────────────────────────────

    println!("Starting multi-component GEPA optimisation…");
    let result = optimize(config).await?;

    // ── Report results ───────────────────────────────────────────────────────

    println!("\n=== Optimisation complete ===");
    println!("Candidates explored      : {}", result.num_candidates());
    println!(
        "Total evaluate() calls   : {}",
        result.total_metric_calls.unwrap_or(0)
    );
    println!(
        "Full validation evals    : {}",
        result.num_full_val_evals.unwrap_or(0)
    );

    if let Ok(best) = result.best_candidate() {
        println!("\nBest candidate:");
        for (component, text) in best {
            println!("  [{component}]\n  {text}\n");
        }
    }

    if let Ok(idx) = result.best_idx() {
        let aggregate = result.val_aggregate_scores[idx];
        println!("Best aggregate validation score: {aggregate:.4}");

        // Multi-objective breakdown (available when objective_scores were set).
        if let Some(ref obj_scores) = result.val_aggregate_subscores {
            if let Some(scores) = obj_scores.get(idx) {
                for (name, value) in scores {
                    println!("  {name}: {value:.4}");
                }
            }
        }
    }

    // Per-objective Pareto front: which candidate was best for each metric?
    if let Some(ref per_obj) = result.per_objective_best_candidates {
        println!("\nPer-objective Pareto winners:");
        for (objective, candidate_set) in per_obj {
            let indices: Vec<_> = candidate_set.iter().collect();
            println!("  {objective}: candidates {indices:?}");
        }
    }

    // Serialise to JSON for offline analysis or result caching.
    // let json = result.to_json()?;
    // std::fs::write("code_gen_result.json", json)?;
    // println!("\nFull result saved to code_gen_result.json");

    Ok(())
}
