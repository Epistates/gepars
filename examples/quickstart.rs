#![allow(clippy::all, clippy::pedantic, unused_mut)]
//! # GEPA Quickstart — Sentiment Classification
//!
//! This example demonstrates the minimal end-to-end workflow for running GEPA
//! prompt optimisation:
//!
//!  1. Define your data item type.
//!  2. Implement `GEPAAdapter` — the single integration point with GEPA.
//!  3. Build train / validation sets with `VecLoader`.
//!  4. Configure and call `optimize()`.
//!  5. Inspect the returned `GEPAResult`.
//!
//! ## Running
//!
//! ```bash
//! # Dry run — uses the mock scorer, no API key required
//! cargo run --example quickstart
//!
//! # Live run — sends reflection calls to OpenAI
//! OPENAI_API_KEY=sk-... cargo run --example quickstart
//! ```
//!
//! When `OPENAI_API_KEY` is not set the example runs with a local no-op LM
//! endpoint so it compiles and runs without any credentials.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use gepa::core::data_loader::VecLoader;
use gepa::{
    Candidate, EvaluationBatch, GEPAAdapter, LMConfig, OptimizeConfig, ReflectiveDataset, Result,
    StopConditionConfig, optimize,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// Data type
// ---------------------------------------------------------------------------

/// A single training / validation example for a sentiment-classification task.
#[derive(Clone)]
struct SentimentExample {
    /// The text to classify.
    text: String,
    /// Ground-truth label: `"positive"` or `"negative"`.
    label: String,
}

impl SentimentExample {
    fn new(text: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            label: label.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/// Sentiment-classification adapter.
///
/// In a real deployment `evaluate` would call an LLM with the prompt stored in
/// `candidate["instructions"]` and score the response against `example.label`.
/// Here we use a deterministic mock so the example compiles and runs without
/// any API credentials.
struct SentimentAdapter;

#[async_trait]
impl GEPAAdapter<SentimentExample, String, String> for SentimentAdapter {
    /// Evaluate `candidate` on `batch` and return per-example scores.
    ///
    /// Scores are in `[0.0, 1.0]` — `1.0` for correct classification.
    async fn evaluate(
        &self,
        batch: &[SentimentExample],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> Result<EvaluationBatch<String, String>> {
        let instructions = candidate
            .get("instructions")
            .map(String::as_str)
            .unwrap_or("Classify the sentiment of the following text.");

        let mut outputs = Vec::with_capacity(batch.len());
        let mut scores = Vec::with_capacity(batch.len());
        let mut trajectories = Vec::with_capacity(batch.len());

        for example in batch {
            // --- Mock classifier -------------------------------------------
            // Replace this block with a real LLM call in production.
            // The mock uses a simple heuristic: if the prompt mentions the word
            // "carefully" the classifier is assumed to be more accurate.
            let prediction = mock_classify(&example.text, instructions);
            // ---------------------------------------------------------------

            let score = if prediction.trim().to_lowercase() == example.label.trim().to_lowercase() {
                1.0_f64
            } else {
                0.0_f64
            };

            let trace = format!(
                "instructions: {instructions}\ninput: {}\nprediction: {prediction}\nlabel: {}",
                example.text, example.label
            );

            outputs.push(prediction);
            scores.push(score);
            trajectories.push(trace);
        }

        let mut batch_result = EvaluationBatch::new(outputs, scores);

        // Trajectories are only required when `capture_traces` is true.
        if capture_traces {
            batch_result = batch_result.with_trajectories(trajectories);
        }

        Ok(batch_result)
    }

    /// Build a reflective dataset the mutation LM will use to propose improved
    /// instructions.
    ///
    /// The recommended record schema is:
    /// ```json
    /// { "Inputs": {...}, "Generated Outputs": "...", "Feedback": "..." }
    /// ```
    async fn make_reflective_dataset(
        &self,
        _candidate: &Candidate,
        eval_batch: &EvaluationBatch<String, String>,
        components: &[String],
    ) -> Result<ReflectiveDataset> {
        // Retrieve traces that were captured during evaluation.
        let traces = eval_batch.trajectories.as_deref().unwrap_or(&[]);

        let mut dataset: ReflectiveDataset = HashMap::new();

        for component in components {
            // Build one record per example where the model was wrong.
            let records = eval_batch
                .scores
                .iter()
                .zip(eval_batch.outputs.iter())
                .zip(traces.iter())
                .filter(|((score, _), _)| **score < 1.0)
                .map(|((_, output), trace)| {
                    json!({
                        "Inputs":            { "trace": trace },
                        "Generated Outputs": output,
                        "Feedback":          "The prediction was incorrect. Revise the instructions to be more precise.",
                    })
                })
                .collect::<Vec<_>>();

            dataset.insert(component.clone(), records);
        }

        Ok(dataset)
    }
}

// ---------------------------------------------------------------------------
// Mock classifier helper
// ---------------------------------------------------------------------------

/// Heuristic mock: words with strong positive/negative connotation determine
/// the label.  The `instructions` string can shift accuracy by including the
/// word "carefully" (simulating a better prompt).
fn mock_classify(text: &str, instructions: &str) -> String {
    let text_lower = text.to_lowercase();
    let boost = instructions.to_lowercase().contains("carefully");

    let positive_words = [
        "great",
        "good",
        "love",
        "excellent",
        "wonderful",
        "fantastic",
    ];
    let negative_words = ["bad", "terrible", "awful", "hate", "horrible", "dreadful"];

    let pos_count = positive_words
        .iter()
        .filter(|w| text_lower.contains(*w))
        .count();
    let neg_count = negative_words
        .iter()
        .filter(|w| text_lower.contains(*w))
        .count();

    // A "better" prompt (with "carefully") always resolves ties toward the
    // correct heuristic; a weaker prompt may flip edge cases.
    if pos_count > neg_count || (boost && pos_count == neg_count) {
        "positive".into()
    } else {
        "negative".into()
    }
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

fn make_train_set() -> Vec<SentimentExample> {
    vec![
        SentimentExample::new("This product is great!", "positive"),
        SentimentExample::new("Absolutely wonderful experience.", "positive"),
        SentimentExample::new("I love the new design.", "positive"),
        SentimentExample::new("Terrible quality, fell apart immediately.", "negative"),
        SentimentExample::new("I hate waiting in long queues.", "negative"),
        SentimentExample::new("Horrible customer service.", "negative"),
        SentimentExample::new("Excellent value for money.", "positive"),
        SentimentExample::new("The food was awful.", "negative"),
    ]
}

fn make_val_set() -> Vec<SentimentExample> {
    vec![
        SentimentExample::new("Fantastic performance on every task.", "positive"),
        SentimentExample::new("Dreadful experience from start to finish.", "negative"),
        SentimentExample::new("Good build quality and fast delivery.", "positive"),
        SentimentExample::new("Bad smell, returned immediately.", "negative"),
    ]
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialise tracing so engine progress is visible.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("gepa=info".parse().unwrap()),
        )
        .init();

    // ── Seed candidate ───────────────────────────────────────────────────────
    //
    // A candidate is a `Candidate` (IndexMap<String, String>) mapping component names to
    // their current text.  Simple single-prompt tasks typically have one
    // component called "instructions".

    let mut seed = Candidate::new();
    seed.insert(
        "instructions".into(),
        "Classify the sentiment of the following text as positive or negative.".into(),
    );

    // ── Data loaders ─────────────────────────────────────────────────────────
    //
    // `VecLoader` is the built-in in-memory loader.  IDs are zero-based usize
    // indices.  Implement `DataLoader` for database-backed or streaming loaders.

    let trainset = Arc::new(VecLoader::new(make_train_set()));
    let valset = Arc::new(VecLoader::new(make_val_set()));

    // ── LM configuration ─────────────────────────────────────────────────────
    //
    // If OPENAI_API_KEY is set, reflection calls go to OpenAI.  Otherwise we
    // point at a localhost endpoint that is assumed unreachable; the engine will
    // fail to obtain improved mutations but will still run and return the seed
    // candidate, demonstrating the full call path without credentials.

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let (base_url, model) = if api_key.is_empty() {
        // No key — use a local stub that won't be called when the reflective
        // dataset is empty.
        eprintln!(
            "OPENAI_API_KEY not set — running without live reflection calls.\n\
             Set the variable to enable full optimisation."
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

    let lm_config = LMConfig {
        model,
        api_key,
        base_url,
        temperature: Some(1.0),
        max_tokens: Some(2048),
        max_retries: 2,
    };

    // ── OptimizeConfig ───────────────────────────────────────────────────────
    //
    // Build with the four required arguments; override individual fields for
    // anything that differs from the default.

    let mut config = OptimizeConfig::new(
        seed,
        trainset,
        valset,
        Arc::new(SentimentAdapter),
        lm_config,
    );

    // Stop after 50 adapter evaluate() calls — appropriate for a quick demo.
    config.stop_condition = StopConditionConfig {
        max_metric_calls: Some(50),
        max_iterations: None,
        timeout: None,
    };

    // Use the default Pareto candidate selector and RoundRobin component selector.
    // config.candidate_selector = CandidateSelectorKind::Pareto;   // default
    // config.component_selector = ComponentSelectorKind::RoundRobin; // default

    // Each iteration samples 4 training examples (small for this demo).
    config.minibatch_size = 4;

    // Disable the merge proposer for this minimal example.
    config.use_merge = false;

    // Ask the engine to unwrap "instructions" from the best candidate dict.
    config.str_candidate_key = Some("instructions".into());

    // ── Run optimisation ─────────────────────────────────────────────────────

    println!("Starting GEPA optimisation...");
    let result = optimize(config).await?;

    // ── Report results ───────────────────────────────────────────────────────

    println!("\n=== Optimisation complete ===");
    println!("Candidates explored : {}", result.num_candidates());
    println!(
        "Total evaluate() calls : {}",
        result.total_metric_calls.unwrap_or(0)
    );
    println!(
        "Validation instances tracked : {}",
        result.num_val_instances()
    );

    if let Some(best_str) = result.best_candidate_str() {
        println!("\nBest instructions:\n  {best_str}");
    } else if let Ok(best_map) = result.best_candidate() {
        println!("\nBest candidate:");
        for (k, v) in best_map {
            println!("  {k}: {v}");
        }
    }

    if let Ok(idx) = result.best_idx() {
        let score = result.val_aggregate_scores[idx];
        println!("\nBest validation score: {score:.4}");
    }

    // Optionally serialise the full result to JSON for later analysis.
    // let json = result.to_json()?;
    // std::fs::write("gepa_result.json", json)?;

    Ok(())
}
