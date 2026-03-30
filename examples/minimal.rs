#![allow(clippy::all, clippy::pedantic)]
//! # Minimal GEPA Example
//!
//! The absolute simplest GEPA setup: a single-component prompt optimized
//! against a tiny dataset with a mock adapter. No API key needed.
//!
//! Run with: `cargo run --example minimal`

use std::sync::Arc;

use async_trait::async_trait;
use gepa::core::adapter::{Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset};
use gepa::core::data_loader::VecLoader;
use gepa::{LMConfig, OptimizeConfig, Result, StopConditionConfig};
use serde_json::json;

// ---------------------------------------------------------------------------
// A trivial adapter that scores candidates by keyword overlap.
// ---------------------------------------------------------------------------

struct KeywordAdapter;

#[async_trait]
impl GEPAAdapter<String, (), String> for KeywordAdapter {
    async fn evaluate(
        &self,
        batch: &[String],
        candidate: &Candidate,
        _capture_traces: bool,
    ) -> Result<EvaluationBatch<(), String>> {
        let instructions = candidate
            .get("prompt")
            .map_or("", String::as_str)
            .to_lowercase();
        let mut scores = Vec::new();
        let mut outputs = Vec::new();

        for example in batch {
            // Score: how many words from the example appear in the prompt.
            let words: Vec<&str> = example.split_whitespace().collect();
            let hits = words
                .iter()
                .filter(|w| instructions.contains(&w.to_lowercase()))
                .count();
            let score = hits as f64 / words.len().max(1) as f64;
            scores.push(score);
            outputs.push(format!("score={score:.2}"));
        }

        Ok(EvaluationBatch::new(outputs, scores))
    }

    async fn make_reflective_dataset(
        &self,
        _candidate: &Candidate,
        batch: &EvaluationBatch<(), String>,
        _components: &[String],
    ) -> Result<ReflectiveDataset> {
        let mut dataset = ReflectiveDataset::new();
        let records: Vec<serde_json::Value> = batch
            .scores
            .iter()
            .zip(batch.outputs.iter())
            .map(|(score, output)| {
                json!({
                    "score": score,
                    "output": output,
                    "feedback": if *score >= 0.5 { "Good overlap" } else { "Missing key terms" },
                })
            })
            .collect();
        dataset.insert("prompt".into(), records);
        Ok(dataset)
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Seed candidate.
    let mut seed = Candidate::new();
    seed.insert("prompt".into(), "Answer the question.".into());

    // Training data: questions about Rust.
    let trainset = Arc::new(VecLoader::new(vec![
        "What is ownership in Rust?".to_string(),
        "How do lifetimes work?".to_string(),
        "Explain the borrow checker.".to_string(),
    ]));

    let mut config = OptimizeConfig::new(
        seed,
        trainset.clone(),
        trainset, // reuse as valset for this minimal demo
        Arc::new(KeywordAdapter),
        LMConfig {
            model: "gpt-4o-mini".into(),
            api_key: String::new(),
            base_url: "http://localhost:19999".into(), // unreachable — adapter handles everything
            ..LMConfig::default()
        },
    );
    config.stop_condition = StopConditionConfig {
        max_iterations: Some(3),
        ..Default::default()
    };

    let result = gepa::optimize(config).await?;

    println!("Candidates explored: {}", result.num_candidates());
    println!(
        "Best score: {:.3}",
        result.val_aggregate_scores[result.best_idx()?]
    );
    println!("Best candidate: {:?}", result.best_candidate()?);

    Ok(())
}
