#![allow(clippy::all, clippy::pedantic)]
//! # Merge Demo
//!
//! Demonstrates GEPA's Algorithm 4 (System-Aware Merge): after reflective
//! mutations produce specialized candidates, merge combines complementary
//! prompt components from different lineages.
//!
//! Run with: `cargo run --example merge_demo`

use std::sync::Arc;

use async_trait::async_trait;
use gepa::core::adapter::{Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset};
use gepa::core::data_loader::VecLoader;
use gepa::core::state::FrontierType;
use gepa::{LMConfig, OptimizeConfig, Result, StopConditionConfig};
use serde_json::json;

// ---------------------------------------------------------------------------
// Dataset: math word problems with two steps (parse + solve).
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MathProblem {
    question: String,
    answer: f64,
}

// ---------------------------------------------------------------------------
// Adapter: scores based on how well the two-component prompt guides the task.
// ---------------------------------------------------------------------------

struct MathAdapter;

#[async_trait]
impl GEPAAdapter<MathProblem, String, String> for MathAdapter {
    async fn evaluate(
        &self,
        batch: &[MathProblem],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> Result<EvaluationBatch<String, String>> {
        let parser_prompt = candidate.get("parser").map_or("", String::as_str);
        let solver_prompt = candidate.get("solver").map_or("", String::as_str);

        let mut scores = Vec::new();
        let mut outputs = Vec::new();
        let mut traces = if capture_traces {
            Some(Vec::new())
        } else {
            None
        };

        for problem in batch {
            // Simulate: longer, more specific prompts score better.
            let parser_quality = (parser_prompt.len() as f64 / 200.0).min(1.0);
            let solver_quality = (solver_prompt.len() as f64 / 200.0).min(1.0);
            let score = (parser_quality + solver_quality) / 2.0;

            scores.push(score);
            outputs.push(format!("answer={:.1}", problem.answer));

            if let Some(ref mut t) = traces {
                t.push(format!(
                    "parser({:.2}) + solver({:.2}) on: {}",
                    parser_quality, solver_quality, problem.question
                ));
            }
        }

        let mut batch_result = EvaluationBatch::new(outputs, scores);
        if let Some(t) = traces {
            batch_result = batch_result.with_trajectories(t);
        }
        Ok(batch_result)
    }

    async fn make_reflective_dataset(
        &self,
        _candidate: &Candidate,
        batch: &EvaluationBatch<String, String>,
        components: &[String],
    ) -> Result<ReflectiveDataset> {
        let mut dataset = ReflectiveDataset::new();
        for component in components {
            let records: Vec<serde_json::Value> = batch
                .scores
                .iter()
                .enumerate()
                .map(|(i, score)| {
                    json!({
                        "component": component,
                        "score": score,
                        "trace": batch.trajectories.as_ref().map(|t| &t[i]),
                        "feedback": if *score > 0.5 { "Adequate" } else { "Needs more detail" },
                    })
                })
                .collect();
            dataset.insert(component.clone(), records);
        }
        Ok(dataset)
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("gepa=info")
        .init();

    // Two-component seed candidate.
    let mut seed = Candidate::new();
    seed.insert("parser".into(), "Parse the math problem.".into());
    seed.insert("solver".into(), "Solve the parsed equation.".into());

    let problems = vec![
        MathProblem {
            question: "If 3 apples cost $6, how much do 5 cost?".into(),
            answer: 10.0,
        },
        MathProblem {
            question: "A train travels 60mph for 2.5 hours. Distance?".into(),
            answer: 150.0,
        },
        MathProblem {
            question: "What is 15% of 200?".into(),
            answer: 30.0,
        },
        MathProblem {
            question: "A rectangle is 8x5. What is the perimeter?".into(),
            answer: 26.0,
        },
        MathProblem {
            question: "If x + 7 = 12, what is x?".into(),
            answer: 5.0,
        },
        MathProblem {
            question: "Convert 3/4 to a decimal.".into(),
            answer: 0.75,
        },
    ];

    let trainset = Arc::new(VecLoader::new(problems.clone()));
    let valset = Arc::new(VecLoader::new(problems));

    let mut config = OptimizeConfig::new(
        seed,
        trainset,
        valset,
        Arc::new(MathAdapter),
        LMConfig {
            model: "gpt-4o-mini".into(),
            api_key: String::new(),
            base_url: "http://localhost:19999".into(),
            ..LMConfig::default()
        },
    );
    // Enable merge (Algorithm 4).
    config.use_merge = true;
    config.max_merge_invocations = 5;
    config.val_overlap_floor = 2;
    config.frontier_type = FrontierType::Instance;
    config.stop_condition = StopConditionConfig {
        max_iterations: Some(5),
        ..Default::default()
    };

    let result = gepa::optimize(config).await?;

    println!("--- Merge Demo Results ---");
    println!("Candidates explored: {}", result.num_candidates());
    println!("Best candidate index: {}", result.best_idx()?);
    println!(
        "Best score: {:.4}",
        result.val_aggregate_scores[result.best_idx()?]
    );

    let best = result.best_candidate()?;
    for (component, text) in best {
        println!("\n[{component}]:\n  {text}");
    }

    Ok(())
}
