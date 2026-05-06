/// `GEPAAdapter` trait and `EvaluationBatch` container.
///
/// This module defines the single integration point between user code and the
/// GEPA optimization engine.  Implementers provide an adapter that the engine
/// calls to evaluate candidates and build reflective datasets for mutation.
///
/// Mirrors the Python `gepa.core.adapter` module.
use std::collections::HashMap;

use async_trait::async_trait;
use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

/// A program candidate: mapping from component name to component text.
///
/// Keys are short names like `"instructions"` or `"refiner_prompt"`.
/// `IndexMap` preserves insertion order, matching the Python reference which
/// uses `list(seed_candidate.keys())` directly without sorting.
pub type Candidate = IndexMap<String, String>;

/// Container for the result of evaluating a candidate on a batch of data.
///
/// Generic over:
/// - `T`: the per-example trajectory type (opaque to the engine).
/// - `RO`: the per-example raw output type. It must be serializable so the
///   engine can preserve outputs in callbacks, cache entries, and best-output
///   tracking.
///
/// Correctness constraints:
/// - `outputs.len() == scores.len() == batch.len()`
/// - When `capture_traces = true` trajectories must be `Some` and have the
///   same length as `outputs`.
#[derive(Debug, Clone)]
pub struct EvaluationBatch<T, RO> {
    /// Raw per-example outputs produced by executing the candidate.
    /// GEPA serializes these for callbacks, cache entries, and best-output
    /// tracking, but does not otherwise interpret them.
    pub outputs: Vec<RO>,

    /// Per-example numeric scores.  Higher is better.  GEPA sums these for
    /// mini-batch acceptance and averages them for Pareto-front tracking.
    pub scores: Vec<f64>,

    /// Optional per-example execution traces.  Must be `Some` (and length-
    /// aligned with `outputs`) when the adapter was called with
    /// `capture_traces = true`.
    pub trajectories: Option<Vec<T>>,

    /// Optional per-example breakdown of named objective scores.
    /// `None` when the evaluator does not expose multi-objective metrics.
    pub objective_scores: Option<Vec<HashMap<String, f64>>>,
}

impl<T, RO> EvaluationBatch<T, RO> {
    /// Construct a batch with only the required fields.
    pub fn new(outputs: Vec<RO>, scores: Vec<f64>) -> Self {
        Self {
            outputs,
            scores,
            trajectories: None,
            objective_scores: None,
        }
    }

    /// Attach execution traces to this batch.
    pub fn with_trajectories(mut self, trajectories: Vec<T>) -> Self {
        self.trajectories = Some(trajectories);
        self
    }

    /// Attach per-example objective breakdowns to this batch.
    pub fn with_objective_scores(mut self, objective_scores: Vec<HashMap<String, f64>>) -> Self {
        self.objective_scores = Some(objective_scores);
        self
    }

    /// Validate that every vector in the batch is aligned with the requested
    /// batch size.
    ///
    /// # Errors
    /// Returns an evaluation error when outputs, scores, trajectories, or
    /// objective scores are length-misaligned, or when any score is not finite.
    pub fn validate_lengths(
        &self,
        expected_len: usize,
        require_trajectories: bool,
    ) -> crate::error::Result<()> {
        if self.outputs.len() != expected_len {
            return Err(crate::error::GEPAError::Evaluation(format!(
                "adapter returned {} outputs for a batch of {expected_len}",
                self.outputs.len()
            )));
        }

        if self.scores.len() != expected_len {
            return Err(crate::error::GEPAError::Evaluation(format!(
                "adapter returned {} scores for a batch of {expected_len}",
                self.scores.len()
            )));
        }

        if let Some((idx, score)) = self
            .scores
            .iter()
            .enumerate()
            .find(|(_, score)| !score.is_finite())
        {
            return Err(crate::error::GEPAError::Evaluation(format!(
                "adapter returned non-finite score at index {idx}: {score}"
            )));
        }

        match (&self.trajectories, require_trajectories) {
            (Some(trajectories), _) if trajectories.len() != expected_len => {
                return Err(crate::error::GEPAError::Evaluation(format!(
                    "adapter returned {} trajectories for a batch of {expected_len}",
                    trajectories.len()
                )));
            }
            (None, true) => {
                return Err(crate::error::GEPAError::Evaluation(
                    "adapter did not return trajectories for a trace-capturing evaluation".into(),
                ));
            }
            _ => {}
        }

        if let Some(objective_scores) = &self.objective_scores {
            if objective_scores.len() != expected_len {
                return Err(crate::error::GEPAError::Evaluation(format!(
                    "adapter returned {} objective-score rows for a batch of {expected_len}",
                    objective_scores.len()
                )));
            }

            for (idx, objectives) in objective_scores.iter().enumerate() {
                if let Some((name, score)) = objectives.iter().find(|(_, score)| !score.is_finite())
                {
                    return Err(crate::error::GEPAError::Evaluation(format!(
                        "adapter returned non-finite objective score at index {idx} for '{name}': {score}"
                    )));
                }
            }
        }

        Ok(())
    }
}

impl<T, RO> EvaluationBatch<T, RO>
where
    RO: Serialize,
{
    /// Convert raw outputs into JSON values, preserving order.
    ///
    /// # Errors
    /// Returns a serialization error if any output cannot be represented as
    /// JSON.
    pub fn outputs_as_json(&self) -> crate::error::Result<Vec<Value>> {
        self.outputs
            .iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

/// A generic reflective dataset: component name → list of JSON records.
///
/// Each record is a `serde_json::Value::Object` (or any valid JSON value).
/// The recommended schema per record is:
/// ```json
/// {
///   "Inputs": { ... },
///   "Generated Outputs": "...",
///   "Feedback": "..."
/// }
/// ```
pub type ReflectiveDataset = HashMap<String, Vec<Value>>;

/// Single integration point between user code and the GEPA engine.
///
/// Implementers supply two async methods:
///
/// 1. **`evaluate`** – execute the full program on a batch and return scores
///    and optional traces.
/// 2. **`make_reflective_dataset`** – given a completed evaluation, build a
///    compact per-component dataset for the teacher LM.
///
/// ### Error handling contract
/// Do **not** return an `Err` for individual example failures.  Instead,
/// return a valid `EvaluationBatch` with failure scores (e.g., `0.0`).
/// Reserve `Err` for unrecoverable systemic failures (network down, etc.).
///
/// ### Thread safety
/// The trait is `Send + Sync` so that adapters can be used across async tasks.
#[async_trait]
pub trait GEPAAdapter<DataInst, T, RO>: Send + Sync
where
    DataInst: Send,
    T: Send,
    RO: Send + Serialize,
{
    /// Execute the program defined by `candidate` on a batch of inputs.
    ///
    /// When `capture_traces` is `true`, `EvaluationBatch::trajectories` must
    /// be populated.
    ///
    /// # Errors
    /// Returns `Err` only for systemic/unrecoverable failures.  Individual
    /// example failures should produce a score of `0.0` in the output batch.
    async fn evaluate(
        &self,
        batch: &[DataInst],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> crate::error::Result<EvaluationBatch<T, RO>>;

    /// Build a per-component reflective dataset to drive instruction refinement.
    ///
    /// `eval_batch` must be the result of `evaluate(..., capture_traces=true)`
    /// on the same batch.  The returned map covers at least every component
    /// name in `components_to_update`.
    ///
    /// # Errors
    /// Returns `Err` for systemic failures only.
    async fn make_reflective_dataset(
        &self,
        candidate: &Candidate,
        eval_batch: &EvaluationBatch<T, RO>,
        components_to_update: &[String],
    ) -> crate::error::Result<ReflectiveDataset>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    // Minimal no-op adapter used only in unit tests.
    struct EchoAdapter;

    #[async_trait]
    impl GEPAAdapter<String, (), String> for EchoAdapter {
        async fn evaluate(
            &self,
            batch: &[String],
            _candidate: &Candidate,
            _capture_traces: bool,
        ) -> Result<EvaluationBatch<(), String>> {
            let n = batch.len();
            Ok(EvaluationBatch::new(batch.to_vec(), vec![1.0_f64; n]))
        }

        async fn make_reflective_dataset(
            &self,
            _candidate: &Candidate,
            _eval_batch: &EvaluationBatch<(), String>,
            components_to_update: &[String],
        ) -> Result<ReflectiveDataset> {
            Ok(components_to_update
                .iter()
                .map(|name| (name.clone(), vec![]))
                .collect())
        }
    }

    #[tokio::test]
    async fn evaluation_batch_round_trip() {
        let adapter = EchoAdapter;
        let mut candidate = Candidate::new();
        candidate.insert("instructions".into(), "Do the task.".into());

        let batch = vec!["hello".to_string(), "world".to_string()];
        let result = adapter
            .evaluate(&batch, &candidate, false)
            .await
            .expect("evaluation should succeed");

        assert_eq!(result.outputs.len(), 2);
        assert_eq!(result.scores, vec![1.0, 1.0]);
        assert!(result.trajectories.is_none());
        assert!(result.objective_scores.is_none());
    }

    #[tokio::test]
    async fn make_reflective_dataset_covers_requested_components() {
        let adapter = EchoAdapter;
        let candidate = Candidate::new();
        let batch = EvaluationBatch::new(vec!["out".to_string()], vec![0.5]);
        let components = vec!["instructions".to_string(), "refiner".to_string()];

        let dataset = adapter
            .make_reflective_dataset(&candidate, &batch, &components)
            .await
            .expect("reflective dataset should succeed");

        assert!(dataset.contains_key("instructions"));
        assert!(dataset.contains_key("refiner"));
    }

    #[test]
    fn evaluation_batch_builder_methods() {
        let batch: EvaluationBatch<String, i32> = EvaluationBatch::new(vec![1, 2], vec![0.5, 0.8])
            .with_trajectories(vec!["t1".into(), "t2".into()])
            .with_objective_scores(vec![
                HashMap::from([("accuracy".into(), 0.9_f64)]),
                HashMap::from([("accuracy".into(), 0.7_f64)]),
            ]);

        assert_eq!(batch.scores.len(), 2);
        assert!(batch.trajectories.is_some());
        assert!(batch.objective_scores.is_some());
    }

    // Ensure EvaluationBatch is Serialize/Deserialize via serde_json when
    // the concrete type params implement those traits.
    #[test]
    fn evaluation_batch_with_json_values() {
        let scores: Vec<f64> = vec![0.8, 0.9];
        let outputs: Vec<serde_json::Value> = vec![
            serde_json::json!({"answer": "yes"}),
            serde_json::json!({"answer": "no"}),
        ];
        let batch: EvaluationBatch<(), serde_json::Value> = EvaluationBatch::new(outputs, scores);
        assert_eq!(batch.scores[0], 0.8);
    }
}
