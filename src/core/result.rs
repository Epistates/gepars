/// `GEPAResult` — immutable snapshot returned at the end of optimisation.
///
/// Captures everything worth knowing about a completed GEPA run: all explored
/// candidates, their validation scores, the Pareto frontier, lineage, and
/// optional raw best outputs.  It is the sole public-facing artefact of the
/// optimisation engine.
///
/// Mirrors the Python `gepa.core.result` module.
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::core::adapter::Candidate;
use crate::core::data_loader::DataId;
use crate::core::state::{GEPAState, ProgramIdx};
use crate::error::Result;

// ---------------------------------------------------------------------------
// Serde helpers — shared with state.rs; serialise HashMap<Id,V> as Vec pairs
// to avoid requiring Id to be a JSON string key.
// ---------------------------------------------------------------------------

use crate::core::serde_helpers::{serde_map_set_as_vec, serde_vec_of_maps};

// ---------------------------------------------------------------------------
// GEPAResult
// ---------------------------------------------------------------------------

/// Immutable snapshot of a completed (or in-progress) GEPA optimisation run.
///
/// ### Serialisation
/// [`GEPAResult`] implements `serde::Serialize` / `Deserialize`.  Round-trip
/// via `to_json()` / `from_json()`.
///
/// ### Construction
/// Prefer [`GEPAResult::from_state`] rather than constructing directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "Id: DataId"))]
pub struct GEPAResult<Id: DataId> {
    // ---- Core data -------------------------------------------------------
    /// All explored candidates in discovery order.
    pub candidates: Vec<Candidate>,

    /// Lineage — `parents[i]` lists parent indices for candidate `i`.
    /// The seed candidate has `[None]` in its parent list.
    pub parents: Vec<Vec<Option<ProgramIdx>>>,

    /// Per-candidate average validation score.
    /// `f64::NEG_INFINITY` for unseen candidates.
    pub val_aggregate_scores: Vec<f64>,

    /// Per-candidate dict of `{val_id: score}` for each validated example.
    #[serde(
        serialize_with = "serde_vec_of_maps::serialize",
        deserialize_with = "serde_vec_of_maps::deserialize"
    )]
    pub val_subscores: Vec<HashMap<Id, f64>>,

    /// Instance-level Pareto frontier — for each validation example, the set
    /// of candidate indices that achieved the best score on that example.
    #[serde(
        serialize_with = "serde_map_set_as_vec::serialize",
        deserialize_with = "serde_map_set_as_vec::deserialize"
    )]
    pub per_val_instance_best_candidates: HashMap<Id, HashSet<ProgramIdx>>,

    /// Number of per-example metric evaluations consumed to discover each candidate.
    pub discovery_eval_counts: Vec<usize>,

    // ---- Optional multi-objective data -----------------------------------
    /// Per-candidate per-objective aggregate scores.
    /// `None` when no objective scores are tracked.
    pub val_aggregate_subscores: Option<Vec<HashMap<String, f64>>>,

    /// Per-objective Pareto frontier.
    /// `None` when no objective scores are tracked.
    pub per_objective_best_candidates: Option<HashMap<String, HashSet<ProgramIdx>>>,

    /// Per-objective best aggregate score achieved across all candidates.
    pub objective_pareto_front: Option<HashMap<String, f64>>,

    // ---- Run metadata ----------------------------------------------------
    /// Total number of per-example metric evaluations consumed by the full run.
    pub total_metric_calls: Option<usize>,

    /// Number of full validation-set evaluations performed.
    pub num_full_val_evals: Option<usize>,

    /// Directory where run artefacts were saved, or `None`.
    pub run_dir: Option<String>,

    /// RNG seed used for the run, or `None`.
    pub seed: Option<u64>,

    /// Internal: when `Some`, `best_candidate()` unwraps the dict and returns
    /// the plain `String` under this key.
    pub str_candidate_key: Option<String>,

    /// Schema version for forward-compatibility detection.
    pub validation_schema_version: u32,
}

impl<Id: DataId> GEPAResult<Id> {
    /// Current serialisation schema version.
    pub const SCHEMA_VERSION: u32 = 2;

    // ------------------------------------------------------------------
    // Convenience properties
    // ------------------------------------------------------------------

    /// Total number of candidates explored during optimisation.
    pub fn num_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Number of validation examples tracked in the instance Pareto front.
    pub fn num_val_instances(&self) -> usize {
        self.per_val_instance_best_candidates.len()
    }

    /// Index of the highest-scoring candidate (by average validation score).
    ///
    /// In case of ties, the candidate with the smaller index (discovered
    /// earlier) wins.
    ///
    /// # Errors
    /// Returns `Err` when there are no candidates.
    ///
    /// # Panics
    /// Cannot panic: the early-return guard ensures the iterator is non-empty.
    pub fn best_idx(&self) -> Result<ProgramIdx> {
        if self.val_aggregate_scores.is_empty() {
            return Err(crate::error::GEPAError::NoCandidates);
        }
        let best = self
            .val_aggregate_scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|(i_a, a), (i_b, b)| {
                a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(i_b.cmp(i_a)) // prefer smaller index on tie
            })
            .map(|(i, _)| i)
            .unwrap();
        Ok(best)
    }

    /// The highest-scoring candidate as a map.
    ///
    /// # Errors
    /// Returns `Err` when there are no candidates.
    pub fn best_candidate(&self) -> Result<&Candidate> {
        let idx = self.best_idx()?;
        Ok(&self.candidates[idx])
    }

    /// The highest-scoring candidate as a plain string.
    ///
    /// When `str_candidate_key` is set and that key is present in the
    /// best candidate, returns `Some(&str)`.  Mirrors the Python behaviour
    /// where `best_candidate` unwraps the dict when `str_candidate_key` is
    /// configured.
    ///
    /// Returns `None` when `str_candidate_key` is not set, the key is absent
    /// from the candidate, or there are no candidates at all.
    pub fn best_candidate_str(&self) -> Option<&str> {
        let key = self.str_candidate_key.as_deref()?;
        let idx = self.best_idx().ok()?;
        self.candidates[idx].get(key).map(String::as_str)
    }

    // ------------------------------------------------------------------
    // Construction from live state
    // ------------------------------------------------------------------

    /// Construct a [`GEPAResult`] snapshot from the engine's live state.
    ///
    /// This is the primary factory used by the engine at the end of
    /// optimisation (or on-demand for intermediate snapshots).
    ///
    /// # Arguments
    /// * `state`             — the live `GEPAState` to snapshot.
    /// * `run_dir`           — directory where artefacts were persisted.
    /// * `seed`              — RNG seed that was used, if any.
    /// * `str_candidate_key` — when set, `best_candidate` unwraps the internal
    ///   dict and returns the plain `String` value stored under this key.
    pub fn from_state(
        state: &GEPAState<Id>,
        run_dir: Option<String>,
        seed: Option<u64>,
        str_candidate_key: Option<String>,
    ) -> Self {
        let objective_scores_list: Vec<HashMap<String, f64>> =
            state.prog_candidate_objective_scores.clone();
        let has_objective_scores = objective_scores_list.iter().any(|m| !m.is_empty());

        let per_objective_best: HashMap<String, HashSet<ProgramIdx>> = state
            .program_at_pareto_front_objectives
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let objective_front: HashMap<String, f64> = state.objective_pareto_front.clone();

        Self {
            candidates: state.program_candidates.clone(),
            parents: state.parent_program_for_candidate.clone(),
            val_aggregate_scores: state.program_full_scores_val_set(),
            val_subscores: state.prog_candidate_val_subscores.clone(),
            per_val_instance_best_candidates: state.program_at_pareto_front_valset.clone(),
            discovery_eval_counts: state.num_metric_calls_by_discovery.clone(),
            val_aggregate_subscores: if has_objective_scores {
                Some(objective_scores_list)
            } else {
                None
            },
            per_objective_best_candidates: if per_objective_best.is_empty() {
                None
            } else {
                Some(per_objective_best)
            },
            objective_pareto_front: if objective_front.is_empty() {
                None
            } else {
                Some(objective_front)
            },
            total_metric_calls: Some(state.total_num_evals),
            num_full_val_evals: Some(state.num_full_ds_evals),
            run_dir,
            seed,
            str_candidate_key,
            validation_schema_version: Self::SCHEMA_VERSION,
        }
    }

    // ------------------------------------------------------------------
    // Serialisation
    // ------------------------------------------------------------------

    /// Serialise to a JSON string.
    ///
    /// # Errors
    /// Returns `Err` if serialisation fails.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialise from a JSON string produced by `to_json()`.
    ///
    /// # Errors
    /// Returns `Err` if the JSON is malformed or the schema version is newer
    /// than what this code supports.
    pub fn from_json(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let version = u32::try_from(
            v.get("validation_schema_version")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(u32::MAX);
        if version > Self::SCHEMA_VERSION {
            return Err(crate::error::GEPAError::Config(format!(
                "Unsupported GEPAResult schema version {version}; \
                 maximum supported is {}",
                Self::SCHEMA_VERSION
            )));
        }
        Ok(serde_json::from_value(v)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::{FrontierType, ValsetEvaluation};

    fn make_state() -> GEPAState<usize> {
        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "test".into());

        let outputs = (0..3).map(|i| serde_json::json!(i)).collect::<Vec<_>>();
        let eval = ValsetEvaluation::from_vecs(vec![0, 1, 2], outputs, vec![0.3, 0.6, 0.9], None);
        GEPAState::new(seed, eval, FrontierType::Instance, None).unwrap()
    }

    #[test]
    fn from_state_produces_valid_result() {
        let state = make_state();
        let result = GEPAResult::from_state(&state, None, None, None);

        assert_eq!(result.num_candidates(), 1);
        assert_eq!(result.num_val_instances(), 3);
        assert!(result.best_idx().is_ok());
    }

    #[test]
    fn best_idx_with_multiple_candidates() {
        let mut state = make_state();

        // Add a better program (score ~0.95 average).
        let mut new_prog = Candidate::new();
        new_prog.insert("instructions".into(), "improved".into());
        let new_eval = ValsetEvaluation::from_vecs(
            vec![0, 1, 2],
            vec![
                serde_json::json!("a"),
                serde_json::json!("b"),
                serde_json::json!("c"),
            ],
            vec![0.95, 0.95, 0.95],
            None,
        );
        state
            .update_state_with_new_program(vec![0], new_prog, new_eval, 3)
            .unwrap();

        let result = GEPAResult::from_state(&state, None, Some(42), None);
        assert_eq!(
            result.best_idx().unwrap(),
            1,
            "improved candidate should be best"
        );
        assert_eq!(result.seed, Some(42));
    }

    #[test]
    fn json_round_trip() {
        let state = make_state();
        let result = GEPAResult::from_state(&state, Some("/tmp/run".into()), None, None);

        let json = result.to_json().expect("serialisation should succeed");
        let restored: GEPAResult<usize> =
            GEPAResult::from_json(&json).expect("deserialisation should succeed");

        assert_eq!(restored.num_candidates(), result.num_candidates());
        assert_eq!(restored.val_aggregate_scores, result.val_aggregate_scores);
        assert_eq!(restored.run_dir, Some("/tmp/run".into()));
    }

    #[test]
    fn from_json_rejects_future_schema_version() {
        let json = serde_json::json!({
            "validation_schema_version": 9999,
            "candidates": [],
            "parents": [],
            "val_aggregate_scores": [],
            "val_subscores": [],
            "per_val_instance_best_candidates": {},
            "discovery_eval_counts": []
        })
        .to_string();

        let result = GEPAResult::<usize>::from_json(&json);
        assert!(result.is_err(), "should reject future schema version");
    }

    // -------------------------------------------------------------------------
    // Gap 59 — best_idx returns the smaller index when scores are tied
    // -------------------------------------------------------------------------

    #[test]
    fn test_best_idx_tie_prefers_smaller_index() {
        // Build a state with two candidates whose average scores are identical.
        let mut state = make_state(); // seed score is avg(0.3, 0.6, 0.9) = 0.6

        let tied_eval = ValsetEvaluation::from_vecs(
            vec![0, 1, 2],
            vec![
                serde_json::json!("a"),
                serde_json::json!("b"),
                serde_json::json!("c"),
            ],
            vec![0.3, 0.6, 0.9], // same average 0.6
            None,
        );
        let mut tied_candidate = Candidate::new();
        tied_candidate.insert("instructions".into(), "tied".into());
        state
            .update_state_with_new_program(vec![0], tied_candidate, tied_eval, 3)
            .unwrap();

        let result = GEPAResult::from_state(&state, None, None, None);
        assert_eq!(result.num_candidates(), 2);

        // Both candidates have the same average score; the smaller index wins.
        assert_eq!(
            result.best_idx().expect("should have candidates"),
            0,
            "smaller index should win on tie"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 60a — best_candidate_str returns the string when key is present
    // -------------------------------------------------------------------------

    #[test]
    fn test_best_candidate_str_returns_value() {
        let state = make_state();
        let result = GEPAResult::from_state(&state, None, None, Some("instructions".into()));

        let s = result.best_candidate_str();
        assert!(
            s.is_some(),
            "best_candidate_str should return Some when key is present"
        );
        assert_eq!(s.unwrap(), "test");
    }

    // -------------------------------------------------------------------------
    // Gap 60b — best_candidate_str returns None when key is absent
    // -------------------------------------------------------------------------

    #[test]
    fn test_best_candidate_str_missing_key_returns_none() {
        let state = make_state();
        // Set str_candidate_key to a key that does not exist in the candidate.
        let result = GEPAResult::from_state(&state, None, None, Some("nonexistent_key".into()));

        assert!(
            result.best_candidate_str().is_none(),
            "best_candidate_str should return None when the key is absent from the candidate"
        );
    }
}
