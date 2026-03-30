/// `GEPAState` — the persistent, mutable core of an optimization run.
///
/// Tracks all explored candidates, their per-example and per-objective
/// scores, Pareto frontiers (one or more strategies), evaluation budget,
/// and an optional evaluation cache.  State can be serialised to JSON via
/// [`GEPAState::to_json`] / [`GEPAState::from_json`] to enable safe run
/// resumption after interruption.
///
/// Mirrors the Python `gepa.core.state` module.
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::adapter::Candidate;
use crate::core::data_loader::DataId;
use crate::error::{GEPAError, Result};

// ---------------------------------------------------------------------------
// Serde helpers for HashMap<Id, V> where Id is not necessarily a string.
// JSON object keys must be strings; we serialise these maps as Vec<(Id, V)>.
// ---------------------------------------------------------------------------

mod serde_map_as_vec {
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;
    use std::hash::Hash;

    pub fn serialize<K, V, S>(map: &HashMap<K, V>, ser: S) -> Result<S::Ok, S::Error>
    where
        K: Serialize + Eq + Hash,
        V: Serialize,
        S: Serializer,
    {
        let pairs: Vec<(&K, &V)> = map.iter().collect();
        pairs.serialize(ser)
    }

    pub fn deserialize<'de, K, V, D>(de: D) -> Result<HashMap<K, V>, D::Error>
    where
        K: DeserializeOwned + Eq + Hash,
        V: DeserializeOwned,
        D: Deserializer<'de>,
    {
        let pairs: Vec<(K, V)> = Vec::deserialize(de)?;
        Ok(pairs.into_iter().collect())
    }
}

use crate::core::serde_helpers::{serde_map_set_as_vec, serde_vec_of_maps};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Index into the `program_candidates` list.
pub type ProgramIdx = usize;

/// Mapping from objective name to aggregated score for a single candidate.
pub type ObjectiveScores = HashMap<String, f64>;

// ---------------------------------------------------------------------------
// FrontierType
// ---------------------------------------------------------------------------

/// Strategy for tracking Pareto frontiers.
///
/// - `Instance`:  per validation example (default — the paper's approach).
/// - `Objective`: per named objective metric.
/// - `Hybrid`:    both per-example and per-objective in a unified mapping.
/// - `Cartesian`: per `(example, objective)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FrontierType {
    #[default]
    Instance,
    Objective,
    Hybrid,
    Cartesian,
}

// ---------------------------------------------------------------------------
// FrontierKey — a type-safe union of all possible frontier keys
// ---------------------------------------------------------------------------

/// Union key type for frontier mappings — varies with `FrontierType`.
///
/// Serialised as a tagged JSON enum so state snapshots remain readable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
#[serde(bound(deserialize = "Id: DataId"))]
pub enum FrontierKey<Id: DataId> {
    /// Keyed by validation-example ID (used in `Instance` and `Hybrid`).
    Instance { val_id: Id },
    /// Keyed by objective name (used in `Objective` and `Hybrid`).
    Objective { name: String },
    /// Keyed by `(val_id, objective_name)` (used in `Cartesian`).
    Cartesian { val_id: Id, objective: String },
}

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

/// Compute a deterministic SHA-256 hash of a candidate dictionary.
///
/// Keys are sorted before serialisation so the hash is invariant to insertion
/// order, matching the Python reference implementation.
pub fn candidate_hash(candidate: &Candidate) -> String {
    let mut pairs: Vec<(&String, &String)> = candidate.iter().collect();
    pairs.sort_by_key(|(k, _)| k.as_str());
    let serialised = serde_json::to_string(&pairs).unwrap_or_default();
    let digest = Sha256::digest(serialised.as_bytes());
    format!("{digest:x}")
}

// ---------------------------------------------------------------------------
// CachedEvaluation
// ---------------------------------------------------------------------------

/// Cached evaluation result for a single `(candidate, example)` pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEvaluation {
    /// Raw program output serialised as JSON.
    pub output: serde_json::Value,
    /// Per-example numeric score.
    pub score: f64,
    /// Optional named-objective breakdown.
    pub objective_scores: Option<ObjectiveScores>,
}

// ---------------------------------------------------------------------------
// EvaluationCache
// ---------------------------------------------------------------------------

/// Evaluation result cache keyed by `"{candidate_hash}:{data_id_json}"`.
///
/// Avoids re-evaluating `(candidate, example)` pairs that have already been
/// scored in previous iterations.  The cache is optional; when caching is
/// disabled, pass `None` in [`GEPAState`].
///
/// The inner map uses a plain `String` composite key so the cache is
/// JSON-serialisable regardless of the `Id` type.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationCache {
    /// Inner map: `"sha256_hex:json_encoded_id"` → cached result.
    inner: HashMap<String, CachedEvaluation>,
}

impl EvaluationCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    fn make_key<Id: Serialize>(candidate: &Candidate, id: &Id) -> String {
        let h = candidate_hash(candidate);
        let eid = serde_json::to_string(id).unwrap_or_default();
        format!("{h}:{eid}")
    }

    /// Retrieve a cached result, or `None` if not present.
    pub fn get<Id: DataId>(
        &self,
        candidate: &Candidate,
        example_id: &Id,
    ) -> Option<&CachedEvaluation> {
        self.inner.get(&Self::make_key(candidate, example_id))
    }

    /// Store a single evaluation result.
    pub fn put<Id: DataId>(
        &mut self,
        candidate: &Candidate,
        example_id: &Id,
        output: serde_json::Value,
        score: f64,
        objective_scores: Option<ObjectiveScores>,
    ) {
        self.inner.insert(
            Self::make_key(candidate, example_id),
            CachedEvaluation {
                output,
                score,
                objective_scores,
            },
        );
    }

    /// Batch lookup — returns `(cached_results, uncached_ids)`.
    pub fn get_batch<Id: DataId>(
        &self,
        candidate: &Candidate,
        example_ids: &[Id],
    ) -> (HashMap<Id, &CachedEvaluation>, Vec<Id>) {
        let mut cached = HashMap::new();
        let mut uncached = Vec::new();

        for id in example_ids {
            let key = Self::make_key(candidate, id);
            if let Some(entry) = self.inner.get(&key) {
                cached.insert(id.clone(), entry);
            } else {
                uncached.push(id.clone());
            }
        }
        (cached, uncached)
    }

    /// Store evaluation results for a batch of examples.
    pub fn put_batch<Id: DataId>(
        &mut self,
        candidate: &Candidate,
        example_ids: &[Id],
        outputs: Vec<serde_json::Value>,
        scores: Vec<f64>,
        objective_scores: Option<Vec<ObjectiveScores>>,
    ) {
        for (i, id) in example_ids.iter().enumerate() {
            self.inner.insert(
                Self::make_key(candidate, id),
                CachedEvaluation {
                    output: outputs[i].clone(),
                    score: scores[i],
                    objective_scores: objective_scores.as_ref().and_then(|v| v.get(i).cloned()),
                },
            );
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ValsetEvaluation
// ---------------------------------------------------------------------------

/// Container for evaluation results on a validation-set batch.
#[derive(Debug, Clone)]
pub struct ValsetEvaluation<Id: DataId> {
    /// Raw per-example outputs, keyed by validation ID.
    pub outputs_by_val_id: HashMap<Id, serde_json::Value>,
    /// Per-example scores, keyed by validation ID.
    pub scores_by_val_id: HashMap<Id, f64>,
    /// Optional per-example objective breakdowns.
    pub objective_scores_by_val_id: Option<HashMap<Id, ObjectiveScores>>,
}

impl<Id: DataId> ValsetEvaluation<Id> {
    /// Construct from aligned vecs of IDs, outputs, and scores.
    pub fn from_vecs(
        ids: Vec<Id>,
        outputs: Vec<serde_json::Value>,
        scores: Vec<f64>,
        objective_scores: Option<Vec<ObjectiveScores>>,
    ) -> Self {
        let n = ids.len();
        let outputs_by_val_id = ids.iter().cloned().zip(outputs).collect::<HashMap<_, _>>();
        let scores_by_val_id = ids.iter().cloned().zip(scores).collect::<HashMap<_, _>>();
        let objective_scores_by_val_id =
            objective_scores.map(|objs| ids.iter().cloned().zip(objs).collect::<HashMap<_, _>>());
        debug_assert_eq!(outputs_by_val_id.len(), n);
        Self {
            outputs_by_val_id,
            scores_by_val_id,
            objective_scores_by_val_id,
        }
    }
}

// ---------------------------------------------------------------------------
// GEPAState
// ---------------------------------------------------------------------------

/// Internal persistent state of a GEPA optimization run.
///
/// Tracks all explored candidates, their per-example and per-objective scores,
/// Pareto frontiers, evaluation budget, and an optional evaluation cache.
///
/// ### Key collections
///
/// - `program_candidates`: ordered list of all explored candidates.
/// - `prog_candidate_val_subscores`: per-candidate `{val_id: score}`.
/// - `pareto_front_valset`: per-example best score achieved by any candidate.
/// - `program_at_pareto_front_valset`: per-example set of candidate indices
///   that achieved the best score.
/// - `list_of_named_predictors`: component names from the seed candidate.
/// - `named_predictor_id_to_update_next`: per-candidate round-robin counter.
/// - `total_num_evals`: budget tracking counter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "Id: DataId"))]
pub struct GEPAState<Id: DataId> {
    // ---- Schema version --------------------------------------------------
    pub validation_schema_version: u32,

    // ---- Candidate registry ---------------------------------------------
    /// All explored candidates in discovery order.
    /// Uses `IndexMap` to preserve insertion order, matching the Python reference.
    pub program_candidates: Vec<Candidate>,

    /// `parents[i]` is the list of parent indices for candidate `i`.
    /// `[None]` for the seed candidate.
    pub parent_program_for_candidate: Vec<Vec<Option<ProgramIdx>>>,

    /// Per-candidate `{val_id: score}` for each evaluated validation example.
    #[serde(
        serialize_with = "serde_vec_of_maps::serialize",
        deserialize_with = "serde_vec_of_maps::deserialize"
    )]
    pub prog_candidate_val_subscores: Vec<HashMap<Id, f64>>,

    /// Per-candidate aggregated objective scores (averaged over val set).
    pub prog_candidate_objective_scores: Vec<ObjectiveScores>,

    /// Number of `evaluate()` calls consumed to discover each candidate.
    pub num_metric_calls_by_discovery: Vec<usize>,

    // ---- Instance-level Pareto front ------------------------------------
    /// Per-example best score achieved across all candidates.
    #[serde(
        serialize_with = "serde_map_as_vec::serialize",
        deserialize_with = "serde_map_as_vec::deserialize"
    )]
    pub pareto_front_valset: HashMap<Id, f64>,

    /// Per-example set of candidate indices that achieved the best score.
    #[serde(
        serialize_with = "serde_map_set_as_vec::serialize",
        deserialize_with = "serde_map_set_as_vec::deserialize"
    )]
    pub program_at_pareto_front_valset: HashMap<Id, HashSet<ProgramIdx>>,

    // ---- Objective-level Pareto front -----------------------------------
    /// Per-objective best aggregate score achieved across all candidates.
    pub objective_pareto_front: ObjectiveScores,

    /// Per-objective set of candidate indices at the Pareto front.
    pub program_at_pareto_front_objectives: HashMap<String, HashSet<ProgramIdx>>,

    // ---- Cartesian (val_id × objective) Pareto front --------------------
    /// Per-`(val_id, objective)` best score.
    /// Skipped during JSON serialisation — tuple keys are not JSON-representable.
    /// Use `pareto_front_cartesian_export()` to obtain a serialisable form.
    #[serde(skip)]
    pub pareto_front_cartesian: HashMap<(Id, String), f64>,

    /// Per-`(val_id, objective)` set of candidate indices at the front.
    #[serde(skip)]
    pub program_at_pareto_front_cartesian: HashMap<(Id, String), HashSet<ProgramIdx>>,

    // ---- Component bookkeeping ------------------------------------------
    /// Component names derived from the seed candidate keys.
    pub list_of_named_predictors: Vec<String>,

    /// Per-candidate round-robin counter: which component to update next.
    pub named_predictor_id_to_update_next: Vec<usize>,

    // ---- Run metadata ---------------------------------------------------
    /// Active frontier strategy.
    pub frontier_type: FrontierType,

    /// Iteration counter.  Starts at `usize::MAX` (sentinel for "before first
    /// iteration"); the engine increments it to `0` before iteration 0.
    pub i: usize,

    /// Total number of full validation-set evaluations performed.
    pub num_full_ds_evals: usize,

    /// Total number of individual adapter `evaluate()` calls consumed.
    pub total_num_evals: usize,

    /// Optional evaluation cache (disabled when `None`).
    #[serde(skip)]
    pub evaluation_cache: Option<EvaluationCache>,

    // ---- Program trace --------------------------------------------------
    /// Full trace of every optimisation iteration.
    /// Each entry is a JSON object with iteration metadata.
    #[serde(default)]
    pub full_program_trace: Vec<serde_json::Value>,

    // ---- Best-output tracking -------------------------------------------
    /// Per-val-id best `(program_idx, output)` seen so far.
    /// `None` when `track_best_outputs` is disabled.
    #[serde(skip)]
    pub best_outputs_valset: Option<HashMap<Id, (ProgramIdx, serde_json::Value)>>,

    #[serde(skip)]
    _phantom: PhantomData<Id>,
}

/// Sentinel value for "before first iteration" (i.e. Python's `i = -1`).
pub const BEFORE_FIRST_ITERATION: usize = usize::MAX;

impl<Id: DataId> GEPAState<Id> {
    /// Current schema version — bump on breaking serialisation changes.
    pub const SCHEMA_VERSION: u32 = 4;

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Initialise state from a seed candidate and its baseline evaluation.
    ///
    /// # Errors
    /// Returns `Err` when `frontier_type` requires objective scores but none
    /// are provided in `base_evaluation`.
    pub fn new(
        seed_candidate: Candidate,
        base_evaluation: ValsetEvaluation<Id>,
        frontier_type: FrontierType,
        evaluation_cache: Option<EvaluationCache>,
    ) -> Result<Self> {
        Self::new_with_options(
            seed_candidate,
            base_evaluation,
            frontier_type,
            evaluation_cache,
            false,
        )
    }

    /// Initialise state with full options including best-output tracking.
    ///
    /// # Errors
    /// Returns `Err` when `frontier_type` requires objective scores but none
    /// are provided in `base_evaluation`.
    pub fn new_with_options(
        seed_candidate: Candidate,
        base_evaluation: ValsetEvaluation<Id>,
        frontier_type: FrontierType,
        evaluation_cache: Option<EvaluationCache>,
        track_best_outputs: bool,
    ) -> Result<Self> {
        if matches!(
            frontier_type,
            FrontierType::Objective | FrontierType::Hybrid | FrontierType::Cartesian
        ) && base_evaluation.objective_scores_by_val_id.is_none()
        {
            return Err(GEPAError::Config(format!(
                "frontier_type={frontier_type:?} requires objective_scores to be provided \
                 by the evaluator, but none were found. \
                 Use an evaluator that returns objective_scores or use FrontierType::Instance."
            )));
        }

        let base_objectives =
            Self::aggregate_objective_scores(base_evaluation.objective_scores_by_val_id.as_ref());

        // Instance-level Pareto front seeded from the seed candidate.
        let pareto_front_valset: HashMap<Id, f64> = base_evaluation.scores_by_val_id.clone();
        let program_at_pareto_front_valset: HashMap<Id, HashSet<ProgramIdx>> = base_evaluation
            .scores_by_val_id
            .keys()
            .map(|id| {
                (id.clone(), {
                    let mut s = HashSet::new();
                    s.insert(0usize);
                    s
                })
            })
            .collect();

        // Objective-level Pareto front.
        let objective_pareto_front = base_objectives.clone();
        let program_at_pareto_front_objectives: HashMap<String, HashSet<ProgramIdx>> =
            base_objectives
                .keys()
                .map(|k| {
                    let mut s = HashSet::new();
                    s.insert(0usize);
                    (k.clone(), s)
                })
                .collect();

        // Cartesian front.
        let (pareto_front_cartesian, program_at_pareto_front_cartesian) = if frontier_type
            == FrontierType::Cartesian
        {
            let obj_by_id = base_evaluation.objective_scores_by_val_id.as_ref()
                    .ok_or_else(|| GEPAError::Config(
                        "Cartesian frontier requires objective_scores_by_val_id in the base evaluation".into()
                    ))?;
            let mut pf: HashMap<(Id, String), f64> = HashMap::new();
            let mut pa: HashMap<(Id, String), HashSet<ProgramIdx>> = HashMap::new();
            for (val_id, obj_scores) in obj_by_id {
                for (obj_name, &score) in obj_scores {
                    pf.insert((val_id.clone(), obj_name.clone()), score);
                    let mut s = HashSet::new();
                    s.insert(0usize);
                    pa.insert((val_id.clone(), obj_name.clone()), s);
                }
            }
            (pf, pa)
        } else {
            (HashMap::new(), HashMap::new())
        };

        // Preserve insertion order from the seed candidate keys.
        // This matches the Python reference: `list(seed_candidate.keys())`
        let list_of_named_predictors: Vec<String> = seed_candidate.keys().cloned().collect();

        let best_outputs_valset = if track_best_outputs {
            Some(
                base_evaluation
                    .outputs_by_val_id
                    .iter()
                    .map(|(id, out)| (id.clone(), (0usize, out.clone())))
                    .collect::<HashMap<Id, (ProgramIdx, serde_json::Value)>>(),
            )
        } else {
            None
        };

        Ok(Self {
            validation_schema_version: Self::SCHEMA_VERSION,
            program_candidates: vec![seed_candidate],
            parent_program_for_candidate: vec![vec![None]],
            prog_candidate_val_subscores: vec![base_evaluation.scores_by_val_id],
            prog_candidate_objective_scores: vec![base_objectives],
            num_metric_calls_by_discovery: vec![0],
            pareto_front_valset,
            program_at_pareto_front_valset,
            objective_pareto_front,
            program_at_pareto_front_objectives,
            pareto_front_cartesian,
            program_at_pareto_front_cartesian,
            list_of_named_predictors,
            named_predictor_id_to_update_next: vec![0],
            frontier_type,
            i: BEFORE_FIRST_ITERATION,
            num_full_ds_evals: 0,
            total_num_evals: 0,
            evaluation_cache,
            full_program_trace: Vec::new(),
            best_outputs_valset,
            _phantom: PhantomData,
        })
    }

    // ------------------------------------------------------------------
    // Aggregate objective scores helper
    // ------------------------------------------------------------------

    /// Average per-objective scores across all validation examples.
    ///
    /// Returns an empty map when `objective_scores_by_val_id` is `None`.
    pub fn aggregate_objective_scores(
        objective_scores_by_val_id: Option<&HashMap<Id, ObjectiveScores>>,
    ) -> ObjectiveScores {
        let Some(by_id) = objective_scores_by_val_id else {
            return ObjectiveScores::new();
        };
        if by_id.is_empty() {
            return ObjectiveScores::new();
        }

        let mut totals: HashMap<String, f64> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for obj_scores in by_id.values() {
            for (name, &score) in obj_scores {
                *totals.entry(name.clone()).or_insert(0.0) += score;
                *counts.entry(name.clone()).or_insert(0) += 1;
            }
        }
        totals
            .into_iter()
            .map(|(k, total)| (k.clone(), total / counts[&k] as f64))
            .collect()
    }

    // ------------------------------------------------------------------
    // Budget tracking
    // ------------------------------------------------------------------

    /// Increment `total_num_evals` by `count`.
    pub fn increment_evals(&mut self, count: usize) {
        self.total_num_evals += count;
    }

    // ------------------------------------------------------------------
    // Per-program score helpers
    // ------------------------------------------------------------------

    /// Return `(mean_score, num_samples)` for a candidate.
    ///
    /// Returns `(f64::NEG_INFINITY, 0)` when no validation examples have been
    /// scored for this candidate.
    pub fn get_program_average_val_subset(&self, program_idx: ProgramIdx) -> (f64, usize) {
        let scores = &self.prog_candidate_val_subscores[program_idx];
        if scores.is_empty() {
            return (f64::NEG_INFINITY, 0);
        }
        let n = scores.len();
        let total: f64 = scores.values().sum();
        (total / n as f64, n)
    }

    /// Per-candidate average validation score (`f64::NEG_INFINITY` for unseen).
    pub fn program_full_scores_val_set(&self) -> Vec<f64> {
        (0..self.program_candidates.len())
            .map(|i| self.get_program_average_val_subset(i).0)
            .collect()
    }

    // ------------------------------------------------------------------
    // Core state mutation
    // ------------------------------------------------------------------

    /// Register a newly accepted candidate and update all Pareto frontiers.
    ///
    /// Called by the engine whenever a mutation or merge produces a candidate
    /// that passes the acceptance test.
    ///
    /// Returns the integer index assigned to the new candidate.
    ///
    /// # Errors
    /// Returns `Err` when `frontier_type` requires objective scores but none
    /// are present in `valset_evaluation`.
    pub fn update_state_with_new_program(
        &mut self,
        parent_program_idx: Vec<ProgramIdx>,
        new_program: Candidate,
        valset_evaluation: ValsetEvaluation<Id>,
        num_metric_calls_by_discovery_of_new_program: usize,
    ) -> Result<ProgramIdx> {
        let new_program_idx = self.program_candidates.len();

        // Advance the round-robin component counter.
        let max_predictor_id = parent_program_idx
            .iter()
            .map(|&p| self.named_predictor_id_to_update_next[p])
            .max()
            .unwrap_or(0);

        self.program_candidates.push(new_program);
        self.num_metric_calls_by_discovery
            .push(num_metric_calls_by_discovery_of_new_program);
        self.named_predictor_id_to_update_next
            .push(max_predictor_id);
        self.parent_program_for_candidate
            .push(parent_program_idx.iter().map(|&p| Some(p)).collect());

        let valset_scores = valset_evaluation.scores_by_val_id.clone();
        let objective_scores =
            Self::aggregate_objective_scores(valset_evaluation.objective_scores_by_val_id.as_ref());

        self.prog_candidate_val_subscores
            .push(valset_scores.clone());
        self.prog_candidate_objective_scores
            .push(objective_scores.clone());

        // Update instance-level front.
        for (val_id, &score) in &valset_scores {
            self.update_pareto_front_for_val_id(val_id, score, new_program_idx);
        }

        // Update best-output tracking (Gap 5).
        if let Some(ref mut best_map) = self.best_outputs_valset {
            for (val_id, &score) in &valset_scores {
                let is_better = best_map.get(val_id).is_none_or(|(_, _)| {
                    // Compare against the current Pareto front score.
                    let prev = self
                        .pareto_front_valset
                        .get(val_id)
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
                    score > prev
                });
                if is_better {
                    let output = valset_evaluation
                        .outputs_by_val_id
                        .get(val_id)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    best_map.insert(val_id.clone(), (new_program_idx, output));
                }
            }
        }

        // Update objective-level front.
        self.update_objective_pareto_front(&objective_scores, new_program_idx);

        // Update cartesian front.
        if self.frontier_type == FrontierType::Cartesian {
            let obj_by_id = valset_evaluation
                .objective_scores_by_val_id
                .as_ref()
                .ok_or_else(|| {
                    GEPAError::Config(
                        "frontier_type=Cartesian requires objective_scores in valset_evaluation"
                            .into(),
                    )
                })?;
            for (val_id, obj_scores) in obj_by_id {
                for (objective, &obj_score) in obj_scores {
                    self.update_pareto_front_for_cartesian(
                        val_id,
                        objective,
                        obj_score,
                        new_program_idx,
                    );
                }
            }
        }

        Ok(new_program_idx)
    }

    // ------------------------------------------------------------------
    // Internal Pareto-front update helpers
    // ------------------------------------------------------------------

    fn update_pareto_front_for_val_id(&mut self, val_id: &Id, score: f64, program_idx: ProgramIdx) {
        let prev = *self
            .pareto_front_valset
            .get(val_id)
            .unwrap_or(&f64::NEG_INFINITY);
        if score > prev {
            self.pareto_front_valset.insert(val_id.clone(), score);
            let mut s = HashSet::new();
            s.insert(program_idx);
            self.program_at_pareto_front_valset
                .insert(val_id.clone(), s);
        } else if score == prev {
            self.program_at_pareto_front_valset
                .entry(val_id.clone())
                .or_default()
                .insert(program_idx);
        }
    }

    fn update_objective_pareto_front(
        &mut self,
        objective_scores: &ObjectiveScores,
        program_idx: ProgramIdx,
    ) {
        for (objective, &score) in objective_scores {
            let prev = *self
                .objective_pareto_front
                .get(objective)
                .unwrap_or(&f64::NEG_INFINITY);
            if score > prev {
                self.objective_pareto_front.insert(objective.clone(), score);
                let mut s = HashSet::new();
                s.insert(program_idx);
                self.program_at_pareto_front_objectives
                    .insert(objective.clone(), s);
            } else if score == prev {
                self.program_at_pareto_front_objectives
                    .entry(objective.clone())
                    .or_default()
                    .insert(program_idx);
            }
        }
    }

    fn update_pareto_front_for_cartesian(
        &mut self,
        val_id: &Id,
        objective: &str,
        objective_score: f64,
        program_idx: ProgramIdx,
    ) {
        let key = (val_id.clone(), objective.to_string());
        let prev = *self
            .pareto_front_cartesian
            .get(&key)
            .unwrap_or(&f64::NEG_INFINITY);
        if objective_score > prev {
            self.pareto_front_cartesian
                .insert(key.clone(), objective_score);
            let mut s = HashSet::new();
            s.insert(program_idx);
            self.program_at_pareto_front_cartesian.insert(key, s);
        } else if objective_score == prev {
            self.program_at_pareto_front_cartesian
                .entry(key)
                .or_default()
                .insert(program_idx);
        }
    }

    // ------------------------------------------------------------------
    // Pareto-front retrieval
    // ------------------------------------------------------------------

    /// Return the active frontier mapping (key → candidate-index set).
    ///
    /// The mapping strategy is determined by `self.frontier_type`.
    pub fn get_pareto_front_mapping(&self) -> HashMap<FrontierKey<Id>, HashSet<ProgramIdx>> {
        match self.frontier_type {
            FrontierType::Instance => self
                .program_at_pareto_front_valset
                .iter()
                .map(|(val_id, front)| {
                    (
                        FrontierKey::Instance {
                            val_id: val_id.clone(),
                        },
                        front.clone(),
                    )
                })
                .collect(),

            FrontierType::Objective => self
                .program_at_pareto_front_objectives
                .iter()
                .map(|(name, front)| (FrontierKey::Objective { name: name.clone() }, front.clone()))
                .collect(),

            FrontierType::Hybrid => {
                let mut combined: HashMap<FrontierKey<Id>, HashSet<ProgramIdx>> = self
                    .program_at_pareto_front_valset
                    .iter()
                    .map(|(val_id, front)| {
                        (
                            FrontierKey::Instance {
                                val_id: val_id.clone(),
                            },
                            front.clone(),
                        )
                    })
                    .collect();
                for (name, front) in &self.program_at_pareto_front_objectives {
                    combined.insert(FrontierKey::Objective { name: name.clone() }, front.clone());
                }
                combined
            }

            FrontierType::Cartesian => self
                .program_at_pareto_front_cartesian
                .iter()
                .map(|((val_id, objective), front)| {
                    (
                        FrontierKey::Cartesian {
                            val_id: val_id.clone(),
                            objective: objective.clone(),
                        },
                        front.clone(),
                    )
                })
                .collect(),
        }
    }

    // ------------------------------------------------------------------
    // Consistency check
    // ------------------------------------------------------------------

    /// Assert all internal lists are in sync.
    ///
    /// # Errors
    /// Returns `Err` when an invariant is violated.
    pub fn is_consistent(&self) -> Result<()> {
        let n = self.program_candidates.len();

        macro_rules! check_len {
            ($field:expr, $name:literal) => {
                if $field.len() != n {
                    return Err(GEPAError::Config(format!(
                        "GEPAState invariant: {} has len {} but program_candidates has len {}",
                        $name,
                        $field.len(),
                        n
                    )));
                }
            };
        }

        check_len!(
            self.parent_program_for_candidate,
            "parent_program_for_candidate"
        );
        check_len!(
            self.named_predictor_id_to_update_next,
            "named_predictor_id_to_update_next"
        );
        check_len!(
            self.prog_candidate_val_subscores,
            "prog_candidate_val_subscores"
        );
        check_len!(
            self.prog_candidate_objective_scores,
            "prog_candidate_objective_scores"
        );
        check_len!(
            self.num_metric_calls_by_discovery,
            "num_metric_calls_by_discovery"
        );

        if self.pareto_front_valset.len() != self.program_at_pareto_front_valset.len() {
            return Err(GEPAError::Config(
                "pareto_front_valset and program_at_pareto_front_valset have different sizes"
                    .into(),
            ));
        }

        for (val_id, front) in &self.program_at_pareto_front_valset {
            if !self.pareto_front_valset.contains_key(val_id) {
                return Err(GEPAError::Config(format!(
                    "val_id {val_id:?} in program_at_pareto_front_valset but not in pareto_front_valset"
                )));
            }
            for &prog_idx in front {
                if prog_idx >= n {
                    return Err(GEPAError::Config(format!(
                        "Program index {prog_idx} in valset Pareto front exceeds \
                         number of program candidates ({n})"
                    )));
                }
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Serialisation
    // ------------------------------------------------------------------

    /// Serialise the state to a JSON string.
    ///
    /// # Errors
    /// Returns `Err` if serialisation fails.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialise state from a JSON string produced by `to_json()`.
    ///
    /// # Errors
    /// Returns `Err` if the JSON is malformed, incompatible, or has a newer
    /// schema version than this code supports.
    pub fn from_json(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let version = u32::try_from(
            v.get("validation_schema_version")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(u32::MAX);
        if version > Self::SCHEMA_VERSION {
            return Err(GEPAError::Config(format!(
                "Unsupported GEPAState schema version {version}; \
                 maximum supported is {}",
                Self::SCHEMA_VERSION
            )));
        }
        Ok(serde_json::from_value(v)?)
    }

    // ------------------------------------------------------------------
    // Gap 2: Cached evaluation helper
    // ------------------------------------------------------------------

    /// Evaluate `candidate` on `example_ids`, using the cache when available.
    ///
    /// Returns `(ValsetEvaluation, num_actual_evals)` where `num_actual_evals`
    /// is the number of examples that were not served from the cache.
    ///
    /// When no `evaluation_cache` is configured this is a simple pass-through
    /// to the provided `evaluate_fn`.
    ///
    /// # Panics
    /// Will not panic — all unwraps are guarded by `is_some()` checks.
    #[allow(clippy::unnecessary_unwrap)]
    pub fn cached_evaluate_valset<Id2, F>(
        &mut self,
        candidate: &Candidate,
        example_ids: &[Id2],
        evaluate_fn: F,
    ) -> Result<(ValsetEvaluation<Id2>, usize)>
    where
        Id2: DataId + std::hash::Hash + Eq,
        F: FnOnce(
            &[Id2],
            &Candidate,
        ) -> Result<(
            HashMap<Id2, serde_json::Value>,
            HashMap<Id2, f64>,
            Option<HashMap<Id2, ObjectiveScores>>,
        )>,
    {
        if self.evaluation_cache.is_some() {
            // ── Phase 1: read from cache; clone entries so no borrow is retained ─
            #[allow(clippy::unwrap_used)]
            let (cached_owned, uncached_ids): (
                HashMap<Id2, CachedEvaluation>,
                Vec<Id2>,
            ) = {
                // SAFETY: guarded by is_some() check above; we need split borrows
                let cache = self.evaluation_cache.as_ref().unwrap();
                let (refs_map, uncached) = cache.get_batch(candidate, example_ids);
                // Clone values to release the borrow on `cache`.
                let owned = refs_map
                    .into_iter()
                    .map(|(id, entry)| (id.clone(), entry.clone()))
                    .collect();
                (owned, uncached)
            };

            // ── Phase 2: evaluate uncached examples (no borrow held) ─────────
            let (new_outputs, new_scores, new_obj) = if uncached_ids.is_empty() {
                (HashMap::new(), HashMap::new(), None)
            } else {
                evaluate_fn(&uncached_ids, candidate)?
            };
            let num_actual = uncached_ids.len();

            // ── Phase 3: write new results back into cache ────────────────────
            {
                let new_outputs_vec: Vec<serde_json::Value> = uncached_ids
                    .iter()
                    .map(|id| {
                        new_outputs
                            .get(id)
                            .cloned()
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect();
                let new_scores_vec: Vec<f64> = uncached_ids
                    .iter()
                    .map(|id| new_scores.get(id).copied().unwrap_or(0.0))
                    .collect();
                let new_obj_vec: Option<Vec<ObjectiveScores>> = new_obj.as_ref().map(|obj_map| {
                    uncached_ids
                        .iter()
                        .map(|id| obj_map.get(id).cloned().unwrap_or_default())
                        .collect()
                });
                #[allow(clippy::unwrap_used)]
                // SAFETY: guarded by is_some() check above; split-phase borrows
                let cache = self.evaluation_cache.as_mut().unwrap();
                cache.put_batch(
                    candidate,
                    &uncached_ids,
                    new_outputs_vec,
                    new_scores_vec,
                    new_obj_vec,
                );
            }

            // ── Phase 4: merge cached + new results ───────────────────────────
            let mut outputs_by_val_id: HashMap<Id2, serde_json::Value> = cached_owned
                .iter()
                .map(|(id, e)| (id.clone(), e.output.clone()))
                .collect();
            let mut scores_by_val_id: HashMap<Id2, f64> = cached_owned
                .iter()
                .map(|(id, e)| (id.clone(), e.score))
                .collect();
            let mut objective_scores_by_val_id: Option<HashMap<Id2, ObjectiveScores>> = {
                let has_cached_obj = cached_owned.values().any(|e| e.objective_scores.is_some());
                if has_cached_obj {
                    Some(
                        cached_owned
                            .iter()
                            .filter_map(|(id, e)| {
                                e.objective_scores.as_ref().map(|o| (id.clone(), o.clone()))
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            };

            outputs_by_val_id.extend(new_outputs);
            scores_by_val_id.extend(new_scores);
            if let Some(ref new_obj_map) = new_obj {
                objective_scores_by_val_id
                    .get_or_insert_with(HashMap::new)
                    .extend(new_obj_map.iter().map(|(k, v)| (k.clone(), v.clone())));
            }

            Ok((
                ValsetEvaluation {
                    outputs_by_val_id,
                    scores_by_val_id,
                    objective_scores_by_val_id,
                },
                num_actual,
            ))
        } else {
            // No cache: evaluate everything directly.
            let (outputs_by_val_id, scores_by_val_id, obj) = evaluate_fn(example_ids, candidate)?;
            let num_actual = example_ids.len();
            Ok((
                ValsetEvaluation {
                    outputs_by_val_id,
                    scores_by_val_id,
                    objective_scores_by_val_id: obj,
                },
                num_actual,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seed_evaluation(ids: &[usize], scores: &[f64]) -> ValsetEvaluation<usize> {
        let outputs = ids
            .iter()
            .map(|i| serde_json::json!({"id": i}))
            .collect::<Vec<_>>();
        ValsetEvaluation::from_vecs(ids.to_vec(), outputs, scores.to_vec(), None)
    }

    fn make_seed_candidate() -> Candidate {
        let mut c = Candidate::new();
        c.insert("instructions".into(), "Do the task.".into());
        c
    }

    #[test]
    fn initial_state_is_consistent() {
        let eval = make_seed_evaluation(&[0, 1, 2], &[0.5, 0.6, 0.7]);
        let state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        state.is_consistent().expect("state should be consistent");
        assert_eq!(state.program_candidates.len(), 1);
        assert_eq!(state.i, BEFORE_FIRST_ITERATION);
    }

    #[test]
    fn update_state_with_new_program_grows_all_collections() {
        let eval = make_seed_evaluation(&[0, 1, 2], &[0.4, 0.5, 0.6]);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        // Add a better program.
        let new_eval = ValsetEvaluation::from_vecs(
            vec![0, 1, 2],
            vec![
                serde_json::json!("out0"),
                serde_json::json!("out1"),
                serde_json::json!("out2"),
            ],
            vec![0.8, 0.9, 1.0],
            None,
        );
        let new_idx = state
            .update_state_with_new_program(vec![0], make_seed_candidate(), new_eval, 3)
            .expect("update should succeed");

        assert_eq!(new_idx, 1);
        assert_eq!(state.program_candidates.len(), 2);
        state
            .is_consistent()
            .expect("state should still be consistent");

        // Pareto front should now point to the new (better) program.
        let scores = state.program_full_scores_val_set();
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn candidate_hash_is_order_invariant() {
        let mut a = Candidate::new();
        a.insert("z".into(), "last".into());
        a.insert("a".into(), "first".into());

        let mut b = Candidate::new();
        b.insert("a".into(), "first".into());
        b.insert("z".into(), "last".into());

        assert_eq!(candidate_hash(&a), candidate_hash(&b));
    }

    #[test]
    fn evaluation_cache_round_trip() {
        let mut cache = EvaluationCache::new();
        let mut candidate = Candidate::new();
        candidate.insert("instructions".into(), "test".into());
        let id: usize = 42;

        cache.put(&candidate, &id, serde_json::json!("output"), 0.75, None);

        let entry = cache
            .get(&candidate, &id)
            .expect("should find cached entry");
        assert!((entry.score - 0.75).abs() < f64::EPSILON);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn get_pareto_front_mapping_returns_correct_keys() {
        let eval = make_seed_evaluation(&[0, 1], &[0.5, 0.8]);
        let state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        let mapping = state.get_pareto_front_mapping();
        assert_eq!(mapping.len(), 2, "should have one key per val_id");
        for (key, front) in &mapping {
            assert!(
                matches!(key, FrontierKey::Instance { .. }),
                "expected Instance keys for FrontierType::Instance"
            );
            assert!(!front.is_empty(), "front should not be empty");
        }
    }

    #[test]
    fn frontier_type_config_error_when_missing_objectives() {
        let eval = make_seed_evaluation(&[0], &[0.5]);
        let result = GEPAState::new(make_seed_candidate(), eval, FrontierType::Objective, None);
        assert!(
            result.is_err(),
            "should fail when objectives are missing for non-instance frontier"
        );
    }

    // -------------------------------------------------------------------------
    // Helper: build a ValsetEvaluation with objective scores attached.
    // -------------------------------------------------------------------------

    fn make_seed_evaluation_with_objectives(
        ids: &[usize],
        scores: &[f64],
        objectives: Vec<ObjectiveScores>,
    ) -> ValsetEvaluation<usize> {
        let outputs = ids
            .iter()
            .map(|i| serde_json::json!({"id": i}))
            .collect::<Vec<_>>();
        ValsetEvaluation::from_vecs(ids.to_vec(), outputs, scores.to_vec(), Some(objectives))
    }

    // -------------------------------------------------------------------------
    // Gap 1 — FrontierType::Objective
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_with_objective_frontier() {
        // Seed supplies objective scores so Objective frontier is valid.
        let seed_objs: Vec<ObjectiveScores> = vec![
            [("accuracy".into(), 0.5_f64)].into_iter().collect(),
            [("accuracy".into(), 0.6_f64)].into_iter().collect(),
        ];
        let eval = make_seed_evaluation_with_objectives(&[0, 1], &[0.5, 0.6], seed_objs);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Objective, None)
            .expect("Objective frontier should be constructed when objectives present");

        // The seed should be on the objective front.
        assert!(
            state
                .program_at_pareto_front_objectives
                .contains_key("accuracy"),
            "accuracy objective should be in the front"
        );

        // Now add a candidate that beats the seed on accuracy.
        let new_objs: Vec<ObjectiveScores> = vec![
            [("accuracy".into(), 0.9_f64)].into_iter().collect(),
            [("accuracy".into(), 0.9_f64)].into_iter().collect(),
        ];
        let new_eval = make_seed_evaluation_with_objectives(&[0, 1], &[0.9, 0.9], new_objs);
        let mut better = make_seed_candidate();
        better.insert("instructions".into(), "better".into());

        let new_idx = state
            .update_state_with_new_program(vec![0], better, new_eval, 2)
            .expect("update should succeed");

        // The new candidate should now dominate the accuracy objective front.
        let acc_front = state
            .program_at_pareto_front_objectives
            .get("accuracy")
            .expect("accuracy key should exist");
        assert!(
            acc_front.contains(&new_idx),
            "new candidate should be on the accuracy front after improving the score"
        );
        // Old seed should have been displaced.
        assert!(
            !acc_front.contains(&0),
            "seed should no longer be on the accuracy front"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 2 — FrontierType::Cartesian
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_with_cartesian_frontier() {
        let seed_objs: Vec<ObjectiveScores> = vec![
            [("f1".into(), 0.4_f64)].into_iter().collect(),
            [("f1".into(), 0.5_f64)].into_iter().collect(),
        ];
        let eval = make_seed_evaluation_with_objectives(&[0, 1], &[0.4, 0.5], seed_objs);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Cartesian, None)
            .expect("Cartesian frontier should be valid with objectives");

        // Cartesian front should already contain entries for (val_id, objective) pairs.
        assert!(
            !state.pareto_front_cartesian.is_empty(),
            "cartesian front should be seeded"
        );

        // Add a better candidate.
        let new_objs: Vec<ObjectiveScores> = vec![
            [("f1".into(), 0.8_f64)].into_iter().collect(),
            [("f1".into(), 0.9_f64)].into_iter().collect(),
        ];
        let new_eval = make_seed_evaluation_with_objectives(&[0, 1], &[0.8, 0.9], new_objs);
        let mut better = make_seed_candidate();
        better.insert("instructions".into(), "cartesian_better".into());

        let new_idx = state
            .update_state_with_new_program(vec![0], better, new_eval, 2)
            .expect("update should succeed");

        // The Cartesian front for (val_id=0, "f1") should now contain new_idx.
        let key = (0usize, "f1".to_string());
        let front = state
            .program_at_pareto_front_cartesian
            .get(&key)
            .expect("(0, f1) key should exist in cartesian front");
        assert!(
            front.contains(&new_idx),
            "new candidate should dominate (0, f1)"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 3 — FrontierType::Hybrid
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_with_hybrid_frontier() {
        let seed_objs: Vec<ObjectiveScores> = vec![
            [("bleu".into(), 0.3_f64)].into_iter().collect(),
            [("bleu".into(), 0.4_f64)].into_iter().collect(),
        ];
        let eval = make_seed_evaluation_with_objectives(&[10, 20], &[0.3, 0.4], seed_objs);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Hybrid, None)
            .expect("Hybrid frontier should be constructed with objectives");

        let mut better = make_seed_candidate();
        better.insert("instructions".into(), "hybrid_better".into());
        let new_objs: Vec<ObjectiveScores> = vec![
            [("bleu".into(), 0.7_f64)].into_iter().collect(),
            [("bleu".into(), 0.8_f64)].into_iter().collect(),
        ];
        let new_eval = make_seed_evaluation_with_objectives(&[10, 20], &[0.7, 0.8], new_objs);
        state
            .update_state_with_new_program(vec![0], better, new_eval, 2)
            .expect("update should succeed");

        let mapping = state.get_pareto_front_mapping();
        let has_instance = mapping
            .keys()
            .any(|k| matches!(k, FrontierKey::Instance { .. }));
        let has_objective = mapping
            .keys()
            .any(|k| matches!(k, FrontierKey::Objective { .. }));
        assert!(has_instance, "Hybrid mapping should contain Instance keys");
        assert!(
            has_objective,
            "Hybrid mapping should contain Objective keys"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 4 — Pareto tie keeps both candidates in the set
    // -------------------------------------------------------------------------

    #[test]
    fn test_pareto_tie_adds_to_set() {
        // Seed has score 0.7 on val_id=0.
        let eval = make_seed_evaluation(&[0, 1], &[0.7, 0.5]);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        // Add a second candidate with the EXACT SAME score on val_id=0.
        let new_eval = ValsetEvaluation::from_vecs(
            vec![0, 1],
            vec![serde_json::json!("a"), serde_json::json!("b")],
            vec![0.7, 0.5], // same score on id=0
            None,
        );
        let mut tied = make_seed_candidate();
        tied.insert("instructions".into(), "tied candidate".into());
        state
            .update_state_with_new_program(vec![0], tied, new_eval, 2)
            .expect("update should succeed");

        // Both index 0 and index 1 should be in the instance front for val_id=0.
        let front = state
            .program_at_pareto_front_valset
            .get(&0usize)
            .expect("val_id=0 should have a front entry");
        assert!(
            front.contains(&0),
            "seed (idx 0) should remain in front on tie"
        );
        assert!(
            front.contains(&1),
            "new candidate (idx 1) should also be in front on tie"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 5 — Cartesian update_state with missing objective_scores returns Err
    // -------------------------------------------------------------------------

    #[test]
    fn test_cartesian_missing_objectives_errors() {
        let seed_objs: Vec<ObjectiveScores> =
            vec![[("rouge".into(), 0.5_f64)].into_iter().collect()];
        let eval = make_seed_evaluation_with_objectives(&[0], &[0.5], seed_objs);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Cartesian, None)
            .expect("construction should succeed");

        // New evaluation deliberately omits objective_scores_by_val_id.
        let bad_eval = ValsetEvaluation::from_vecs(
            vec![0usize],
            vec![serde_json::json!("out")],
            vec![0.9],
            None, // ← missing objectives required by Cartesian
        );
        let mut new_candidate = make_seed_candidate();
        new_candidate.insert("instructions".into(), "no-obj candidate".into());

        let result = state.update_state_with_new_program(vec![0], new_candidate, bad_eval, 1);
        assert!(
            result.is_err(),
            "Cartesian frontier update without objective scores should return Err"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 6 — JSON round-trip with multiple candidates
    // -------------------------------------------------------------------------

    #[test]
    fn test_state_json_round_trip() {
        let eval = make_seed_evaluation(&[0, 1, 2], &[0.3, 0.5, 0.7]);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        // Add two more candidates.
        for (i, scores) in [(vec![0.4, 0.6, 0.8]), (vec![0.5, 0.7, 0.9])]
            .iter()
            .enumerate()
        {
            let new_eval = ValsetEvaluation::from_vecs(
                vec![0, 1, 2],
                vec![
                    serde_json::json!("x"),
                    serde_json::json!("y"),
                    serde_json::json!("z"),
                ],
                scores.clone(),
                None,
            );
            let mut c = make_seed_candidate();
            c.insert("instructions".into(), format!("candidate {i}"));
            state
                .update_state_with_new_program(vec![0], c, new_eval, i + 2)
                .expect("update should succeed");
        }

        assert_eq!(state.program_candidates.len(), 3);

        let json = state.to_json().expect("serialisation should succeed");
        let restored: GEPAState<usize> =
            GEPAState::from_json(&json).expect("deserialisation should succeed");

        restored
            .is_consistent()
            .expect("restored state should be consistent");
        assert_eq!(
            restored.program_candidates.len(),
            state.program_candidates.len(),
        );

        // Compare scores with tolerance to allow for JSON floating-point round-trip imprecision.
        let original_scores = state.program_full_scores_val_set();
        let restored_scores = restored.program_full_scores_val_set();
        assert_eq!(
            restored_scores.len(),
            original_scores.len(),
            "restored scores length should match"
        );
        for (orig, rest) in original_scores.iter().zip(restored_scores.iter()) {
            assert!(
                (orig - rest).abs() < 1e-10,
                "score {rest} should be approximately equal to {orig} after JSON round-trip"
            );
        }

        assert_eq!(
            restored.pareto_front_valset.len(),
            state.pareto_front_valset.len(),
        );
    }

    // -------------------------------------------------------------------------
    // Gap 7 — from_json rejects a schema version from the future
    // -------------------------------------------------------------------------

    #[test]
    fn test_state_from_json_rejects_future_schema() {
        let future_version = GEPAState::<usize>::SCHEMA_VERSION + 1;
        let json = serde_json::json!({
            "validation_schema_version": future_version,
            "program_candidates": [],
            "parent_program_for_candidate": [],
            "prog_candidate_val_subscores": [],
            "prog_candidate_objective_scores": [],
            "num_metric_calls_by_discovery": [],
            "pareto_front_valset": [],
            "program_at_pareto_front_valset": [],
            "objective_pareto_front": {},
            "program_at_pareto_front_objectives": {},
            "list_of_named_predictors": [],
            "named_predictor_id_to_update_next": [],
            "frontier_type": "instance",
            "i": 0,
            "num_full_ds_evals": 0,
            "total_num_evals": 0
        })
        .to_string();

        let result = GEPAState::<usize>::from_json(&json);
        assert!(
            result.is_err(),
            "from_json should reject a future schema version"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 8 — round-robin counter inherits the max parent counter
    // -------------------------------------------------------------------------

    #[test]
    fn test_round_robin_counter_inherits_max_parent() {
        let eval = make_seed_evaluation(&[0, 1], &[0.5, 0.6]);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        // Add candidate 1 (parent: 0).
        let eval1 = ValsetEvaluation::from_vecs(
            vec![0, 1],
            vec![serde_json::json!("a"), serde_json::json!("b")],
            vec![0.6, 0.7],
            None,
        );
        let mut c1 = make_seed_candidate();
        c1.insert("instructions".into(), "child1".into());
        state
            .update_state_with_new_program(vec![0], c1, eval1, 2)
            .expect("update should succeed");

        // Manually advance the counter for candidate 1 to simulate usage.
        state.named_predictor_id_to_update_next[1] = 5;

        // Add candidate 2 (parent: 1).
        let eval2 = ValsetEvaluation::from_vecs(
            vec![0, 1],
            vec![serde_json::json!("c"), serde_json::json!("d")],
            vec![0.7, 0.8],
            None,
        );
        let mut c2 = make_seed_candidate();
        c2.insert("instructions".into(), "child2".into());
        state
            .update_state_with_new_program(vec![1], c2, eval2, 3)
            .expect("update should succeed");

        // Candidate 2 should have inherited max counter from parent 1 (value=5).
        assert_eq!(
            state.named_predictor_id_to_update_next[2], 5,
            "child should inherit the max parent round-robin counter"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 9 — is_consistent detects a length mismatch
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_consistent_detects_length_mismatch() {
        let eval = make_seed_evaluation(&[0, 1], &[0.5, 0.6]);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        // Manually break the invariant by pushing an extra entry to the parent list
        // without a corresponding entry in program_candidates.
        state.parent_program_for_candidate.push(vec![Some(0)]);

        let result = state.is_consistent();
        assert!(
            result.is_err(),
            "is_consistent should detect the length mismatch"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 10 — evaluation cache get_batch splits cached / uncached correctly
    // -------------------------------------------------------------------------

    #[test]
    fn test_evaluation_cache_get_batch() {
        let mut cache = EvaluationCache::new();
        let mut candidate = make_seed_candidate();
        candidate.insert("instructions".into(), "cache_test".into());

        // Pre-populate ids 0, 1, 2.
        for id in 0usize..3 {
            cache.put(
                &candidate,
                &id,
                serde_json::json!(id),
                id as f64 * 0.1,
                None,
            );
        }

        // Query ids 0, 1, 2 (cached) and 3, 4 (uncached).
        let query_ids: Vec<usize> = vec![0, 1, 2, 3, 4];
        let (cached_results, uncached_ids) = cache.get_batch(&candidate, &query_ids);

        assert_eq!(cached_results.len(), 3, "should find 3 cached entries");
        assert_eq!(uncached_ids.len(), 2, "should report 2 uncached entries");
        assert!(uncached_ids.contains(&3));
        assert!(uncached_ids.contains(&4));

        // Verify cached scores are correct.
        assert!((cached_results[&0usize].score - 0.0).abs() < f64::EPSILON);
        assert!((cached_results[&1usize].score - 0.1).abs() < f64::EPSILON);
        assert!((cached_results[&2usize].score - 0.2).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Gap 11 — scale test: 100 candidates
    // -------------------------------------------------------------------------

    #[test]
    fn test_state_at_scale_100_candidates() {
        let val_ids: Vec<usize> = (0..5).collect();
        let seed_scores: Vec<f64> = val_ids.iter().map(|i| *i as f64 * 0.1).collect();
        let eval = make_seed_evaluation(&val_ids, &seed_scores);
        let mut state = GEPAState::new(make_seed_candidate(), eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        // Add 99 more candidates with varied scores (total 100).
        for n in 1usize..100 {
            let scores: Vec<f64> = val_ids
                .iter()
                .map(|i| ((n * 7 + i) % 11) as f64 / 10.0)
                .collect();
            let outputs: Vec<serde_json::Value> =
                scores.iter().map(|s| serde_json::json!(s)).collect();
            let new_eval = ValsetEvaluation::from_vecs(val_ids.clone(), outputs, scores, None);
            let mut c = make_seed_candidate();
            c.insert("instructions".into(), format!("candidate {n}"));
            state
                .update_state_with_new_program(vec![0], c, new_eval, n)
                .expect("update should succeed");
        }

        assert_eq!(state.program_candidates.len(), 100);
        state
            .is_consistent()
            .expect("state should be consistent at scale");

        let full_scores = state.program_full_scores_val_set();
        assert_eq!(full_scores.len(), 100);

        let mapping = state.get_pareto_front_mapping();
        // Each val_id should have at least one candidate on the front.
        assert_eq!(mapping.len(), val_ids.len());
        for front in mapping.values() {
            assert!(!front.is_empty());
        }
    }
}
