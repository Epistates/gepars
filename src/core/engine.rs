/// `GEPAEngine` — Algorithm 1 of the GEPA paper.
///
/// The engine owns the full optimisation loop:
/// 1. Load a saved [`GEPAState`] when available, otherwise evaluate the seed
///    candidate on the full validation set and build the initial state.
/// 2. Loop until the stop condition fires:
///    a. If a merge attempt is scheduled, invoke [`MergeProposer::propose_mut`].
///       Accept if `sum(new_scores) >= max(p1_sum, p2_sum)`.
///    b. Otherwise, invoke [`ReflectiveMutationProposer::propose_mut`].
///       Accept if `sum(new_scores) > sum(old_scores)` (strict improvement).
///    c. On acceptance, evaluate the new candidate on the full validation set
///       and add it to the state via `update_state_with_new_program`.
///    d. On a successful mutation, schedule a merge for the next iteration.
/// 3. Return a [`GEPAResult`] snapshot.
///
/// Mirrors `gepa.core.engine.GEPAEngine`.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;
use tracing::{debug, info};

use crate::core::adapter::{Candidate, EvaluationBatch, GEPAAdapter};
use crate::core::callbacks::{
    BudgetUpdatedEvent, CandidateAcceptedEvent, CandidateRejectedEvent, EvaluationEndEvent,
    EvaluationStartEvent, GEPACallback, IterationEndEvent, IterationStartEvent, MergeAcceptedEvent,
    MergeAttemptedEvent, MergeRejectedEvent, OptimizationEndEvent, OptimizationStartEvent,
    StateSavedEvent, ValsetEvaluatedEvent, notify_callbacks,
};
use crate::core::data_loader::{DataId, DataLoader};
use crate::core::result::GEPAResult;
use crate::core::state::{
    CachedEvaluation, EvaluationCache, FrontierType, GEPAState, ObjectiveScores, ValsetEvaluation,
};
use crate::error::{GEPAError, Result};
use crate::proposer::merge::MergeProposer;
use crate::proposer::reflective_mutation::ReflectiveMutationProposer;
use crate::strategies::eval_policy::EvalPolicy;
use crate::tracking::ExperimentTracker;
use crate::utils::stop_condition::StopCondition;

// ---------------------------------------------------------------------------
// GEPAEngine
// ---------------------------------------------------------------------------

/// The top-level optimisation engine.
///
/// ### Type parameters
/// - `Id`   — data example identifier (implements [`DataId`]).
/// - `Item` — training / validation data instance type.
/// - `T`    — execution trace type (opaque to the engine).
/// - `RO`   — raw rollout output type, serialized for callbacks/cache/results.
///
/// ### Ownership model
/// The engine owns all strategies and proposers.  All shared resources
/// (adapter, data loaders) are wrapped in `Arc`.
pub struct GEPAEngine<Id, Item, T, RO>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + Serialize + 'static,
{
    // ---- Data ---------------------------------------------------------------
    /// Training data loader (used by proposers internally via `Arc` clone).
    pub trainset: Arc<dyn DataLoader<Id, Item>>,
    /// Validation data loader.
    pub valset: Arc<dyn DataLoader<Id, Item>>,

    // ---- Adapter ------------------------------------------------------------
    /// User-supplied adapter: evaluates candidates and builds reflective datasets.
    pub adapter: Arc<dyn GEPAAdapter<Item, T, RO>>,

    // ---- Seed candidate -----------------------------------------------------
    /// The initial candidate the engine starts from.
    pub(crate) seed_candidate: Candidate,

    // ---- Proposers ----------------------------------------------------------
    /// Reflective mutation proposer (Algorithm 3).
    pub mutation_proposer: ReflectiveMutationProposer<Id, Item, T, RO>,
    /// Merge proposer (Algorithm 4).
    pub merge_proposer: MergeProposer<Id, Item, T, RO>,

    // ---- Strategies ---------------------------------------------------------
    /// Evaluation policy: determines which val IDs to evaluate and picks best.
    pub eval_policy: Box<dyn EvalPolicy<Id>>,
    /// Stop condition(s) checked at the top of every iteration.
    pub stop_condition: Box<dyn StopCondition<Id>>,

    // ---- Frontier configuration ---------------------------------------------
    /// Frontier strategy forwarded to `GEPAState::new`.
    pub frontier_type: FrontierType,

    // ---- Callbacks ----------------------------------------------------------
    /// Observational callbacks registered for this run.
    pub(crate) callbacks: Vec<Box<dyn GEPACallback<Id>>>,

    // ---- Run configuration --------------------------------------------------
    /// RNG seed used (metadata only — stored in the result).
    pub rng_seed: Option<u64>,
    /// Directory to persist run artefacts (metadata only — stored in the result).
    pub(crate) run_dir: Option<String>,
    /// When set, `GEPAResult::best_candidate` unwraps to the plain string stored
    /// under this key.
    pub str_candidate_key: Option<String>,
    /// Whether to track best per-val-id outputs in the state.
    pub track_best_outputs: bool,
    /// Whether to use an evaluation cache.
    pub cache_evaluation: bool,
    /// Optional experiment tracker (Gap 9).
    pub tracker: Box<dyn ExperimentTracker>,
}

impl<Id, Item, T, RO> GEPAEngine<Id, Item, T, RO>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + Serialize + 'static,
{
    // ------------------------------------------------------------------
    // run — Algorithm 1
    // ------------------------------------------------------------------

    /// Execute the full GEPA optimisation loop and return a result snapshot.
    ///
    /// # Errors
    /// Returns `Err` on unrecoverable adapter failures or when the seed
    /// evaluation produces no validation examples.
    pub async fn run(&mut self) -> Result<GEPAResult<Id>> {
        // ── Step 1: collect validation IDs ───────────────────────────────────
        let all_val_ids = self.valset.all_ids();

        if all_val_ids.is_empty() {
            return Err(GEPAError::Config(
                "Validation set is empty — cannot initialise GEPAState".into(),
            ));
        }

        let trainset_ids = self.trainset.all_ids();

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_optimization_start(&OptimizationStartEvent {
                seed_candidate: self.seed_candidate.clone(),
                trainset_size: trainset_ids.len(),
                valset_size: all_val_ids.len(),
                config: HashMap::new(),
            });
        });

        // ── Step 2: build or resume state ────────────────────────────────────
        let state_path = self
            .run_dir
            .as_ref()
            .map(|run_dir| Path::new(run_dir).join("gepa_state.json"));

        let loaded_state = if let Some(state_path) = state_path.as_ref() {
            if state_path.exists() {
                info!(path = %state_path.display(), "Resuming from saved state");
                let json = std::fs::read_to_string(state_path)
                    .map_err(|e| GEPAError::Config(format!("Failed to read state file: {e}")))?;
                let mut loaded = GEPAState::<Id>::from_json(&json)?;
                // Sync cache with current run setting.
                if self.cache_evaluation {
                    if loaded.evaluation_cache.is_none() {
                        loaded.evaluation_cache = Some(EvaluationCache::new());
                    }
                } else {
                    loaded.evaluation_cache = None;
                }
                Some(loaded)
            } else {
                None
            }
        } else {
            None
        };

        let (mut state, initial_budget_delta) = if let Some(loaded) = loaded_state {
            (loaded, 0)
        } else {
            info!(
                valset_size = all_val_ids.len(),
                "Evaluating seed candidate on full validation set"
            );

            notify_callbacks(&self.callbacks, |cb| {
                cb.on_evaluation_start(&EvaluationStartEvent {
                    iteration: 0,
                    candidate_idx: None,
                    batch_size: all_val_ids.len(),
                    capture_traces: false,
                    parent_ids: vec![],
                    is_seed_candidate: true,
                });
            });

            let seed_outcome = self
                .evaluate_valset_uncached(&self.seed_candidate, &all_val_ids)
                .await?;

            notify_callbacks(&self.callbacks, |cb| {
                cb.on_evaluation_end(&EvaluationEndEvent {
                    iteration: 0,
                    candidate_idx: None,
                    scores: seed_outcome.scores.clone(),
                    has_trajectories: false,
                    parent_ids: vec![],
                    outputs: seed_outcome.outputs.clone(),
                    objective_scores: seed_outcome.objective_scores.clone(),
                    is_seed_candidate: true,
                });
            });

            let mut cache = if self.cache_evaluation {
                Some(EvaluationCache::new())
            } else {
                None
            };
            if let Some(ref mut cache) = cache {
                cache.put_batch(
                    &self.seed_candidate,
                    &all_val_ids,
                    seed_outcome.outputs.clone(),
                    seed_outcome.scores.clone(),
                    seed_outcome.objective_scores.clone(),
                );
            }

            let state = GEPAState::new_with_options(
                self.seed_candidate.clone(),
                seed_outcome.valset_evaluation,
                self.frontier_type,
                cache,
                self.track_best_outputs,
            )?;
            (state, seed_outcome.metric_evals)
        };

        if state.evaluation_cache.is_some() != self.cache_evaluation {
            state.evaluation_cache = if self.cache_evaluation {
                Some(EvaluationCache::new())
            } else {
                None
            };
        }

        // Seed counted as one logical full-valset evaluation only on fresh runs.
        if state.num_full_ds_evals == 0 {
            state.num_full_ds_evals = 1;
            state.total_num_evals = initial_budget_delta;
        }

        if state.program_candidates.is_empty() {
            return Err(GEPAError::NoCandidates);
        }

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_budget_updated(&BudgetUpdatedEvent {
                iteration: 0,
                metric_calls_used: state.total_num_evals,
                metric_calls_delta: initial_budget_delta,
                metric_calls_remaining: None,
            });
        });

        let (seed_score, _) = state.get_program_average_val_subset(0);
        info!(seed_score, "Seed candidate available");

        // Log seed metrics to tracker (Gap 9).
        self.tracker.log_metric("seed_score", seed_score, 0);

        // ── Step 3: main optimisation loop ───────────────────────────────────
        loop {
            // Advance iteration counter.
            // BEFORE_FIRST_ITERATION = usize::MAX; wrapping_add(1) → 0.
            state.i = state.i.wrapping_add(1);

            if self.stop_condition.should_stop(&state) {
                info!(
                    iteration = state.i,
                    total_evals = state.total_num_evals,
                    stop_reason = %self.stop_condition.description(),
                    "Stop condition satisfied — exiting optimisation loop"
                );
                break;
            }

            notify_callbacks(&self.callbacks, |cb| {
                cb.on_iteration_start(&IterationStartEvent {
                    iteration: state.i,
                    state: &state,
                    trainset_size: trainset_ids.len(),
                });
            });

            debug!(iteration = state.i, "Starting iteration");

            // Snapshot candidate count before step to detect acceptance.
            let candidates_before = state.program_candidates.len();
            let tag = if self.merge_proposer.merges_due > 0 {
                "merge"
            } else {
                "reflective_mutation"
            };

            let accepted = if self.merge_proposer.merges_due > 0 {
                // ── Merge attempt (Algorithm 4) ──────────────────────────────
                self.try_merge_step(&mut state, &all_val_ids).await?
            } else {
                // ── Reflective mutation (Algorithm 3) ────────────────────────
                self.try_mutation_step(&mut state, &all_val_ids).await?
            };

            // ── Gap 4: push a trace entry ────────────────────────────────────
            let new_program_idx = if accepted && state.program_candidates.len() > candidates_before
            {
                Some(state.program_candidates.len() - 1)
            } else {
                None
            };
            let trace_entry = serde_json::json!({
                "iteration": state.i,
                "accepted": accepted,
                "new_program_idx": new_program_idx,
                "tag": tag,
                "total_num_evals": state.total_num_evals,
            });
            state.full_program_trace.push(trace_entry);

            // ── Gap 9: log iteration metrics to tracker ───────────────────────
            let best_score = state
                .program_full_scores_val_set()
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            self.tracker.log_metric("best_score", best_score, state.i);
            self.tracker
                .log_metric("total_evals", state.total_num_evals as f64, state.i);

            // ── Gap 3: auto-save state after acceptance ───────────────────────
            if accepted && let Some(ref run_dir) = self.run_dir {
                let state_path = Path::new(run_dir).join("gepa_state.json");
                match state.to_json() {
                    Ok(json) => {
                        let tmp_path = Path::new(run_dir).join("gepa_state.json.tmp");
                        if let Err(e) = std::fs::create_dir_all(run_dir)
                            .and_then(|()| std::fs::write(&tmp_path, &json))
                            .and_then(|()| std::fs::rename(&tmp_path, &state_path))
                        {
                            tracing::warn!(error = %e, "Failed to save state to disk");
                        } else {
                            debug!(path = %state_path.display(), "State saved");
                            notify_callbacks(&self.callbacks, |cb| {
                                cb.on_state_saved(&StateSavedEvent {
                                    iteration: state.i,
                                    run_dir: Some(run_dir.clone()),
                                });
                            });
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to serialise state");
                    }
                }
            }

            notify_callbacks(&self.callbacks, |cb| {
                cb.on_iteration_end(&IterationEndEvent {
                    iteration: state.i,
                    state: &state,
                    proposal_accepted: accepted,
                });
            });
        }

        // ── Step 5: build and return result snapshot ─────────────────────────
        let best_idx = self.eval_policy.get_best_program(&state)?;

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_optimization_end(&OptimizationEndEvent {
                best_candidate_idx: best_idx,
                total_iterations: state.i.wrapping_add(1),
                total_metric_calls: state.total_num_evals,
                final_state: &state,
            });
        });

        // Finish tracker (Gap 9).
        self.tracker.finish();

        let result = GEPAResult::from_state(
            &state,
            self.run_dir.clone(),
            self.rng_seed,
            self.str_candidate_key.clone(),
        );

        info!(
            total_candidates = result.num_candidates(),
            total_evals = state.total_num_evals,
            best_idx,
            "Optimisation complete"
        );

        Ok(result)
    }

    // ------------------------------------------------------------------
    // Merge step (Algorithm 4 acceptance logic)
    // ------------------------------------------------------------------

    /// Attempt a merge proposal.  Returns `true` if the merge was accepted and
    /// the new candidate was registered in the state.
    async fn try_merge_step(
        &mut self,
        state: &mut GEPAState<Id>,
        all_val_ids: &[Id],
    ) -> Result<bool> {
        debug!(
            iteration = state.i,
            merges_due = self.merge_proposer.merges_due,
            "Attempting merge"
        );

        let proposal = self.merge_proposer.propose_mut(state).await?;

        let Some(proposal) = proposal else {
            debug!(iteration = state.i, "Merge proposer returned None");
            return Ok(false);
        };

        let merged_candidate = proposal.candidate.clone();
        let parent_ids = proposal.parent_program_ids.clone();

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_merge_attempted(&MergeAttemptedEvent {
                iteration: state.i,
                parent_ids: parent_ids.clone(),
                merged_candidate: merged_candidate.clone(),
            });
        });

        // Acceptance: sum(new_scores) >= max(sum_p1, sum_p2).
        // The merge proposer encodes `subsample_scores_before = [p1_sum, p2_sum]`
        // and `subsample_scores_after = new_sub_scores`.
        let accepted = if let (Some(before), Some(after)) = (
            &proposal.subsample_scores_before,
            &proposal.subsample_scores_after,
        ) {
            let max_parent_sum = before.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_after: f64 = after.iter().sum();
            sum_after >= max_parent_sum
        } else {
            false
        };

        if !accepted {
            notify_callbacks(&self.callbacks, |cb| {
                cb.on_merge_rejected(&MergeRejectedEvent {
                    iteration: state.i,
                    parent_ids: parent_ids.clone(),
                    reason: "sum(new) < max(sum_p1, sum_p2)".into(),
                });
            });
            debug!(iteration = state.i, "Merge rejected");
            return Ok(false);
        }

        // ── Full validation-set evaluation for the accepted merge ──────────
        let (new_idx, evals_delta) = self
            .full_valset_eval_and_accept(state, merged_candidate, parent_ids.clone(), all_val_ids)
            .await?;

        // Only decrement merges_due on acceptance (matching reference behavior).
        if self.merge_proposer.merges_due > 0 {
            self.merge_proposer.merges_due -= 1;
        }

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_merge_accepted(&MergeAcceptedEvent {
                iteration: state.i,
                new_candidate_idx: new_idx,
                parent_ids: parent_ids.clone(),
            });
        });

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_budget_updated(&BudgetUpdatedEvent {
                iteration: state.i,
                metric_calls_used: state.total_num_evals,
                metric_calls_delta: evals_delta,
                metric_calls_remaining: None,
            });
        });

        info!(
            iteration = state.i,
            new_idx,
            parents = ?parent_ids,
            "Merge accepted"
        );

        Ok(true)
    }

    // ------------------------------------------------------------------
    // Reflective mutation step (Algorithm 3 acceptance logic)
    // ------------------------------------------------------------------

    /// Attempt a reflective mutation proposal.  Returns `true` if accepted.
    async fn try_mutation_step(
        &mut self,
        state: &mut GEPAState<Id>,
        all_val_ids: &[Id],
    ) -> Result<bool> {
        let proposal = self.mutation_proposer.propose_mut(state).await?;

        let Some(proposal) = proposal else {
            debug!(
                iteration = state.i,
                "Mutation proposer returned None — skipping iteration"
            );
            return Ok(false);
        };

        // Mutation acceptance: strict improvement (sum_after > sum_before).
        let accepted = proposal.shows_improvement().unwrap_or(false);
        let candidate = proposal.candidate.clone();
        let parent_ids = proposal.parent_program_ids.clone();

        if !accepted {
            let sum_before: f64 = proposal
                .subsample_scores_before
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .sum();
            let sum_after: f64 = proposal
                .subsample_scores_after
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .sum();

            notify_callbacks(&self.callbacks, |cb| {
                cb.on_candidate_rejected(&CandidateRejectedEvent {
                    iteration: state.i,
                    old_score: sum_before,
                    new_score: sum_after,
                    reason: "no strict improvement on minibatch".into(),
                });
            });

            debug!(
                iteration = state.i,
                sum_before, sum_after, "Mutation rejected — no strict minibatch improvement"
            );
            return Ok(false);
        }

        // ── Full validation-set evaluation for the accepted mutation ───────
        let (new_idx, evals_delta) = self
            .full_valset_eval_and_accept(state, candidate, parent_ids.clone(), all_val_ids)
            .await?;

        let new_score = state.get_program_average_val_subset(new_idx).0;

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_candidate_accepted(&CandidateAcceptedEvent {
                iteration: state.i,
                new_candidate_idx: new_idx,
                new_score,
                parent_ids: parent_ids.clone(),
            });
        });

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_budget_updated(&BudgetUpdatedEvent {
                iteration: state.i,
                metric_calls_used: state.total_num_evals,
                metric_calls_delta: evals_delta,
                metric_calls_remaining: None,
            });
        });

        // Schedule a merge after every successful mutation.
        self.merge_proposer.last_iter_found_new_program = true;
        self.merge_proposer.schedule_if_needed();

        info!(
            iteration = state.i,
            new_idx,
            new_score,
            parents = ?parent_ids,
            "Mutation accepted"
        );

        Ok(true)
    }

    // ------------------------------------------------------------------
    // Shared helper: full valset evaluation + state update
    // ------------------------------------------------------------------

    /// Evaluate `candidate` on the full validation set and register it in the
    /// state.  Returns `(new_program_idx, num_eval_calls)`.
    async fn full_valset_eval_and_accept(
        &self,
        state: &mut GEPAState<Id>,
        candidate: Candidate,
        parent_ids: Vec<usize>,
        all_val_ids: &[Id],
    ) -> Result<(usize, usize)> {
        notify_callbacks(&self.callbacks, |cb| {
            cb.on_evaluation_start(&EvaluationStartEvent {
                iteration: state.i,
                candidate_idx: None, // not yet assigned
                batch_size: all_val_ids.len(),
                capture_traces: false,
                parent_ids: parent_ids.clone(),
                is_seed_candidate: false,
            });
        });

        let outcome = self
            .evaluate_valset_with_cache(state, &candidate, all_val_ids)
            .await?;

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_evaluation_end(&EvaluationEndEvent {
                iteration: state.i,
                candidate_idx: None,
                scores: outcome.scores.clone(),
                has_trajectories: false,
                parent_ids: parent_ids.clone(),
                outputs: outcome.outputs.clone(),
                objective_scores: outcome.objective_scores.clone(),
                is_seed_candidate: false,
            });
        });

        let evals_delta = outcome.metric_evals;
        state.increment_evals(evals_delta);
        state.num_full_ds_evals += 1;

        // Note: `total_num_evals` is already updated by `increment_evals`; use it
        // as the discovery-count for the new program.
        let discovery_count = state.total_num_evals;
        let new_idx = state.update_state_with_new_program(
            parent_ids,
            candidate,
            outcome.valset_evaluation,
            discovery_count,
        )?;

        let avg_score = state.get_program_average_val_subset(new_idx).0;

        let scores_by_val_id: HashMap<serde_json::Value, f64> = all_val_ids
            .iter()
            .zip(outcome.scores.iter())
            .map(|(id, &s)| {
                (
                    serde_json::to_value(id).unwrap_or(serde_json::Value::Null),
                    s,
                )
            })
            .collect();

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_valset_evaluated(&ValsetEvaluatedEvent {
                iteration: state.i,
                candidate_idx: new_idx,
                candidate: state.program_candidates[new_idx].clone(),
                scores_by_val_id: scores_by_val_id.clone(),
                average_score: avg_score,
                num_examples_evaluated: all_val_ids.len(),
                total_valset_size: all_val_ids.len(),
                parent_ids: state.parent_program_for_candidate[new_idx]
                    .iter()
                    .filter_map(|p| *p)
                    .collect(),
                is_best_program: {
                    let scores = state.program_full_scores_val_set();
                    scores.get(new_idx).copied().unwrap_or(f64::NEG_INFINITY)
                        >= scores.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                },
            });
        });

        Ok((new_idx, evals_delta))
    }

    async fn evaluate_valset_with_cache(
        &self,
        state: &mut GEPAState<Id>,
        candidate: &Candidate,
        all_val_ids: &[Id],
    ) -> Result<FullValsetEvaluation<Id>> {
        if state.evaluation_cache.is_none() {
            return self.evaluate_valset_uncached(candidate, all_val_ids).await;
        }

        let (cached_owned, uncached_ids) = {
            let Some(cache) = state.evaluation_cache.as_ref() else {
                return self.evaluate_valset_uncached(candidate, all_val_ids).await;
            };
            let (cached_refs, uncached) = cache.get_batch(candidate, all_val_ids);
            let cached = cached_refs
                .into_iter()
                .map(|(id, entry)| (id, entry.clone()))
                .collect::<HashMap<_, _>>();
            (cached, uncached)
        };

        let uncached = if uncached_ids.is_empty() {
            None
        } else {
            Some(
                self.evaluate_valset_uncached(candidate, &uncached_ids)
                    .await?,
            )
        };

        if let Some(outcome) = uncached.as_ref()
            && let Some(cache) = state.evaluation_cache.as_mut()
        {
            cache.put_batch(
                candidate,
                &uncached_ids,
                outcome.outputs.clone(),
                outcome.scores.clone(),
                outcome.objective_scores.clone(),
            );
        }

        merge_cached_and_uncached(all_val_ids, cached_owned, uncached_ids, uncached)
    }

    async fn evaluate_valset_uncached(
        &self,
        candidate: &Candidate,
        val_ids: &[Id],
    ) -> Result<FullValsetEvaluation<Id>> {
        let items = self.valset.fetch(val_ids)?;
        let batch = self.adapter.evaluate(&items, candidate, false).await?;
        build_valset_evaluation_with_metric_delta(val_ids, &batch, val_ids.len())
    }
}

#[derive(Debug, Clone)]
struct FullValsetEvaluation<Id: DataId> {
    valset_evaluation: ValsetEvaluation<Id>,
    scores: Vec<f64>,
    outputs: Vec<serde_json::Value>,
    objective_scores: Option<Vec<ObjectiveScores>>,
    metric_evals: usize,
}

// ---------------------------------------------------------------------------
// Helper: build ValsetEvaluation from EvaluationBatch
// ---------------------------------------------------------------------------

/// Build a [`ValsetEvaluation`] from parallel ID and batch-result slices.
#[cfg(test)]
fn build_valset_evaluation<Id: DataId, T: Send, RO: Send + Serialize>(
    ids: &[Id],
    batch: &EvaluationBatch<T, RO>,
) -> Result<FullValsetEvaluation<Id>> {
    build_valset_evaluation_with_metric_delta(ids, batch, ids.len())
}

fn build_valset_evaluation_with_metric_delta<Id: DataId, T: Send, RO: Send + Serialize>(
    ids: &[Id],
    batch: &EvaluationBatch<T, RO>,
    metric_evals: usize,
) -> Result<FullValsetEvaluation<Id>> {
    batch.validate_lengths(ids.len(), false)?;
    let outputs = batch.outputs_as_json()?;

    let scores_by_val_id: std::collections::HashMap<Id, f64> = ids
        .iter()
        .cloned()
        .zip(batch.scores.iter().copied())
        .collect();

    let objective_scores_by_val_id = batch.objective_scores.as_ref().map(|obj_vec| {
        ids.iter()
            .cloned()
            .zip(obj_vec.iter().cloned())
            .collect::<std::collections::HashMap<Id, _>>()
    });

    let outputs_by_val_id: std::collections::HashMap<Id, serde_json::Value> =
        ids.iter().cloned().zip(outputs.iter().cloned()).collect();

    Ok(FullValsetEvaluation {
        valset_evaluation: ValsetEvaluation {
            outputs_by_val_id,
            scores_by_val_id,
            objective_scores_by_val_id,
        },
        scores: batch.scores.clone(),
        outputs,
        objective_scores: batch.objective_scores.clone(),
        metric_evals,
    })
}

fn merge_cached_and_uncached<Id: DataId>(
    all_val_ids: &[Id],
    cached: HashMap<Id, CachedEvaluation>,
    uncached_ids: Vec<Id>,
    uncached: Option<FullValsetEvaluation<Id>>,
) -> Result<FullValsetEvaluation<Id>> {
    let metric_evals = uncached.as_ref().map_or(0, |outcome| outcome.metric_evals);

    let mut outputs_by_val_id: HashMap<Id, serde_json::Value> = cached
        .iter()
        .map(|(id, entry)| (id.clone(), entry.output.clone()))
        .collect();
    let mut scores_by_val_id: HashMap<Id, f64> = cached
        .iter()
        .map(|(id, entry)| (id.clone(), entry.score))
        .collect();
    let mut objective_scores_by_val_id: Option<HashMap<Id, ObjectiveScores>> = {
        let cached_objectives = cached
            .iter()
            .filter_map(|(id, entry)| {
                entry
                    .objective_scores
                    .as_ref()
                    .map(|objectives| (id.clone(), objectives.clone()))
            })
            .collect::<HashMap<_, _>>();
        if cached_objectives.is_empty() {
            None
        } else {
            Some(cached_objectives)
        }
    };

    if let Some(outcome) = uncached {
        outputs_by_val_id.extend(outcome.valset_evaluation.outputs_by_val_id);
        scores_by_val_id.extend(outcome.valset_evaluation.scores_by_val_id);

        if let Some(objectives) = outcome.valset_evaluation.objective_scores_by_val_id {
            objective_scores_by_val_id
                .get_or_insert_with(HashMap::new)
                .extend(objectives);
        }
    } else if !uncached_ids.is_empty() {
        return Err(GEPAError::Evaluation(
            "internal cache error: uncached IDs were present but no evaluation was provided".into(),
        ));
    }

    let mut ordered_outputs = Vec::with_capacity(all_val_ids.len());
    let mut ordered_scores = Vec::with_capacity(all_val_ids.len());
    let mut ordered_objectives = objective_scores_by_val_id
        .as_ref()
        .map(|_| Vec::with_capacity(all_val_ids.len()));

    for id in all_val_ids {
        let output = outputs_by_val_id.get(id).cloned().ok_or_else(|| {
            GEPAError::Evaluation(format!("missing output for validation id {id:?}"))
        })?;
        let score = scores_by_val_id.get(id).copied().ok_or_else(|| {
            GEPAError::Evaluation(format!("missing score for validation id {id:?}"))
        })?;

        ordered_outputs.push(output);
        ordered_scores.push(score);
        if let Some(ref mut objectives) = ordered_objectives {
            let objective_scores = objective_scores_by_val_id
                .as_ref()
                .and_then(|by_id| by_id.get(id))
                .cloned()
                .ok_or_else(|| {
                    GEPAError::Evaluation(format!(
                        "missing objective scores for validation id {id:?}"
                    ))
                })?;
            objectives.push(objective_scores);
        }
    }

    Ok(FullValsetEvaluation {
        valset_evaluation: ValsetEvaluation {
            outputs_by_val_id,
            scores_by_val_id,
            objective_scores_by_val_id,
        },
        scores: ordered_scores,
        outputs: ordered_outputs,
        objective_scores: ordered_objectives,
        metric_evals,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::core::adapter::{Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset};
    use crate::core::data_loader::{DataLoader, VecLoader};
    use crate::core::state::FrontierType;
    use crate::error::Result;
    use crate::lm::LanguageModel;
    use crate::proposer::merge::MergeProposer;
    use crate::proposer::reflective_mutation::ReflectiveMutationProposer;
    use crate::strategies::batch_sampler::EpochShuffledSampler;
    use crate::strategies::candidate_selector::CurrentBestSelector;
    use crate::strategies::component_selector::AllComponentSelector;
    use crate::strategies::eval_policy::FullEvalPolicy;
    use crate::utils::stop_condition::MaxIterationsStopper;

    // ---- Stub implementations ------------------------------------------------

    /// Adapter that always returns score 1.0 and an empty reflective dataset.
    struct ConstantAdapter;

    #[async_trait]
    impl GEPAAdapter<String, (), String> for ConstantAdapter {
        async fn evaluate(
            &self,
            batch: &[String],
            _candidate: &Candidate,
            _capture_traces: bool,
        ) -> Result<EvaluationBatch<(), String>> {
            Ok(EvaluationBatch::new(batch.to_vec(), vec![1.0; batch.len()]))
        }

        async fn make_reflective_dataset(
            &self,
            _candidate: &Candidate,
            _eval_batch: &EvaluationBatch<(), String>,
            components: &[String],
        ) -> Result<ReflectiveDataset> {
            Ok(components.iter().map(|k| (k.clone(), vec![])).collect())
        }
    }

    /// Mock LM that always returns a fenced improved instruction.
    struct MockLM;

    #[async_trait]
    impl LanguageModel for MockLM {
        async fn complete(&self, _prompt: &str) -> Result<String> {
            Ok("```\nImproved instructions v2\n```".into())
        }
    }

    /// Build a minimal engine for unit testing.
    fn make_engine(max_iters: usize) -> GEPAEngine<usize, String, (), String> {
        let trainset_items = vec!["item0".to_string(), "item1".to_string()];
        let valset_items = vec!["val0".to_string(), "val1".to_string()];

        let trainset: Arc<dyn DataLoader<usize, String>> = Arc::new(VecLoader::new(trainset_items));
        let valset: Arc<dyn DataLoader<usize, String>> = Arc::new(VecLoader::new(valset_items));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(ConstantAdapter);

        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "Do the task well.".into());

        let mutation_proposer = ReflectiveMutationProposer {
            trainset: trainset.clone(),
            adapter: adapter.clone(),
            candidate_selector: Box::new(CurrentBestSelector),
            component_selector: Box::new(AllComponentSelector),
            batch_sampler: Box::new(EpochShuffledSampler::new(2, 42).expect("valid sampler")),
            reflection_lm: Arc::new(MockLM),
            reflection_prompt_template: None,
            component_metadata: crate::core::component::ComponentMetaMap::new(),
            perfect_score: Some(1.0),
            skip_perfect_score: false,
        };

        let merge_proposer = MergeProposer::new(
            valset.clone(),
            adapter.clone(),
            false, // merge disabled for unit tests
            0,     // max_merge_invocations
            1,     // val_overlap_floor
            42,    // seed
        )
        .expect("valid merge proposer");

        GEPAEngine {
            trainset,
            valset,
            adapter,
            seed_candidate: seed,
            mutation_proposer,
            merge_proposer,
            eval_policy: Box::new(FullEvalPolicy),
            stop_condition: Box::new(MaxIterationsStopper::new(max_iters)),
            frontier_type: FrontierType::Instance,
            callbacks: vec![],
            rng_seed: None,
            run_dir: None,
            str_candidate_key: None,
            track_best_outputs: false,
            cache_evaluation: false,
            tracker: Box::new(crate::tracking::NoopTracker),
        }
    }

    #[tokio::test]
    async fn engine_runs_and_returns_result() {
        let mut engine = make_engine(2);
        let result = engine.run().await.expect("engine should complete");
        assert!(result.num_candidates() >= 1);
    }

    #[tokio::test]
    async fn engine_seed_candidate_is_first() {
        let mut engine = make_engine(1);
        let result = engine.run().await.expect("engine should complete");
        assert_eq!(
            result.candidates[0].get("instructions").unwrap(),
            "Do the task well."
        );
    }

    #[tokio::test]
    async fn engine_respects_stop_at_zero_iterations() {
        // With max 0 iterations the engine should return after evaluating the
        // seed but before running any iteration body.
        let mut engine = make_engine(0);
        let result = engine.run().await.expect("engine should complete");
        // Only the seed candidate should be present.
        assert_eq!(result.num_candidates(), 1);
    }

    #[tokio::test]
    async fn engine_resume_skips_seed_evaluation() {
        struct FailingAdapter;

        #[async_trait]
        impl GEPAAdapter<String, (), String> for FailingAdapter {
            async fn evaluate(
                &self,
                _batch: &[String],
                _candidate: &Candidate,
                _capture_traces: bool,
            ) -> Result<EvaluationBatch<(), String>> {
                Err(crate::error::GEPAError::Evaluation(
                    "seed evaluation should have been skipped".into(),
                ))
            }

            async fn make_reflective_dataset(
                &self,
                _candidate: &Candidate,
                _eval_batch: &EvaluationBatch<(), String>,
                components: &[String],
            ) -> Result<ReflectiveDataset> {
                Ok(components.iter().map(|k| (k.clone(), vec![])).collect())
            }
        }

        let run_dir = std::env::temp_dir().join(format!(
            "gepa-resume-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_nanos()
        ));
        std::fs::create_dir_all(&run_dir).expect("create temp run dir");

        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "saved".into());
        let eval = ValsetEvaluation::from_vecs(
            vec![0usize],
            vec![serde_json::json!("saved-output")],
            vec![0.7],
            None,
        );
        let mut saved_state =
            GEPAState::new_with_options(seed.clone(), eval, FrontierType::Instance, None, true)
                .expect("state should construct");
        saved_state.num_full_ds_evals = 1;
        saved_state.total_num_evals = 1;
        std::fs::write(
            run_dir.join("gepa_state.json"),
            saved_state.to_json().expect("state should serialize"),
        )
        .expect("write state file");

        let trainset: Arc<dyn DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["train".into()]));
        let valset: Arc<dyn DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["val".into()]));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(FailingAdapter);

        let mutation_proposer = ReflectiveMutationProposer {
            trainset: trainset.clone(),
            adapter: adapter.clone(),
            candidate_selector: Box::new(CurrentBestSelector),
            component_selector: Box::new(AllComponentSelector),
            batch_sampler: Box::new(EpochShuffledSampler::new(1, 0).expect("valid sampler")),
            reflection_lm: Arc::new(MockLM),
            reflection_prompt_template: None,
            component_metadata: crate::core::component::ComponentMetaMap::new(),
            perfect_score: None,
            skip_perfect_score: false,
        };
        let merge_proposer = MergeProposer::new(valset.clone(), adapter.clone(), false, 0, 1, 0)
            .expect("valid merge proposer");

        let mut engine = GEPAEngine {
            trainset,
            valset,
            adapter,
            seed_candidate: seed,
            mutation_proposer,
            merge_proposer,
            eval_policy: Box::new(FullEvalPolicy),
            stop_condition: Box::new(MaxIterationsStopper::new(0)),
            frontier_type: FrontierType::Instance,
            callbacks: vec![],
            rng_seed: None,
            run_dir: Some(run_dir.to_string_lossy().into_owned()),
            str_candidate_key: None,
            track_best_outputs: true,
            cache_evaluation: false,
            tracker: Box::new(crate::tracking::NoopTracker),
        };

        let result = engine.run().await.expect("resume should skip evaluation");
        assert_eq!(result.num_candidates(), 1);

        std::fs::remove_dir_all(run_dir).expect("cleanup temp run dir");
    }

    #[tokio::test]
    async fn engine_empty_valset_returns_error() {
        let trainset: Arc<dyn DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["item".to_string()]));
        let valset: Arc<dyn DataLoader<usize, String>> =
            Arc::new(VecLoader::<String>::new(Vec::new()));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(ConstantAdapter);

        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "test".into());

        let mutation_proposer = ReflectiveMutationProposer {
            trainset: trainset.clone(),
            adapter: adapter.clone(),
            candidate_selector: Box::new(CurrentBestSelector),
            component_selector: Box::new(AllComponentSelector),
            batch_sampler: Box::new(EpochShuffledSampler::new(2, 0).expect("valid sampler")),
            reflection_lm: Arc::new(MockLM),
            reflection_prompt_template: None,
            component_metadata: crate::core::component::ComponentMetaMap::new(),
            perfect_score: Some(1.0),
            skip_perfect_score: false,
        };

        let merge_proposer = MergeProposer::new(valset.clone(), adapter.clone(), false, 0, 1, 0)
            .expect("valid merge proposer");

        let mut engine = GEPAEngine {
            trainset,
            valset,
            adapter,
            seed_candidate: seed,
            mutation_proposer,
            merge_proposer,
            eval_policy: Box::new(FullEvalPolicy),
            stop_condition: Box::new(MaxIterationsStopper::new(1)),
            frontier_type: FrontierType::Instance,
            callbacks: vec![],
            rng_seed: None,
            run_dir: None,
            str_candidate_key: None,
            track_best_outputs: false,
            cache_evaluation: false,
            tracker: Box::new(crate::tracking::NoopTracker),
        };

        let result = engine.run().await;
        assert!(
            result.is_err(),
            "empty valset should produce a Config error"
        );
    }

    #[tokio::test]
    async fn engine_fires_optimization_start_callback() {
        use std::sync::atomic::Ordering;

        struct CountingCallback(Arc<std::sync::atomic::AtomicUsize>);

        impl GEPACallback<usize> for CountingCallback {
            fn on_optimization_start(&self, _event: &OptimizationStartEvent) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut engine = make_engine(0);
        engine
            .callbacks
            .push(Box::new(CountingCallback(counter.clone())));

        engine.run().await.expect("should complete");
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "on_optimization_start should be called exactly once"
        );
    }

    #[tokio::test]
    async fn build_valset_evaluation_maps_correctly() {
        let ids = vec![0usize, 1, 2];
        let batch: EvaluationBatch<(), String> = EvaluationBatch::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec![0.3, 0.6, 0.9],
        );

        let eval = build_valset_evaluation(&ids, &batch).expect("valid batch");
        assert_eq!(eval.valset_evaluation.scores_by_val_id.get(&0), Some(&0.3));
        assert_eq!(eval.valset_evaluation.scores_by_val_id.get(&1), Some(&0.6));
        assert_eq!(eval.valset_evaluation.scores_by_val_id.get(&2), Some(&0.9));
        assert_eq!(
            eval.valset_evaluation.outputs_by_val_id.get(&0),
            Some(&serde_json::json!("a"))
        );
        assert!(eval.valset_evaluation.objective_scores_by_val_id.is_none());
    }

    #[tokio::test]
    async fn build_valset_evaluation_rejects_length_mismatch() {
        let ids = vec![0usize, 1, 2];
        let batch: EvaluationBatch<(), String> = EvaluationBatch::new(vec!["a".into()], vec![0.3]);

        let err = build_valset_evaluation(&ids, &batch).unwrap_err();
        assert!(err.to_string().contains("batch of 3"));
    }

    #[tokio::test]
    async fn evaluate_valset_with_cache_only_evaluates_misses() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingAdapter(Arc<AtomicUsize>);

        #[async_trait]
        impl GEPAAdapter<String, (), String> for CountingAdapter {
            async fn evaluate(
                &self,
                batch: &[String],
                _candidate: &Candidate,
                _capture_traces: bool,
            ) -> Result<EvaluationBatch<(), String>> {
                self.0.fetch_add(batch.len(), Ordering::SeqCst);
                Ok(EvaluationBatch::new(batch.to_vec(), vec![0.5; batch.len()]))
            }

            async fn make_reflective_dataset(
                &self,
                _candidate: &Candidate,
                _eval_batch: &EvaluationBatch<(), String>,
                components: &[String],
            ) -> Result<ReflectiveDataset> {
                Ok(components.iter().map(|k| (k.clone(), vec![])).collect())
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let valset: Arc<dyn DataLoader<usize, String>> = Arc::new(VecLoader::new(vec![
            "cached".into(),
            "miss1".into(),
            "miss2".into(),
        ]));
        let trainset: Arc<dyn DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["train".into()]));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> =
            Arc::new(CountingAdapter(counter.clone()));

        let mut candidate = Candidate::new();
        candidate.insert("instructions".into(), "seed".into());
        let seed_eval = ValsetEvaluation::from_vecs(
            vec![0usize, 1, 2],
            vec![
                serde_json::json!("seed0"),
                serde_json::json!("seed1"),
                serde_json::json!("seed2"),
            ],
            vec![0.1, 0.1, 0.1],
            None,
        );
        let mut cache = EvaluationCache::new();
        cache.put(
            &candidate,
            &0usize,
            serde_json::json!("cached-output"),
            0.9,
            None,
        );
        let mut state = GEPAState::new(
            candidate.clone(),
            seed_eval,
            FrontierType::Instance,
            Some(cache),
        )
        .expect("state should construct");

        let engine = GEPAEngine {
            trainset: trainset.clone(),
            valset,
            adapter: adapter.clone(),
            seed_candidate: candidate.clone(),
            mutation_proposer: ReflectiveMutationProposer {
                trainset,
                adapter: adapter.clone(),
                candidate_selector: Box::new(CurrentBestSelector),
                component_selector: Box::new(AllComponentSelector),
                batch_sampler: Box::new(EpochShuffledSampler::new(1, 0).expect("valid sampler")),
                reflection_lm: Arc::new(MockLM),
                reflection_prompt_template: None,
                component_metadata: crate::core::component::ComponentMetaMap::new(),
                perfect_score: None,
                skip_perfect_score: false,
            },
            merge_proposer: MergeProposer::new(
                Arc::new(VecLoader::new(vec!["cached".to_string()])),
                adapter,
                false,
                0,
                1,
                0,
            )
            .expect("valid merge proposer"),
            eval_policy: Box::new(FullEvalPolicy),
            stop_condition: Box::new(MaxIterationsStopper::new(0)),
            frontier_type: FrontierType::Instance,
            callbacks: vec![],
            rng_seed: None,
            run_dir: None,
            str_candidate_key: None,
            track_best_outputs: false,
            cache_evaluation: true,
            tracker: Box::new(crate::tracking::NoopTracker),
        };

        let outcome = engine
            .evaluate_valset_with_cache(&mut state, &candidate, &[0, 1, 2])
            .await
            .expect("cache-backed evaluation should succeed");

        assert_eq!(outcome.metric_evals, 2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(outcome.scores, vec![0.9, 0.5, 0.5]);
        assert_eq!(outcome.outputs[0], serde_json::json!("cached-output"));
        assert_eq!(
            state.evaluation_cache.as_ref().map(EvaluationCache::len),
            Some(3)
        );
    }

    // -------------------------------------------------------------------------
    // Gap 12 — mutation is rejected when there is no strict improvement
    // -------------------------------------------------------------------------

    /// Adapter that returns score 0.5 for both the current and proposed
    /// candidate on the training minibatch, so `sum_after > sum_before` is
    /// false.  It also returns trajectories so the mutation proposer does not
    /// bail out early at the "no trajectories" guard, and provides a non-empty
    /// reflective dataset so the proposer does not bail at the
    /// "no components" guard.
    struct FlatScoreAdapter;

    #[async_trait]
    impl GEPAAdapter<String, (), String> for FlatScoreAdapter {
        async fn evaluate(
            &self,
            batch: &[String],
            _candidate: &Candidate,
            capture_traces: bool,
        ) -> Result<EvaluationBatch<(), String>> {
            let n = batch.len();
            let mut eb = EvaluationBatch::new(batch.to_vec(), vec![0.5; n]);
            if capture_traces {
                eb = eb.with_trajectories(vec![(); n]);
            }
            Ok(eb)
        }

        async fn make_reflective_dataset(
            &self,
            _candidate: &Candidate,
            _eval_batch: &EvaluationBatch<(), String>,
            components: &[String],
        ) -> Result<ReflectiveDataset> {
            // Return a non-empty record so the proposer proceeds past the
            // "component has no data" guard.
            Ok(components
                .iter()
                .map(|k| {
                    (
                        k.clone(),
                        vec![serde_json::json!({"Inputs": {}, "Generated Outputs": "", "Feedback": ""})],
                    )
                })
                .collect())
        }
    }

    /// Build an engine using the `FlatScoreAdapter`.
    fn make_flat_engine(max_iters: usize) -> GEPAEngine<usize, String, (), String> {
        let trainset_items = vec!["item0".to_string(), "item1".to_string()];
        let valset_items = vec!["val0".to_string(), "val1".to_string()];

        let trainset: Arc<dyn DataLoader<usize, String>> = Arc::new(VecLoader::new(trainset_items));
        let valset: Arc<dyn DataLoader<usize, String>> = Arc::new(VecLoader::new(valset_items));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(FlatScoreAdapter);

        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "Do the task well.".into());

        let mutation_proposer = ReflectiveMutationProposer {
            trainset: trainset.clone(),
            adapter: adapter.clone(),
            candidate_selector: Box::new(CurrentBestSelector),
            component_selector: Box::new(AllComponentSelector),
            batch_sampler: Box::new(EpochShuffledSampler::new(2, 42).expect("valid sampler")),
            reflection_lm: Arc::new(MockLM),
            reflection_prompt_template: None,
            component_metadata: crate::core::component::ComponentMetaMap::new(),
            perfect_score: None,
            skip_perfect_score: false,
        };

        let merge_proposer = MergeProposer::new(valset.clone(), adapter.clone(), false, 0, 1, 42)
            .expect("valid merge proposer");

        GEPAEngine {
            trainset,
            valset,
            adapter,
            seed_candidate: seed,
            mutation_proposer,
            merge_proposer,
            eval_policy: Box::new(FullEvalPolicy),
            stop_condition: Box::new(MaxIterationsStopper::new(max_iters)),
            frontier_type: FrontierType::Instance,
            callbacks: vec![],
            rng_seed: None,
            run_dir: None,
            str_candidate_key: None,
            track_best_outputs: false,
            cache_evaluation: false,
            tracker: Box::new(crate::tracking::NoopTracker),
        }
    }

    #[tokio::test]
    async fn test_mutation_rejection_when_no_improvement() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct RejectionCounter(Arc<AtomicUsize>);

        impl GEPACallback<usize> for RejectionCounter {
            fn on_candidate_rejected(&self, _event: &CandidateRejectedEvent) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let rejection_count = Arc::new(AtomicUsize::new(0));
        let mut engine = make_flat_engine(3);
        engine
            .callbacks
            .push(Box::new(RejectionCounter(rejection_count.clone())));

        let result = engine.run().await.expect("engine should complete");

        // With flat scores (0.5 before and after) strict improvement fails.
        // The engine should reject all mutation proposals and produce only the
        // seed candidate.
        assert_eq!(
            result.num_candidates(),
            1,
            "no candidate should be accepted when there is no strict improvement"
        );
        assert!(
            rejection_count.load(Ordering::SeqCst) > 0,
            "on_candidate_rejected should have fired at least once"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 13 — stop condition is respected after exactly N iterations
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_stop_condition_respected_mid_run() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct IterationCounter(Arc<AtomicUsize>);

        impl GEPACallback<usize> for IterationCounter {
            fn on_iteration_start(&self, _event: &IterationStartEvent<usize>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let iter_count = Arc::new(AtomicUsize::new(0));
        // MaxIterationsStopper(3): the engine should run exactly 3 iterations.
        let mut engine = make_engine(3);
        engine
            .callbacks
            .push(Box::new(IterationCounter(iter_count.clone())));

        engine.run().await.expect("engine should complete");

        assert_eq!(
            iter_count.load(Ordering::SeqCst),
            3,
            "engine should fire on_iteration_start exactly 3 times"
        );
    }

    // -------------------------------------------------------------------------
    // Gap 14 — build_valset_evaluation with objective_scores=Some
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_build_valset_evaluation_with_objectives() {
        let ids = vec![0usize, 1];
        let obj_scores: Vec<HashMap<String, f64>> = vec![
            [("precision".to_string(), 0.8_f64)].into_iter().collect(),
            [("precision".to_string(), 0.6_f64)].into_iter().collect(),
        ];
        let batch: EvaluationBatch<(), String> =
            EvaluationBatch::new(vec!["a".into(), "b".into()], vec![0.8, 0.6])
                .with_objective_scores(obj_scores);

        let eval = build_valset_evaluation(&ids, &batch).expect("valid batch");

        assert!(
            eval.valset_evaluation.objective_scores_by_val_id.is_some(),
            "objective_scores_by_val_id should be Some when batch contains objective_scores"
        );

        let obj_by_id = eval.valset_evaluation.objective_scores_by_val_id.unwrap();
        assert_eq!(obj_by_id.len(), 2, "should have one entry per val_id");

        let precision_0 = obj_by_id
            .get(&0usize)
            .and_then(|m| m.get("precision"))
            .copied()
            .expect("precision score for id=0 should be present");
        assert!(
            (precision_0 - 0.8).abs() < f64::EPSILON,
            "precision for id=0 should be 0.8"
        );

        let precision_1 = obj_by_id
            .get(&1usize)
            .and_then(|m| m.get("precision"))
            .copied()
            .expect("precision score for id=1 should be present");
        assert!(
            (precision_1 - 0.6).abs() < f64::EPSILON,
            "precision for id=1 should be 0.6"
        );
    }
}
