/// `GEPAEngine` — Algorithm 1 of the GEPA paper.
///
/// The engine owns the full optimisation loop:
/// 1. Evaluate the seed candidate on the full validation set and build the
///    initial [`GEPAState`].
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
use crate::core::state::{EvaluationCache, FrontierType, GEPAState, ValsetEvaluation};
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
/// - `RO`   — raw rollout output type (opaque to the engine).
///
/// ### Ownership model
/// The engine owns all strategies and proposers.  All shared resources
/// (adapter, data loaders) are wrapped in `Arc`.
pub struct GEPAEngine<Id, Item, T, RO>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + 'static,
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
    RO: Send + Sync + 'static,
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
        // ── Step 1: collect validation IDs and items ─────────────────────────
        let all_val_ids = self.valset.all_ids();
        let all_val_items = self.valset.fetch(&all_val_ids)?;

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

        // ── Step 2: evaluate the seed on the full valset ─────────────────────
        info!(
            valset_size = all_val_ids.len(),
            "Evaluating seed candidate on full validation set"
        );

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_evaluation_start(&EvaluationStartEvent {
                iteration: 0,
                candidate_idx: None,
                batch_size: all_val_items.len(),
                capture_traces: false,
                parent_ids: vec![],
                is_seed_candidate: true,
            });
        });

        let seed_eval = self
            .adapter
            .evaluate(&all_val_items, &self.seed_candidate, false)
            .await?;

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_evaluation_end(&EvaluationEndEvent {
                iteration: 0,
                candidate_idx: None,
                scores: seed_eval.scores.clone(),
                has_trajectories: seed_eval.trajectories.is_some(),
                parent_ids: vec![],
                outputs: seed_eval
                    .outputs
                    .iter()
                    .map(|_| serde_json::Value::Null)
                    .collect(),
                objective_scores: seed_eval.objective_scores.clone(),
                is_seed_candidate: true,
            });
        });

        let seed_valset_eval = build_valset_evaluation(&all_val_ids, &seed_eval);

        // ── Step 3: build or resume the state (Gap 3) ────────────────────────
        let evaluation_cache = if self.cache_evaluation {
            Some(EvaluationCache::new())
        } else {
            None
        };

        let mut state = if let Some(ref run_dir) = self.run_dir {
            let state_path = Path::new(run_dir).join("gepa_state.json");
            if state_path.exists() {
                info!(path = %state_path.display(), "Resuming from saved state");
                let json = std::fs::read_to_string(&state_path)
                    .map_err(|e| GEPAError::Config(format!("Failed to read state file: {e}")))?;
                let mut loaded = GEPAState::<Id>::from_json(&json)?;
                // Sync cache with current run setting.
                if evaluation_cache.is_none() {
                    loaded.evaluation_cache = None;
                } else if loaded.evaluation_cache.is_none() {
                    loaded.evaluation_cache = evaluation_cache;
                }
                loaded
            } else {
                GEPAState::new_with_options(
                    self.seed_candidate.clone(),
                    seed_valset_eval,
                    self.frontier_type,
                    evaluation_cache,
                    self.track_best_outputs,
                )?
            }
        } else {
            GEPAState::new_with_options(
                self.seed_candidate.clone(),
                seed_valset_eval,
                self.frontier_type,
                evaluation_cache,
                self.track_best_outputs,
            )?
        };

        // Seed counted as one full-valset evaluation (only when initialising fresh).
        if state.num_full_ds_evals == 0 {
            state.num_full_ds_evals = 1;
            state.total_num_evals = all_val_ids.len();
        }

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_budget_updated(&BudgetUpdatedEvent {
                iteration: 0,
                metric_calls_used: state.total_num_evals,
                metric_calls_delta: all_val_ids.len(),
                metric_calls_remaining: None,
            });
        });

        let (seed_score, _) = state.get_program_average_val_subset(0);
        info!(seed_score, "Seed candidate evaluated");

        // Log seed metrics to tracker (Gap 9).
        self.tracker.log_metric("seed_score", seed_score, 0);

        // ── Step 4: main optimisation loop ───────────────────────────────────
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
                self.try_merge_step(&mut state, &all_val_ids, &all_val_items)
                    .await?
            } else {
                // ── Reflective mutation (Algorithm 3) ────────────────────────
                self.try_mutation_step(&mut state, &all_val_ids, &all_val_items)
                    .await?
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
        all_val_items: &[Item],
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
            .full_valset_eval_and_accept(
                state,
                merged_candidate,
                parent_ids.clone(),
                all_val_ids,
                all_val_items,
            )
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
        all_val_items: &[Item],
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
            .full_valset_eval_and_accept(
                state,
                candidate,
                parent_ids.clone(),
                all_val_ids,
                all_val_items,
            )
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
        all_val_items: &[Item],
    ) -> Result<(usize, usize)> {
        let evals_delta = all_val_items.len();

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_evaluation_start(&EvaluationStartEvent {
                iteration: state.i,
                candidate_idx: None, // not yet assigned
                batch_size: evals_delta,
                capture_traces: false,
                parent_ids: parent_ids.clone(),
                is_seed_candidate: false,
            });
        });

        let eval = self
            .adapter
            .evaluate(all_val_items, &candidate, false)
            .await?;

        notify_callbacks(&self.callbacks, |cb| {
            cb.on_evaluation_end(&EvaluationEndEvent {
                iteration: state.i,
                candidate_idx: None,
                scores: eval.scores.clone(),
                has_trajectories: eval.trajectories.is_some(),
                parent_ids: parent_ids.clone(),
                outputs: eval
                    .outputs
                    .iter()
                    .map(|_| serde_json::Value::Null)
                    .collect(),
                objective_scores: eval.objective_scores.clone(),
                is_seed_candidate: false,
            });
        });

        let valset_eval = build_valset_evaluation(all_val_ids, &eval);
        state.increment_evals(evals_delta);
        state.num_full_ds_evals += 1;

        // Note: `total_num_evals` is already updated by `increment_evals`; use it
        // as the discovery-count for the new program.
        let discovery_count = state.total_num_evals;
        let new_idx = state.update_state_with_new_program(
            parent_ids,
            candidate,
            valset_eval,
            discovery_count,
        )?;

        let avg_score = state.get_program_average_val_subset(new_idx).0;

        let scores_by_val_id: HashMap<serde_json::Value, f64> = all_val_ids
            .iter()
            .zip(eval.scores.iter())
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
}

// ---------------------------------------------------------------------------
// Helper: build ValsetEvaluation from EvaluationBatch
// ---------------------------------------------------------------------------

/// Build a [`ValsetEvaluation`] from parallel ID and batch-result slices.
fn build_valset_evaluation<Id: DataId, T: Send, RO: Send>(
    ids: &[Id],
    batch: &EvaluationBatch<T, RO>,
) -> ValsetEvaluation<Id> {
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

    // ValsetEvaluation tracks outputs as JSON but the engine only uses scores.
    let outputs_by_val_id: std::collections::HashMap<Id, serde_json::Value> = ids
        .iter()
        .cloned()
        .map(|id| (id, serde_json::Value::Null))
        .collect();

    ValsetEvaluation {
        outputs_by_val_id,
        scores_by_val_id,
        objective_scores_by_val_id,
    }
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

        let eval = build_valset_evaluation(&ids, &batch);
        assert_eq!(eval.scores_by_val_id.get(&0), Some(&0.3));
        assert_eq!(eval.scores_by_val_id.get(&1), Some(&0.6));
        assert_eq!(eval.scores_by_val_id.get(&2), Some(&0.9));
        assert!(eval.objective_scores_by_val_id.is_none());
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

        let eval = build_valset_evaluation(&ids, &batch);

        assert!(
            eval.objective_scores_by_val_id.is_some(),
            "objective_scores_by_val_id should be Some when batch contains objective_scores"
        );

        let obj_by_id = eval.objective_scores_by_val_id.unwrap();
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
