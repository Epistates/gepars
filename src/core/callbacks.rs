/// Callback trait and event system for GEPA optimisation instrumentation.
///
/// Callbacks are purely observational hooks — they receive a reference to the
/// current `GEPAState` for maximum flexibility but must not modify it.
///
/// Each callback method accepts a dedicated event struct containing all
/// relevant parameters.  This design allows future fields to be added without
/// breaking existing implementations.
///
/// Callbacks are registered by passing a `Vec<Box<dyn GEPACallback>>` to the
/// engine's `optimize` function.  Any type implementing one or more methods on
/// `GEPACallback` is a valid callback.
///
/// Mirrors the Python `gepa.core.callbacks` module.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::adapter::Candidate;
use crate::core::data_loader::DataId;
use crate::core::state::{GEPAState, ProgramIdx};

// =============================================================================
// Event structs
// =============================================================================

/// Fired once when optimisation begins, before any iterations.
#[derive(Debug, Clone)]
pub struct OptimizationStartEvent {
    pub seed_candidate: Candidate,
    pub trainset_size: usize,
    pub valset_size: usize,
    pub config: HashMap<String, serde_json::Value>,
}

/// Fired once when optimisation completes or is interrupted.
#[derive(Debug)]
pub struct OptimizationEndEvent<'a, Id: DataId> {
    pub best_candidate_idx: ProgramIdx,
    pub total_iterations: usize,
    pub total_metric_calls: usize,
    /// Reference to the terminal state (read-only).
    pub final_state: &'a GEPAState<Id>,
}

/// Fired at the beginning of each optimisation iteration.
#[derive(Debug)]
pub struct IterationStartEvent<'a, Id: DataId> {
    pub iteration: usize,
    /// Current state (read-only).
    pub state: &'a GEPAState<Id>,
    pub trainset_size: usize,
}

/// Fired at the end of each optimisation iteration.
#[derive(Debug)]
pub struct IterationEndEvent<'a, Id: DataId> {
    pub iteration: usize,
    /// Updated state (read-only).
    pub state: &'a GEPAState<Id>,
    pub proposal_accepted: bool,
}

/// Fired when a candidate from the Pareto front is selected for mutation.
#[derive(Debug, Clone)]
pub struct CandidateSelectedEvent {
    pub iteration: usize,
    pub candidate_idx: ProgramIdx,
    pub candidate: Candidate,
    pub score: f64,
}

/// Fired when a training mini-batch is drawn for the acceptance test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinibatchSampledEvent {
    pub iteration: usize,
    pub minibatch_ids: Vec<serde_json::Value>,
    pub trainset_size: usize,
}

/// Fired immediately before a batch evaluation begins.
#[derive(Debug, Clone)]
pub struct EvaluationStartEvent {
    pub iteration: usize,
    /// `None` for the seed candidate's initial evaluation.
    pub candidate_idx: Option<ProgramIdx>,
    pub batch_size: usize,
    pub capture_traces: bool,
    pub parent_ids: Vec<ProgramIdx>,
    pub is_seed_candidate: bool,
}

/// Fired immediately after a batch evaluation completes.
#[derive(Debug, Clone)]
pub struct EvaluationEndEvent {
    pub iteration: usize,
    pub candidate_idx: Option<ProgramIdx>,
    pub scores: Vec<f64>,
    pub has_trajectories: bool,
    pub parent_ids: Vec<ProgramIdx>,
    pub outputs: Vec<serde_json::Value>,
    pub objective_scores: Option<Vec<HashMap<String, f64>>>,
    pub is_seed_candidate: bool,
}

/// Fired when an evaluation is skipped or its results are discarded.
#[derive(Debug, Clone)]
pub struct EvaluationSkippedEvent {
    pub iteration: usize,
    pub candidate_idx: ProgramIdx,
    /// Human-readable explanation.
    pub reason: String,
    pub scores: Option<Vec<f64>>,
    pub is_seed_candidate: bool,
}

/// Fired after the reflective dataset is assembled for a mutation step.
#[derive(Debug, Clone)]
pub struct ReflectiveDatasetBuiltEvent {
    pub iteration: usize,
    pub candidate_idx: ProgramIdx,
    pub components: Vec<String>,
    pub dataset: HashMap<String, Vec<serde_json::Value>>,
}

/// Fired before the instruction proposer is invoked.
#[derive(Debug, Clone)]
pub struct ProposalStartEvent {
    pub iteration: usize,
    pub parent_candidate: Candidate,
    pub components: Vec<String>,
    pub reflective_dataset: HashMap<String, Vec<serde_json::Value>>,
}

/// Fired after the instruction proposer returns new component texts.
#[derive(Debug, Clone)]
pub struct ProposalEndEvent {
    pub iteration: usize,
    pub new_instructions: Candidate,
    pub prompts: HashMap<String, serde_json::Value>,
    pub raw_lm_outputs: HashMap<String, String>,
}

/// Fired when a proposed candidate passes the acceptance test.
#[derive(Debug, Clone)]
pub struct CandidateAcceptedEvent {
    pub iteration: usize,
    pub new_candidate_idx: ProgramIdx,
    pub new_score: f64,
    pub parent_ids: Vec<ProgramIdx>,
}

/// Fired when a proposed candidate fails the acceptance test.
#[derive(Debug, Clone)]
pub struct CandidateRejectedEvent {
    pub iteration: usize,
    pub old_score: f64,
    pub new_score: f64,
    pub reason: String,
}

/// Fired when a cross-candidate merge is attempted.
#[derive(Debug, Clone)]
pub struct MergeAttemptedEvent {
    pub iteration: usize,
    pub parent_ids: Vec<ProgramIdx>,
    pub merged_candidate: Candidate,
}

/// Fired when a merged candidate passes the acceptance test.
#[derive(Debug, Clone)]
pub struct MergeAcceptedEvent {
    pub iteration: usize,
    pub new_candidate_idx: ProgramIdx,
    pub parent_ids: Vec<ProgramIdx>,
}

/// Fired when a merged candidate fails the acceptance test.
#[derive(Debug, Clone)]
pub struct MergeRejectedEvent {
    pub iteration: usize,
    pub parent_ids: Vec<ProgramIdx>,
    pub reason: String,
}

/// Fired when the instance-level Pareto front changes.
#[derive(Debug, Clone)]
pub struct ParetoFrontUpdatedEvent {
    pub iteration: usize,
    /// Candidate indices now on the Pareto front.
    pub new_front: Vec<ProgramIdx>,
    /// Candidate indices displaced from the front by the update.
    pub displaced_candidates: Vec<ProgramIdx>,
}

/// Fired after a candidate's full validation-set evaluation completes.
#[derive(Debug, Clone)]
pub struct ValsetEvaluatedEvent {
    pub iteration: usize,
    pub candidate_idx: ProgramIdx,
    pub candidate: Candidate,
    pub scores_by_val_id: HashMap<serde_json::Value, f64>,
    pub average_score: f64,
    pub num_examples_evaluated: usize,
    pub total_valset_size: usize,
    pub parent_ids: Vec<ProgramIdx>,
    pub is_best_program: bool,
}

/// Fired after the state is successfully written to disk.
#[derive(Debug, Clone)]
pub struct StateSavedEvent {
    pub iteration: usize,
    pub run_dir: Option<String>,
}

/// Fired whenever the evaluation budget counter changes.
#[derive(Debug, Clone, Copy)]
pub struct BudgetUpdatedEvent {
    pub iteration: usize,
    pub metric_calls_used: usize,
    pub metric_calls_delta: usize,
    /// `None` if no budget cap is set.
    pub metric_calls_remaining: Option<usize>,
}

/// Fired when an exception occurs during an optimisation iteration.
#[derive(Debug)]
pub struct ErrorEvent {
    pub iteration: usize,
    pub error: String,
    /// Whether the engine will continue to the next iteration (vs. abort).
    pub will_continue: bool,
}

// =============================================================================
// GEPACallback trait
// =============================================================================

/// Trait for GEPA optimisation callbacks.
///
/// All methods have default no-op implementations — implement only the events
/// you care about.
///
/// ### Contract
/// - Callbacks are invoked synchronously by the engine.
/// - Callbacks must not modify the `state` they receive.
/// - Panics inside callback methods will propagate to the engine; catch errors
///   internally.
pub trait GEPACallback<Id: DataId>: Send + Sync {
    // ---- Optimization Lifecycle ------------------------------------------

    fn on_optimization_start(&self, _event: &OptimizationStartEvent) {}
    fn on_optimization_end(&self, _event: &OptimizationEndEvent<'_, Id>) {}

    // ---- Iteration Lifecycle ---------------------------------------------

    fn on_iteration_start(&self, _event: &IterationStartEvent<'_, Id>) {}
    fn on_iteration_end(&self, _event: &IterationEndEvent<'_, Id>) {}

    // ---- Candidate Selection and Sampling --------------------------------

    fn on_candidate_selected(&self, _event: &CandidateSelectedEvent) {}
    fn on_minibatch_sampled(&self, _event: &MinibatchSampledEvent) {}

    // ---- Evaluation Events ----------------------------------------------

    fn on_evaluation_start(&self, _event: &EvaluationStartEvent) {}
    fn on_evaluation_end(&self, _event: &EvaluationEndEvent) {}
    fn on_evaluation_skipped(&self, _event: &EvaluationSkippedEvent) {}
    fn on_valset_evaluated(&self, _event: &ValsetEvaluatedEvent) {}

    // ---- Reflection Events ----------------------------------------------

    fn on_reflective_dataset_built(&self, _event: &ReflectiveDatasetBuiltEvent) {}
    fn on_proposal_start(&self, _event: &ProposalStartEvent) {}
    fn on_proposal_end(&self, _event: &ProposalEndEvent) {}

    // ---- Acceptance/Rejection Events ------------------------------------

    fn on_candidate_accepted(&self, _event: &CandidateAcceptedEvent) {}
    fn on_candidate_rejected(&self, _event: &CandidateRejectedEvent) {}

    // ---- Merge Events ---------------------------------------------------

    fn on_merge_attempted(&self, _event: &MergeAttemptedEvent) {}
    fn on_merge_accepted(&self, _event: &MergeAcceptedEvent) {}
    fn on_merge_rejected(&self, _event: &MergeRejectedEvent) {}

    // ---- State Events ---------------------------------------------------

    fn on_pareto_front_updated(&self, _event: &ParetoFrontUpdatedEvent) {}
    fn on_state_saved(&self, _event: &StateSavedEvent) {}

    // ---- Budget Tracking ------------------------------------------------

    fn on_budget_updated(&self, _event: &BudgetUpdatedEvent) {}

    // ---- Error Handling -------------------------------------------------

    fn on_error(&self, _event: &ErrorEvent) {}
}

// =============================================================================
// notify_callbacks convenience function
// =============================================================================

/// Notify a slice of callback objects of an event by calling `method` on each.
///
/// Individual panics from callbacks propagate; callers should use
/// `catch_unwind` or document that callbacks must not panic.
///
/// This is a macro-based helper because Rust does not have Python-style
/// `getattr` dispatch; callers simply invoke the concrete method:
///
/// ```rust,ignore
/// notify_callbacks(&callbacks, |cb| cb.on_iteration_end(&event));
/// ```
pub fn notify_callbacks<Id: DataId, F>(callbacks: &[Box<dyn GEPACallback<Id>>], mut notify: F)
where
    F: FnMut(&dyn GEPACallback<Id>),
{
    for cb in callbacks {
        notify(cb.as_ref());
    }
}

// =============================================================================
// CompositeCallback
// =============================================================================

/// Multiplexes events to an ordered list of child callbacks.
///
/// Prefer this over passing multiple callbacks directly when you need
/// guaranteed delivery order or want to add/remove callbacks at runtime.
pub struct CompositeCallback<Id: DataId> {
    callbacks: Vec<Box<dyn GEPACallback<Id>>>,
}

impl<Id: DataId> CompositeCallback<Id> {
    /// Construct an empty composite.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Append `callback` to the list.
    pub fn add(&mut self, callback: Box<dyn GEPACallback<Id>>) {
        self.callbacks.push(callback);
    }

    /// Return a reference to the registered callbacks.
    pub fn callbacks(&self) -> &[Box<dyn GEPACallback<Id>>] {
        &self.callbacks
    }
}

impl<Id: DataId> Default for CompositeCallback<Id> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: DataId> GEPACallback<Id> for CompositeCallback<Id> {
    fn on_optimization_start(&self, event: &OptimizationStartEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_optimization_start(event));
    }

    fn on_optimization_end(&self, event: &OptimizationEndEvent<'_, Id>) {
        notify_callbacks(&self.callbacks, |cb| cb.on_optimization_end(event));
    }

    fn on_iteration_start(&self, event: &IterationStartEvent<'_, Id>) {
        notify_callbacks(&self.callbacks, |cb| cb.on_iteration_start(event));
    }

    fn on_iteration_end(&self, event: &IterationEndEvent<'_, Id>) {
        notify_callbacks(&self.callbacks, |cb| cb.on_iteration_end(event));
    }

    fn on_candidate_selected(&self, event: &CandidateSelectedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_candidate_selected(event));
    }

    fn on_minibatch_sampled(&self, event: &MinibatchSampledEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_minibatch_sampled(event));
    }

    fn on_evaluation_start(&self, event: &EvaluationStartEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_evaluation_start(event));
    }

    fn on_evaluation_end(&self, event: &EvaluationEndEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_evaluation_end(event));
    }

    fn on_evaluation_skipped(&self, event: &EvaluationSkippedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_evaluation_skipped(event));
    }

    fn on_valset_evaluated(&self, event: &ValsetEvaluatedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_valset_evaluated(event));
    }

    fn on_reflective_dataset_built(&self, event: &ReflectiveDatasetBuiltEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_reflective_dataset_built(event));
    }

    fn on_proposal_start(&self, event: &ProposalStartEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_proposal_start(event));
    }

    fn on_proposal_end(&self, event: &ProposalEndEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_proposal_end(event));
    }

    fn on_candidate_accepted(&self, event: &CandidateAcceptedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_candidate_accepted(event));
    }

    fn on_candidate_rejected(&self, event: &CandidateRejectedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_candidate_rejected(event));
    }

    fn on_merge_attempted(&self, event: &MergeAttemptedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_merge_attempted(event));
    }

    fn on_merge_accepted(&self, event: &MergeAcceptedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_merge_accepted(event));
    }

    fn on_merge_rejected(&self, event: &MergeRejectedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_merge_rejected(event));
    }

    fn on_pareto_front_updated(&self, event: &ParetoFrontUpdatedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_pareto_front_updated(event));
    }

    fn on_state_saved(&self, event: &StateSavedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_state_saved(event));
    }

    fn on_budget_updated(&self, event: &BudgetUpdatedEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_budget_updated(event));
    }

    fn on_error(&self, event: &ErrorEvent) {
        notify_callbacks(&self.callbacks, |cb| cb.on_error(event));
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // A simple recording callback for tests.
    struct RecordingCallback {
        accepted_count: Arc<Mutex<usize>>,
        rejected_count: Arc<Mutex<usize>>,
        budget_events: Arc<Mutex<Vec<BudgetUpdatedEvent>>>,
    }

    impl RecordingCallback {
        fn new() -> (
            Self,
            Arc<Mutex<usize>>,
            Arc<Mutex<usize>>,
            Arc<Mutex<Vec<BudgetUpdatedEvent>>>,
        ) {
            let accepted = Arc::new(Mutex::new(0usize));
            let rejected = Arc::new(Mutex::new(0usize));
            let budget = Arc::new(Mutex::new(Vec::new()));
            let cb = Self {
                accepted_count: accepted.clone(),
                rejected_count: rejected.clone(),
                budget_events: budget.clone(),
            };
            (cb, accepted, rejected, budget)
        }
    }

    impl GEPACallback<usize> for RecordingCallback {
        fn on_candidate_accepted(&self, _event: &CandidateAcceptedEvent) {
            *self.accepted_count.lock().unwrap() += 1;
        }

        fn on_candidate_rejected(&self, _event: &CandidateRejectedEvent) {
            *self.rejected_count.lock().unwrap() += 1;
        }

        fn on_budget_updated(&self, event: &BudgetUpdatedEvent) {
            self.budget_events.lock().unwrap().push(*event);
        }
    }

    #[test]
    fn composite_routes_events_to_all_children() {
        let (cb1, accepted1, _, _) = RecordingCallback::new();
        let (cb2, accepted2, _, _) = RecordingCallback::new();

        let mut composite: CompositeCallback<usize> = CompositeCallback::new();
        composite.add(Box::new(cb1));
        composite.add(Box::new(cb2));

        let event = CandidateAcceptedEvent {
            iteration: 0,
            new_candidate_idx: 1,
            new_score: 0.9,
            parent_ids: vec![0],
        };
        composite.on_candidate_accepted(&event);

        assert_eq!(*accepted1.lock().unwrap(), 1);
        assert_eq!(*accepted2.lock().unwrap(), 1);
    }

    #[test]
    fn notify_callbacks_reaches_each_callback() {
        let (cb1, _, rejected1, _) = RecordingCallback::new();
        let (cb2, _, rejected2, _) = RecordingCallback::new();

        let callbacks: Vec<Box<dyn GEPACallback<usize>>> = vec![Box::new(cb1), Box::new(cb2)];

        let event = CandidateRejectedEvent {
            iteration: 1,
            old_score: 0.5,
            new_score: 0.4,
            reason: "score decreased".into(),
        };
        notify_callbacks(&callbacks, |cb| cb.on_candidate_rejected(&event));

        assert_eq!(*rejected1.lock().unwrap(), 1);
        assert_eq!(*rejected2.lock().unwrap(), 1);
    }

    #[test]
    fn budget_event_fields_are_preserved() {
        let (cb, _, _, budget_events) = RecordingCallback::new();

        let callbacks: Vec<Box<dyn GEPACallback<usize>>> = vec![Box::new(cb)];
        let event = BudgetUpdatedEvent {
            iteration: 3,
            metric_calls_used: 50,
            metric_calls_delta: 5,
            metric_calls_remaining: Some(150),
        };
        notify_callbacks(&callbacks, |c| c.on_budget_updated(&event));

        let events = budget_events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].metric_calls_used, 50);
        assert_eq!(events[0].metric_calls_remaining, Some(150));
    }

    #[test]
    fn default_no_op_methods_do_not_panic() {
        struct NoOpCallback;
        impl GEPACallback<usize> for NoOpCallback {}

        let cb = NoOpCallback;
        // Call every default method — they should all be silent no-ops.
        cb.on_optimization_start(&OptimizationStartEvent {
            seed_candidate: Candidate::new(),
            trainset_size: 10,
            valset_size: 5,
            config: HashMap::new(),
        });
        cb.on_candidate_selected(&CandidateSelectedEvent {
            iteration: 0,
            candidate_idx: 0,
            candidate: Candidate::new(),
            score: 0.5,
        });
    }
}
