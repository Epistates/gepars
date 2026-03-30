use std::sync::Arc;
/// Stop conditions for the GEPA optimization loop.
///
/// A `StopCondition` is a stateless predicate that inspects the current
/// `GEPAState` and returns `true` when the engine should stop iterating.
///
/// The included concrete implementations cover the most common scenarios:
/// - [`MaxMetricCallsStopper`]   — stop after a fixed number of `evaluate()` calls.
/// - [`TimeoutStopper`]          — stop after a wall-clock duration.
/// - [`CompositeStopper`]        — combine any/all predicates.
/// - [`FileStopper`]             — stop when a sentinel file appears on disk.
/// - [`SignalStopper`]           — stop after a SIGTERM / SIGINT has been received.
/// - [`NoImprovementStopper`]    — stop when the best score hasn't improved for N iterations.
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use crate::core::data_loader::DataId;
use crate::core::state::GEPAState;

// ---------------------------------------------------------------------------
// StopCondition trait
// ---------------------------------------------------------------------------

/// Predicate inspected at the start of each optimisation iteration.
///
/// Returns `true` when optimisation should stop.
pub trait StopCondition<Id: DataId>: Send + Sync {
    /// Evaluate the stop condition against the current state.
    ///
    /// Called at the beginning of each iteration, before any work is done.
    fn should_stop(&self, state: &GEPAState<Id>) -> bool;

    /// Human-readable description of this condition (used in log messages).
    fn description(&self) -> String;
}

// ---------------------------------------------------------------------------
// MaxMetricCallsStopper
// ---------------------------------------------------------------------------

/// Stop after a fixed total number of adapter `evaluate()` calls.
///
/// The check uses `state.total_num_evals` which the engine increments after
/// every evaluation.
#[derive(Debug, Clone, Copy)]
pub struct MaxMetricCallsStopper {
    /// Maximum number of adapter `evaluate()` calls allowed.
    pub max_calls: usize,
}

impl MaxMetricCallsStopper {
    /// Construct with a given budget.
    pub fn new(max_calls: usize) -> Self {
        Self { max_calls }
    }
}

impl<Id: DataId> StopCondition<Id> for MaxMetricCallsStopper {
    fn should_stop(&self, state: &GEPAState<Id>) -> bool {
        state.total_num_evals >= self.max_calls
    }

    fn description(&self) -> String {
        format!("MaxMetricCalls({})", self.max_calls)
    }
}

// ---------------------------------------------------------------------------
// MaxIterationsStopper
// ---------------------------------------------------------------------------

/// Stop after a fixed number of optimisation iterations.
///
/// The check uses `state.i` (the iteration counter).
#[derive(Debug, Clone, Copy)]
pub struct MaxIterationsStopper {
    /// Maximum number of iterations to run.
    pub max_iterations: usize,
}

impl MaxIterationsStopper {
    /// Construct with a given iteration budget.
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl<Id: DataId> StopCondition<Id> for MaxIterationsStopper {
    fn should_stop(&self, state: &GEPAState<Id>) -> bool {
        // `state.i` starts at `BEFORE_FIRST_ITERATION` (usize::MAX).  After
        // the engine increments it at the start of iteration 0, `state.i`
        // becomes 0.  We stop when `state.i >= max_iterations`.
        // `usize::MAX` is never >= a sane `max_iterations` budget.
        state.i != crate::core::state::BEFORE_FIRST_ITERATION && state.i >= self.max_iterations
    }

    fn description(&self) -> String {
        format!("MaxIterations({})", self.max_iterations)
    }
}

// ---------------------------------------------------------------------------
// TimeoutStopper
// ---------------------------------------------------------------------------

/// Stop after a wall-clock [`Duration`] has elapsed since construction.
///
/// The start time is captured when `TimeoutStopper::new` is called, not when
/// the engine starts.  Construct this immediately before calling `optimize`.
#[derive(Debug)]
pub struct TimeoutStopper {
    start: Instant,
    duration: Duration,
}

impl TimeoutStopper {
    /// Construct with the given timeout duration.  The clock starts now.
    pub fn new(duration: Duration) -> Self {
        Self {
            start: Instant::now(),
            duration,
        }
    }

    /// Time remaining before the timeout fires.  Returns `None` when already
    /// expired.
    pub fn remaining(&self) -> Option<Duration> {
        self.duration.checked_sub(self.start.elapsed())
    }
}

impl<Id: DataId> StopCondition<Id> for TimeoutStopper {
    fn should_stop(&self, _state: &GEPAState<Id>) -> bool {
        self.start.elapsed() >= self.duration
    }

    fn description(&self) -> String {
        format!("Timeout({:.1}s)", self.duration.as_secs_f64())
    }
}

// ---------------------------------------------------------------------------
// CompositeStopper
// ---------------------------------------------------------------------------

/// Combination mode for [`CompositeStopper`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeMode {
    /// Stop when **any** child condition returns `true` (logical OR).
    Any,
    /// Stop only when **all** child conditions return `true` (logical AND).
    All,
}

/// Combines multiple stop conditions with `Any` or `All` semantics.
pub struct CompositeStopper<Id: DataId> {
    conditions: Vec<Box<dyn StopCondition<Id>>>,
    mode: CompositeMode,
}

impl<Id: DataId> CompositeStopper<Id> {
    /// Construct an empty composite.  Add conditions with [`CompositeStopper::push_condition`].
    pub fn new(mode: CompositeMode) -> Self {
        Self {
            conditions: Vec::new(),
            mode,
        }
    }

    /// Convenience constructor: stop when any condition fires.
    pub fn any() -> Self {
        Self::new(CompositeMode::Any)
    }

    /// Convenience constructor: stop when all conditions fire.
    pub fn all() -> Self {
        Self::new(CompositeMode::All)
    }

    /// Append a child stop condition.
    pub fn push_condition(mut self, condition: impl StopCondition<Id> + 'static) -> Self {
        self.conditions.push(Box::new(condition));
        self
    }
}

impl<Id: DataId> StopCondition<Id> for CompositeStopper<Id> {
    fn should_stop(&self, state: &GEPAState<Id>) -> bool {
        match self.mode {
            CompositeMode::Any => self.conditions.iter().any(|c| c.should_stop(state)),
            CompositeMode::All => self.conditions.iter().all(|c| c.should_stop(state)),
        }
    }

    fn description(&self) -> String {
        let inner: Vec<String> = self.conditions.iter().map(|c| c.description()).collect();
        match self.mode {
            CompositeMode::Any => format!("Any({})", inner.join(", ")),
            CompositeMode::All => format!("All({})", inner.join(", ")),
        }
    }
}

// ---------------------------------------------------------------------------
// FileStopper (Gap 8)
// ---------------------------------------------------------------------------

/// Stop the optimisation loop when a sentinel file appears on disk.
///
/// This is useful for graceful shutdown via an external process or script:
/// touching the sentinel file causes the engine to stop at the next iteration
/// boundary, after which the current state is serialised normally.
///
/// # Example
/// ```no_run
/// use gepa::utils::stop_condition::FileStopper;
/// let stopper = FileStopper::new("/tmp/stop_gepa");
/// ```
pub struct FileStopper {
    /// Path of the sentinel file to watch.
    pub path: std::path::PathBuf,
}

impl FileStopper {
    /// Construct a stopper that watches `path`.
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl<Id: DataId> StopCondition<Id> for FileStopper {
    fn should_stop(&self, _state: &GEPAState<Id>) -> bool {
        self.path.exists()
    }

    fn description(&self) -> String {
        format!("FileStopper({})", self.path.display())
    }
}

// ---------------------------------------------------------------------------
// SignalStopper (Gap 8)
// ---------------------------------------------------------------------------

/// Stop when an OS signal (SIGTERM / SIGINT) has been received.
///
/// Callers create a `SignalStopper` and register the shared `Arc<AtomicBool>`
/// with their signal handler (e.g., via the `ctrlc` crate).  When the flag is
/// set to `true`, the stopper fires.
///
/// ```rust
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicBool, Ordering};
/// use gepa::utils::stop_condition::SignalStopper;
///
/// let flag = Arc::new(AtomicBool::new(false));
/// let stopper = SignalStopper::new(Arc::clone(&flag));
///
/// // In a signal handler:
/// // flag.store(true, Ordering::SeqCst);
/// ```
pub struct SignalStopper {
    /// Shared flag: set to `true` to trigger the stop.
    pub flag: Arc<AtomicBool>,
}

impl SignalStopper {
    /// Construct from an existing `Arc<AtomicBool>`.
    pub fn new(flag: Arc<AtomicBool>) -> Self {
        Self { flag }
    }

    /// Create a new `SignalStopper` with its own internal flag.
    ///
    /// Returns both the stopper and the flag so the caller can set it from a
    /// signal handler or other thread.
    pub fn with_flag() -> (Self, Arc<AtomicBool>) {
        let flag = Arc::new(AtomicBool::new(false));
        let stopper = Self::new(Arc::clone(&flag));
        (stopper, flag)
    }

    /// Set the stop flag programmatically (e.g., from a signal handler thread).
    pub fn signal(&self) {
        self.flag.store(true, AtomicOrdering::SeqCst);
    }

    /// Query whether the stop flag has been set.
    pub fn is_set(&self) -> bool {
        self.flag.load(AtomicOrdering::SeqCst)
    }
}

impl<Id: DataId> StopCondition<Id> for SignalStopper {
    fn should_stop(&self, _state: &GEPAState<Id>) -> bool {
        self.flag.load(AtomicOrdering::SeqCst)
    }

    fn description(&self) -> String {
        "SignalStopper".to_owned()
    }
}

// ---------------------------------------------------------------------------
// NoImprovementStopper (Gap 8)
// ---------------------------------------------------------------------------

/// Stop when the best validation score has not improved for `patience` iterations.
///
/// "Improvement" means the aggregate best score across all candidates increased
/// by more than `min_delta` relative to the previous best.  If scores remain
/// flat (or worsen) for `patience` consecutive iterations, the engine stops.
///
/// The stopper uses `std::sync::atomic` types for interior mutability so it
/// is both `Send` and `Sync` without any `unsafe` code.
///
/// # Example
/// ```rust
/// use gepa::utils::stop_condition::NoImprovementStopper;
/// let stopper = NoImprovementStopper::new(5, 1e-4);
/// ```
pub struct NoImprovementStopper {
    /// Number of iterations without improvement before stopping.
    pub patience: usize,
    /// Minimum absolute improvement required to reset the counter.
    pub min_delta: f64,
    /// Best score seen so far, stored as IEEE-754 bits.
    best_score_bits: std::sync::atomic::AtomicU64,
    /// Number of consecutive iterations without sufficient improvement.
    no_improve_count: std::sync::atomic::AtomicUsize,
}

impl NoImprovementStopper {
    /// Construct with a patience window and minimum delta.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_score_bits: std::sync::atomic::AtomicU64::new(f64::NEG_INFINITY.to_bits()),
            no_improve_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Current best score recorded by the stopper.
    pub fn current_best(&self) -> f64 {
        f64::from_bits(
            self.best_score_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Consecutive-no-improvement counter.
    pub fn stagnation_count(&self) -> usize {
        self.no_improve_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<Id: DataId> StopCondition<Id> for NoImprovementStopper {
    fn should_stop(&self, state: &GEPAState<Id>) -> bool {
        // Compute the current best aggregate score across all candidates.
        let current_best = state
            .program_full_scores_val_set()
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);

        let prev_best = f64::from_bits(
            self.best_score_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        );

        if current_best > prev_best + self.min_delta {
            // Improvement detected — reset the counter.
            self.best_score_bits
                .store(current_best.to_bits(), std::sync::atomic::Ordering::Relaxed);
            self.no_improve_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
            false
        } else {
            // No improvement this iteration.
            let count = self
                .no_improve_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            count >= self.patience
        }
    }

    fn description(&self) -> String {
        format!(
            "NoImprovementStopper(patience={}, min_delta={})",
            self.patience, self.min_delta
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adapter::Candidate;
    use crate::core::state::{BEFORE_FIRST_ITERATION, FrontierType, GEPAState, ValsetEvaluation};

    fn make_state_with_evals(num_evals: usize) -> GEPAState<usize> {
        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "test".into());
        let outputs = (0..2).map(|i| serde_json::json!(i)).collect();
        let eval = ValsetEvaluation::from_vecs(vec![0usize, 1], outputs, vec![0.5, 0.8], None);
        let mut state = GEPAState::new(seed, eval, FrontierType::Instance, None).unwrap();
        state.total_num_evals = num_evals;
        state
    }

    #[test]
    fn max_metric_calls_fires_when_budget_exhausted() {
        let stopper = MaxMetricCallsStopper::new(100);
        let state_under = make_state_with_evals(99);
        let state_at = make_state_with_evals(100);
        let state_over = make_state_with_evals(101);

        assert!(!stopper.should_stop(&state_under));
        assert!(stopper.should_stop(&state_at));
        assert!(stopper.should_stop(&state_over));
    }

    #[test]
    fn max_iterations_stops_after_n_iterations() {
        let stopper = MaxIterationsStopper::new(5);

        let mut state = make_state_with_evals(0);
        // Before any iteration: should not stop.
        assert_eq!(state.i, BEFORE_FIRST_ITERATION);
        assert!(!stopper.should_stop(&state));

        // Simulate the engine incrementing `i`.
        state.i = 5; // >= max_iterations → stop
        assert!(stopper.should_stop(&state));

        state.i = 4; // < max_iterations → continue
        assert!(!stopper.should_stop(&state));
    }

    #[test]
    fn timeout_stopper_does_not_fire_immediately() {
        let stopper = TimeoutStopper::new(Duration::from_secs(60));
        let state = make_state_with_evals(0);
        assert!(!stopper.should_stop(&state));
        assert!(stopper.remaining().is_some());
    }

    #[test]
    fn timeout_stopper_fires_after_expiry() {
        // Create with a near-zero timeout to simulate expiry.
        let stopper = TimeoutStopper::new(Duration::from_nanos(1));
        // Allow the timeout to expire.
        std::thread::sleep(Duration::from_millis(1));
        let state = make_state_with_evals(0);
        assert!(stopper.should_stop(&state));
    }

    #[test]
    fn composite_any_stops_when_one_fires() {
        let stopper: CompositeStopper<usize> = CompositeStopper::any()
            .push_condition(MaxMetricCallsStopper::new(1000))
            .push_condition(MaxMetricCallsStopper::new(10));

        let state_under = make_state_with_evals(9);
        let state_at = make_state_with_evals(10);

        assert!(!stopper.should_stop(&state_under));
        assert!(stopper.should_stop(&state_at));
    }

    #[test]
    fn composite_all_requires_all_to_fire() {
        let stopper: CompositeStopper<usize> = CompositeStopper::all()
            .push_condition(MaxMetricCallsStopper::new(10))
            .push_condition(MaxMetricCallsStopper::new(20));

        let state_10 = make_state_with_evals(10);
        let state_20 = make_state_with_evals(20);

        // Only first fires at 10.
        assert!(!stopper.should_stop(&state_10));
        // Both fire at 20.
        assert!(stopper.should_stop(&state_20));
    }

    #[test]
    fn descriptions_are_human_readable() {
        let s = MaxMetricCallsStopper::new(200);
        assert!(StopCondition::<usize>::description(&s).contains("200"));

        let t = TimeoutStopper::new(Duration::from_secs(30));
        assert!(StopCondition::<usize>::description(&t).contains("30"));

        let c: CompositeStopper<usize> =
            CompositeStopper::any().push_condition(MaxMetricCallsStopper::new(100));
        assert!(c.description().contains("Any"));
        assert!(c.description().contains("100"));
    }

    // ------------------------------------------------------------------
    // Gap 54: remaining() returns None after timeout expiry
    // ------------------------------------------------------------------

    #[test]
    fn test_timeout_remaining_none_after_expiry() {
        let stopper = TimeoutStopper::new(Duration::from_nanos(1));
        // Allow the 1ns timeout to elapse.
        std::thread::sleep(Duration::from_millis(1));
        assert!(
            stopper.remaining().is_none(),
            "remaining() should return None after the timeout has expired"
        );
    }

    // ------------------------------------------------------------------
    // Gap 55: empty any() composite returns false (no conditions → never stop)
    // ------------------------------------------------------------------

    #[test]
    fn test_composite_empty_any_returns_false() {
        let stopper: CompositeStopper<usize> = CompositeStopper::any();
        let state = make_state_with_evals(9999);
        assert!(
            !stopper.should_stop(&state),
            "empty any() composite must return false (no condition can fire)"
        );
    }

    // ------------------------------------------------------------------
    // Gap 56: empty all() composite returns true (vacuously all conditions met)
    // ------------------------------------------------------------------

    #[test]
    fn test_composite_empty_all_returns_true() {
        let stopper: CompositeStopper<usize> = CompositeStopper::all();
        let state = make_state_with_evals(0);
        assert!(
            stopper.should_stop(&state),
            "empty all() composite must return true (vacuous conjunction)"
        );
    }

    // ------------------------------------------------------------------
    // Gap 8: FileStopper tests
    // ------------------------------------------------------------------

    #[test]
    fn file_stopper_does_not_fire_without_file() {
        let stopper = FileStopper::new("/tmp/gepa_test_sentinel_NONEXISTENT_XYZ");
        let state = make_state_with_evals(0);
        assert!(
            !stopper.should_stop(&state),
            "FileStopper should not fire when the file does not exist"
        );
    }

    #[test]
    fn file_stopper_fires_when_file_exists() {
        let tmp = std::env::temp_dir().join("gepa_test_file_stopper_sentinel");
        std::fs::write(&tmp, "stop").expect("write sentinel");
        let stopper = FileStopper::new(&tmp);
        let state = make_state_with_evals(0);
        let fires = stopper.should_stop(&state);
        std::fs::remove_file(&tmp).ok();
        assert!(
            fires,
            "FileStopper should fire when the sentinel file exists"
        );
    }

    #[test]
    fn file_stopper_description_contains_path() {
        let stopper = FileStopper::new("/tmp/sentinel");
        let desc = StopCondition::<usize>::description(&stopper);
        assert!(
            desc.contains("sentinel"),
            "description should contain the path"
        );
    }

    // ------------------------------------------------------------------
    // Gap 8: SignalStopper tests
    // ------------------------------------------------------------------

    #[test]
    fn signal_stopper_does_not_fire_before_signal() {
        let (stopper, _flag) = SignalStopper::with_flag();
        let state = make_state_with_evals(0);
        assert!(
            !stopper.should_stop(&state),
            "SignalStopper should not fire before the flag is set"
        );
    }

    #[test]
    fn signal_stopper_fires_after_signal() {
        let (stopper, flag) = SignalStopper::with_flag();
        let state = make_state_with_evals(0);
        flag.store(true, AtomicOrdering::SeqCst);
        assert!(
            stopper.should_stop(&state),
            "SignalStopper should fire after the flag is set"
        );
    }

    #[test]
    fn signal_stopper_signal_method_triggers_stop() {
        let (stopper, _flag) = SignalStopper::with_flag();
        assert!(!stopper.is_set());
        stopper.signal();
        assert!(stopper.is_set());
        let state = make_state_with_evals(0);
        assert!(stopper.should_stop(&state));
    }

    // ------------------------------------------------------------------
    // Gap 8: NoImprovementStopper tests
    // ------------------------------------------------------------------

    #[test]
    fn no_improvement_stopper_does_not_fire_on_first_call() {
        // On first call the internal best is -inf, so any real score is an
        // improvement; the counter resets and the stopper should not fire.
        let stopper = NoImprovementStopper::new(3, 0.0);
        let state = make_state_with_evals(0); // seed score avg ~0.65
        assert!(
            !stopper.should_stop(&state),
            "NoImprovementStopper must not fire on the first call"
        );
        assert_eq!(stopper.stagnation_count(), 0);
    }

    #[test]
    fn no_improvement_stopper_fires_after_patience_exceeded() {
        let stopper = NoImprovementStopper::new(3, 0.0);
        let state = make_state_with_evals(0);

        // First call: improvement detected (−inf → actual score).
        assert!(!stopper.should_stop(&state));
        // Next 3 calls: no improvement (same score).
        assert!(!stopper.should_stop(&state)); // count=1
        assert!(!stopper.should_stop(&state)); // count=2
        assert!(stopper.should_stop(&state)); // count=3 >= patience → fire
    }

    #[test]
    fn no_improvement_stopper_resets_on_improvement() {
        let stopper = NoImprovementStopper::new(2, 0.0);
        let state_low = make_state_with_evals(0); // scores [0.5, 0.8] → avg ~0.65

        // Advance stagnation counter once.
        assert!(!stopper.should_stop(&state_low)); // first call: improvement → count=0
        assert!(!stopper.should_stop(&state_low)); // no improvement: count=1

        // Now inject a better state by adding a second candidate.
        let mut state_high = make_state_with_evals(0);
        let better_eval = crate::core::state::ValsetEvaluation::from_vecs(
            vec![0usize, 1],
            vec![serde_json::json!("a"), serde_json::json!("b")],
            vec![0.9, 1.0],
            None,
        );
        let mut better_cand = Candidate::new();
        better_cand.insert("instructions".into(), "better".into());
        state_high
            .update_state_with_new_program(vec![0], better_cand, better_eval, 2)
            .expect("update");

        // Improvement detected → counter resets.
        assert!(!stopper.should_stop(&state_high));
        assert_eq!(stopper.stagnation_count(), 0);
    }
}
