/// Utility modules for the GEPA optimisation engine.
pub mod pareto;
pub mod stop_condition;

// ---------------------------------------------------------------------------
// Flat re-exports
// ---------------------------------------------------------------------------

// Pareto utilities (Algorithm 2)
pub use pareto::{
    find_dominator_programs, idxmax, is_dominated, remove_dominated_programs,
    select_program_candidate_from_pareto_front,
};

// Stop conditions
pub use stop_condition::{
    CompositeMode, CompositeStopper, MaxIterationsStopper, MaxMetricCallsStopper, StopCondition,
    TimeoutStopper,
};
