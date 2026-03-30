/// Core types and the main optimisation engine for GEPA.
///
/// This module re-exports all public types that engine code and user-facing
/// APIs consume.  Organise imports from here rather than from the submodule
/// paths directly.
pub mod adapter;
pub mod callbacks;
pub mod data_loader;
pub mod engine;
pub mod result;
pub(crate) mod serde_helpers;
pub mod state;

// ---------------------------------------------------------------------------
// Flat re-exports for ergonomic use
// ---------------------------------------------------------------------------

// Adapter
pub use adapter::{Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset};

// DataLoader
pub use data_loader::{DataId, DataLoader, MutableDataLoader, VecLoader, ensure_loader};

// State
pub use state::{
    BEFORE_FIRST_ITERATION, CachedEvaluation, EvaluationCache, FrontierKey, FrontierType,
    GEPAState, ObjectiveScores, ProgramIdx, ValsetEvaluation, candidate_hash,
};

// Result
pub use result::GEPAResult;

// Engine
pub use engine::GEPAEngine;

// Callbacks
pub use callbacks::{
    BudgetUpdatedEvent, CandidateAcceptedEvent, CandidateRejectedEvent, CandidateSelectedEvent,
    CompositeCallback, ErrorEvent, EvaluationEndEvent, EvaluationSkippedEvent,
    EvaluationStartEvent, GEPACallback, IterationEndEvent, IterationStartEvent, MergeAcceptedEvent,
    MergeAttemptedEvent, MergeRejectedEvent, MinibatchSampledEvent, OptimizationEndEvent,
    OptimizationStartEvent, ParetoFrontUpdatedEvent, ProposalEndEvent, ProposalStartEvent,
    ReflectiveDatasetBuiltEvent, StateSavedEvent, ValsetEvaluatedEvent, notify_callbacks,
};
