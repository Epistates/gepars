/// GEPA — Reflective Prompt Evolution implemented in Rust.
///
/// This crate is a full port of the GEPA algorithm described in
/// "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
/// (2025).
///
/// ## Module organisation
///
/// - [`error`]      — unified error type.
/// - [`core`]       — adapter, data loader, state, result, callbacks, engine.
/// - [`lm`]         — language-model abstraction and OpenAI-compatible client.
/// - [`proposer`]   — candidate proposal base types, reflective mutation, merge.
/// - [`strategies`] — Pareto-front selection, component selection, batch sampling,
///                    evaluation policy, instruction proposal.
/// - [`utils`]      — stop conditions.
/// - [`api`]        — high-level `optimize()` function.
pub mod adapters;
pub mod api;
pub mod core;
pub mod error;
pub mod lm;
pub mod proposer;
pub mod strategies;
pub mod tracking;
pub mod utils;

// ---------------------------------------------------------------------------
// Top-level re-exports
// ---------------------------------------------------------------------------

pub use error::{GEPAError, Result};

// Core
pub use core::adapter::{Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset};
pub use core::callbacks::{
    CandidateAcceptedEvent, CandidateRejectedEvent, CompositeCallback, GEPACallback,
    OptimizationEndEvent, OptimizationStartEvent, notify_callbacks,
};
pub use core::component::{ComponentKind, ComponentMeta, ComponentMetaMap, component_kind};
pub use core::data_loader::{DataId, DataLoader, MutableDataLoader, VecLoader, ensure_loader};
pub use core::engine::GEPAEngine;
pub use core::result::GEPAResult;
pub use core::state::{
    EvaluationCache, FrontierKey, FrontierType, GEPAState, ObjectiveScores, ProgramIdx,
    ValsetEvaluation, candidate_hash,
};

// LM
pub use lm::{LanguageModel, OpenAICompatibleLM};

// Utils
pub use utils::pareto::{
    find_dominator_programs, idxmax, is_dominated, remove_dominated_programs,
    select_program_candidate_from_pareto_front,
};
pub use utils::stop_condition::{
    CompositeMode, CompositeStopper, FileStopper, MaxIterationsStopper, MaxMetricCallsStopper,
    NoImprovementStopper, SignalStopper, StopCondition, TimeoutStopper,
};

// Proposer
pub use proposer::base::CandidateProposal;
pub use proposer::merge::MergeProposer;
pub use proposer::reflective_mutation::ReflectiveMutationProposer;

// Strategies
pub use strategies::batch_sampler::{BatchSampler, EpochShuffledSampler};
pub use strategies::candidate_selector::{
    CandidateSelector, CurrentBestSelector, EpsilonGreedySelector, ParetoCandidateSelector,
    TopKParetoCandidateSelector,
};
pub use strategies::component_selector::{
    AllComponentSelector, ComponentSelector, RoundRobinSelector,
};
pub use strategies::eval_policy::{EvalPolicy, FullEvalPolicy};
pub use strategies::instruction_proposal::{
    CODE_META_PROMPT_TEMPLATE, CONFIG_META_PROMPT_TEMPLATE, META_PROMPT_TEMPLATE, extract_output,
    format_samples_as_markdown, render_code_prompt, render_config_prompt, render_prompt,
};

// API
pub use api::{
    CandidateSelectorKind, ComponentSelectorKind, LMConfig, OptimizeConfig, StopConditionConfig,
    optimize,
};

// Tracking
pub use tracking::{ExperimentTracker, NoopTracker};

// Proposer (additional)
pub use proposer::reflective_mutation::PromptTemplateConfig;
