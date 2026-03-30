/// Strategy modules for the GEPA optimization loop.
///
/// Each strategy module is independently pluggable: swap out any strategy
/// without changing the engine or other strategies.
pub mod batch_sampler;
pub mod candidate_selector;
pub mod component_selector;
pub mod eval_policy;
pub mod instruction_proposal;

// Re-export the most commonly used types.
pub use batch_sampler::{BatchSampler, EpochShuffledSampler};
pub use candidate_selector::{
    CandidateSelector, CurrentBestSelector, EpsilonGreedySelector, ParetoCandidateSelector,
    TopKParetoCandidateSelector,
};
pub use component_selector::{AllComponentSelector, ComponentSelector, RoundRobinSelector};
pub use eval_policy::{EvalPolicy, FullEvalPolicy};
pub use instruction_proposal::{
    META_PROMPT_TEMPLATE, extract_output, format_samples_as_markdown, render_prompt,
    render_prompt_with_template, validate_prompt_template,
};
