/// Proposer subsystem for the GEPA optimisation engine.
///
/// Proposers generate new candidate programs from the current state.
/// The engine drives concrete proposers via their `propose_mut` methods.
pub mod base;
pub mod merge;
pub mod reflective_mutation;

// ---------------------------------------------------------------------------
// Flat re-exports
// ---------------------------------------------------------------------------

pub use base::CandidateProposal;
pub use merge::MergeProposer;
pub use reflective_mutation::ReflectiveMutationProposer;
