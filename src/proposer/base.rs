/// Base types for the proposer subsystem.
///
/// This module defines the data structures shared by every proposer strategy
/// in GEPA.  Concrete proposers (reflective mutation, merge, etc.) return
/// [`CandidateProposal`] objects that the engine uses to make acceptance
/// decisions.
///
/// Mirrors the Python `gepa.proposer.base` module.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::adapter::Candidate;
use crate::core::data_loader::DataId;
use crate::core::state::ProgramIdx;

// ---------------------------------------------------------------------------
// CandidateProposal
// ---------------------------------------------------------------------------

/// Encapsulates a proposed new candidate and the context of its creation.
///
/// Returned by every `ProposeCandidate` strategy.  The engine uses
/// `subsample_scores_before` and `subsample_scores_after` to make the fast
/// mini-batch acceptance decision before committing to an expensive full
/// validation evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "Id: DataId"))]
pub struct CandidateProposal<Id: DataId> {
    /// The proposed program — mapping from component name to component text.
    pub candidate: Candidate,

    /// Indices (within the current `GEPAState`) of the parent candidate(s)
    /// that this proposal was derived from.
    pub parent_program_ids: Vec<ProgramIdx>,

    /// IDs of the training examples used during the proposer's evaluation.
    /// `None` when the proposer did not subsample (e.g., merge proposals that
    /// do their own selection).
    pub subsample_indices: Option<Vec<Id>>,

    /// Per-example scores of the *parent* program on `subsample_indices`.
    /// Used by the engine to check for improvement.
    pub subsample_scores_before: Option<Vec<f64>>,

    /// Per-example scores of the *proposed* program on `subsample_indices`.
    /// Used by the engine to check for improvement.
    pub subsample_scores_after: Option<Vec<f64>>,

    /// Free-form label identifying the proposal strategy
    /// (e.g., `"reflective_mutation"` or `"merge"`).  Used for logging.
    pub tag: String,

    /// Optional additional key-value pairs for logging/tracing.
    pub metadata: HashMap<String, Value>,
}

impl<Id: DataId> CandidateProposal<Id> {
    /// Construct a minimal proposal with only the required fields.
    pub fn new(candidate: Candidate, parent_program_ids: Vec<ProgramIdx>) -> Self {
        Self {
            candidate,
            parent_program_ids,
            subsample_indices: None,
            subsample_scores_before: None,
            subsample_scores_after: None,
            tag: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Attach a tag identifying the proposal strategy.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = tag.into();
        self
    }

    /// Attach subsample context (indices + before/after scores).
    pub fn with_subsample(
        mut self,
        indices: Vec<Id>,
        scores_before: Vec<f64>,
        scores_after: Vec<f64>,
    ) -> Self {
        self.subsample_indices = Some(indices);
        self.subsample_scores_before = Some(scores_before);
        self.subsample_scores_after = Some(scores_after);
        self
    }

    /// Attach arbitrary metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Return `true` if the mini-batch scores show improvement.
    ///
    /// Computes `sum(after) > sum(before)`.  Returns `None` when subsample
    /// scores are not available.
    pub fn shows_improvement(&self) -> Option<bool> {
        let before = self.subsample_scores_before.as_ref()?;
        let after = self.subsample_scores_after.as_ref()?;
        let sum_before: f64 = before.iter().sum();
        let sum_after: f64 = after.iter().sum();
        Some(sum_after > sum_before)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adapter::Candidate;

    #[test]
    fn candidate_proposal_builder() {
        let mut cand = Candidate::new();
        cand.insert("instructions".into(), "improved".into());

        let proposal = CandidateProposal::new(cand.clone(), vec![0])
            .with_tag("reflective_mutation")
            .with_subsample(vec![0usize, 1], vec![0.5, 0.4], vec![0.7, 0.8])
            .with_metadata("ancestor", serde_json::json!(3));

        assert_eq!(proposal.tag, "reflective_mutation");
        assert!(proposal.subsample_indices.is_some());
        assert_eq!(
            proposal.metadata.get("ancestor"),
            Some(&serde_json::json!(3))
        );
    }

    #[test]
    fn shows_improvement_returns_correct_result() {
        let mut cand = Candidate::new();
        cand.insert("instructions".into(), "v2".into());

        let improving = CandidateProposal::new(cand.clone(), vec![0]).with_subsample(
            vec![0usize],
            vec![0.4],
            vec![0.8],
        );
        assert_eq!(improving.shows_improvement(), Some(true));

        let regressing = CandidateProposal::new(cand.clone(), vec![0]).with_subsample(
            vec![0usize],
            vec![0.8],
            vec![0.4],
        );
        assert_eq!(regressing.shows_improvement(), Some(false));

        let no_data = CandidateProposal::<usize>::new(cand, vec![0]);
        assert_eq!(no_data.shows_improvement(), None);
    }
}
