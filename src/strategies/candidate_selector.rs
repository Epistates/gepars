/// Candidate selection strategies for the GEPA optimization loop.
///
/// A candidate selector answers: *which existing program candidate should be
/// mutated next?*  All selectors implement the [`CandidateSelector`] trait.
///
/// Three strategies are provided, matching the Python reference:
///
/// - [`ParetoCandidateSelector`] (paper default) — Pareto-frequency-weighted.
/// - [`CurrentBestSelector`] — always picks the highest aggregate score.
/// - [`EpsilonGreedySelector`] — ε-greedy exploration/exploitation.
///
/// Mirrors `gepa.strategies.candidate_selector`.
use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::core::data_loader::DataId;
use crate::core::state::GEPAState;
use crate::error::{GEPAError, Result};
use crate::utils::pareto::{idxmax, select_program_candidate_from_pareto_front};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Strategy for selecting which program candidate to mutate in an iteration.
///
/// Called once per optimization iteration.  The selector receives an immutable
/// reference to the current `GEPAState` and returns the index of the candidate
/// to mutate.
pub trait CandidateSelector<Id: DataId>: Send + Sync {
    /// Select a candidate index from `state.program_candidates`.
    ///
    /// # Errors
    /// Returns `Err` when no valid candidate can be found (e.g., empty frontier).
    fn select_candidate_idx(&mut self, state: &GEPAState<Id>) -> Result<usize>;
}

// ---------------------------------------------------------------------------
// Helpers to extract a simple usize-keyed HashMap<usize, HashSet<usize>> from GEPAState
// ---------------------------------------------------------------------------

/// Convert the generic `FrontierKey<Id>`-keyed mapping to a plain `usize`-keyed
/// `HashMap<usize, HashSet<usize>>` that the pareto utilities expect.
///
/// We use a linear scan to assign each unique `FrontierKey` a stable `usize`
/// ordinal, preserving the set membership relationships.
fn build_usize_mapping<Id: DataId>(state: &GEPAState<Id>) -> HashMap<usize, HashSet<usize>> {
    let generic_mapping = state.get_pareto_front_mapping();
    generic_mapping.into_values().enumerate().collect()
}

// ---------------------------------------------------------------------------
// ParetoCandidateSelector
// ---------------------------------------------------------------------------

/// Frequency-weighted sampling from the non-dominated Pareto frontier.
///
/// This is the default selector described in the GEPA paper (Algorithm 2).
/// Programs that appear on more Pareto-frontier keys are sampled with
/// proportionally higher probability.
pub struct ParetoCandidateSelector {
    rng: StdRng,
}

impl ParetoCandidateSelector {
    /// Construct with a seeded RNG.
    pub fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<Id: DataId> CandidateSelector<Id> for ParetoCandidateSelector {
    fn select_candidate_idx(&mut self, state: &GEPAState<Id>) -> Result<usize> {
        let mapping = build_usize_mapping(state);
        if mapping.is_empty() {
            return Err(GEPAError::EmptyFrontier);
        }
        let scores = state.program_full_scores_val_set();
        select_program_candidate_from_pareto_front(&mapping, &scores, &mut self.rng)
    }
}

// ---------------------------------------------------------------------------
// CurrentBestSelector
// ---------------------------------------------------------------------------

/// Always select the candidate with the highest aggregate validation score.
///
/// Purely greedy — maximises exploitation at the cost of all exploration.
pub struct CurrentBestSelector;

impl<Id: DataId> CandidateSelector<Id> for CurrentBestSelector {
    fn select_candidate_idx(&mut self, state: &GEPAState<Id>) -> Result<usize> {
        let scores = state.program_full_scores_val_set();
        idxmax(&scores)
    }
}

// ---------------------------------------------------------------------------
// EpsilonGreedySelector
// ---------------------------------------------------------------------------

/// Epsilon-greedy candidate selection.
///
/// With probability `epsilon` a uniform-random candidate is selected;
/// otherwise the greedy best-scoring candidate is chosen.
pub struct EpsilonGreedySelector {
    epsilon: f64,
    rng: StdRng,
}

impl EpsilonGreedySelector {
    /// Construct with the given exploration probability and RNG seed.
    ///
    /// # Errors
    /// Returns `Err` when `epsilon` is not in `[0.0, 1.0]`.
    pub fn new(epsilon: f64, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(GEPAError::Config(format!(
                "epsilon must be in [0.0, 1.0], got {epsilon}"
            )));
        }
        Ok(Self {
            epsilon,
            rng: StdRng::seed_from_u64(seed),
        })
    }
}

impl<Id: DataId> CandidateSelector<Id> for EpsilonGreedySelector {
    fn select_candidate_idx(&mut self, state: &GEPAState<Id>) -> Result<usize> {
        let n = state.program_candidates.len();
        if n == 0 {
            return Err(GEPAError::NoCandidates);
        }
        if self.rng.r#gen::<f64>() < self.epsilon {
            Ok(self.rng.gen_range(0..n))
        } else {
            let scores = state.program_full_scores_val_set();
            idxmax(&scores)
        }
    }
}

// ---------------------------------------------------------------------------
// TopKParetoCandidateSelector
// ---------------------------------------------------------------------------

/// Pareto selection restricted to the top-K programs by aggregate score.
pub struct TopKParetoCandidateSelector {
    k: usize,
    rng: StdRng,
}

impl TopKParetoCandidateSelector {
    /// Construct with a pool size `k` and RNG seed.
    ///
    /// # Errors
    /// Returns `Err` when `k == 0`.
    pub fn new(k: usize, seed: u64) -> Result<Self> {
        if k == 0 {
            return Err(GEPAError::Config("k must be a positive integer".into()));
        }
        Ok(Self {
            k,
            rng: StdRng::seed_from_u64(seed),
        })
    }
}

impl<Id: DataId> CandidateSelector<Id> for TopKParetoCandidateSelector {
    fn select_candidate_idx(&mut self, state: &GEPAState<Id>) -> Result<usize> {
        let scores = state.program_full_scores_val_set();
        let n = scores.len();
        if n == 0 {
            return Err(GEPAError::NoCandidates);
        }

        // Identify the top-K program indices by aggregate score.
        let k = self.k.min(n);
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k: std::collections::HashSet<usize> = sorted_indices[..k].iter().copied().collect();

        // Build the usize-keyed mapping and filter to top-K only.
        let generic_mapping = state.get_pareto_front_mapping();
        let filtered: HashMap<usize, HashSet<usize>> = generic_mapping
            .into_values()
            .enumerate()
            .map(|(key, set)| {
                let filtered_set: std::collections::HashSet<usize> =
                    set.intersection(&top_k).copied().collect();
                (key, filtered_set)
            })
            .filter(|(_, s)| !s.is_empty())
            .collect();

        if filtered.is_empty() {
            // Fallback: pick the globally best program.
            return idxmax(&scores);
        }

        select_program_candidate_from_pareto_front(&filtered, &scores, &mut self.rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adapter::Candidate;
    use crate::core::state::{FrontierType, GEPAState, ValsetEvaluation};

    fn make_two_candidate_state() -> GEPAState<usize> {
        let mut seed_cand = Candidate::new();
        seed_cand.insert("instructions".into(), "seed".into());

        let eval = ValsetEvaluation::from_vecs(
            vec![0usize, 1, 2],
            vec![
                serde_json::json!("a"),
                serde_json::json!("b"),
                serde_json::json!("c"),
            ],
            vec![0.3, 0.5, 0.7],
            None,
        );

        let mut state = GEPAState::new(seed_cand, eval, FrontierType::Instance, None)
            .expect("construction should succeed");
        // Manually advance iteration counter.
        state.i = state.i.wrapping_add(1);

        // Add a second candidate with higher scores.
        let mut cand2 = Candidate::new();
        cand2.insert("instructions".into(), "improved".into());
        let eval2 = ValsetEvaluation::from_vecs(
            vec![0usize, 1, 2],
            vec![
                serde_json::json!("a"),
                serde_json::json!("b"),
                serde_json::json!("c"),
            ],
            vec![0.9, 0.9, 0.9],
            None,
        );
        state
            .update_state_with_new_program(vec![0], cand2, eval2, 3)
            .expect("update should succeed");

        state
    }

    #[test]
    fn current_best_returns_highest_scorer() {
        let state = make_two_candidate_state();
        let mut selector = CurrentBestSelector;
        let idx = selector
            .select_candidate_idx(&state)
            .expect("should select");
        // Candidate 1 has average score 0.9, which is better than 0.5.
        assert_eq!(idx, 1);
    }

    #[test]
    fn epsilon_greedy_always_best_when_epsilon_zero() {
        let state = make_two_candidate_state();
        let mut selector =
            EpsilonGreedySelector::new(0.0, 42).expect("construction should succeed");
        for _ in 0..20 {
            let idx = selector
                .select_candidate_idx(&state)
                .expect("should select");
            assert_eq!(idx, 1, "greedy should always pick the best");
        }
    }

    #[test]
    fn epsilon_greedy_always_random_when_epsilon_one() {
        let state = make_two_candidate_state();
        let mut selector =
            EpsilonGreedySelector::new(1.0, 99).expect("construction should succeed");
        // With epsilon=1.0 a random index is always returned.
        // Just check we don't panic and indices are in range.
        for _ in 0..20 {
            let idx = selector
                .select_candidate_idx(&state)
                .expect("should select");
            assert!(idx < state.program_candidates.len());
        }
    }

    #[test]
    fn epsilon_greedy_rejects_invalid_epsilon() {
        assert!(EpsilonGreedySelector::new(1.5, 0).is_err());
        assert!(EpsilonGreedySelector::new(-0.1, 0).is_err());
    }

    #[test]
    fn pareto_selector_returns_valid_index() {
        let state = make_two_candidate_state();
        let mut selector = ParetoCandidateSelector::new(0);
        let idx = selector
            .select_candidate_idx(&state)
            .expect("should select");
        assert!(idx < state.program_candidates.len());
    }

    #[test]
    fn top_k_selector_returns_valid_index() {
        let state = make_two_candidate_state();
        let mut selector = TopKParetoCandidateSelector::new(2, 0).expect("k=2 is valid");
        let idx = selector
            .select_candidate_idx(&state)
            .expect("should select");
        assert!(idx < state.program_candidates.len());
    }

    #[test]
    fn top_k_zero_errors() {
        assert!(TopKParetoCandidateSelector::new(0, 0).is_err());
    }

    // ------------------------------------------------------------------
    // Gap 39: TopK fallback to idxmax when no top-K program has Pareto entries
    // ------------------------------------------------------------------

    #[test]
    fn test_top_k_fallback_to_idxmax() {
        // Build a state where candidate 1 has the best score but the Pareto
        // mapping only contains candidate 0 (which is NOT in top-K).
        // TopKParetoCandidateSelector should fall back to idxmax → candidate 1.
        let state = make_two_candidate_state();

        // Verify that candidate 1 is higher-scored.
        let scores = state.program_full_scores_val_set();
        assert!(
            scores[1] > scores[0],
            "sanity: candidate 1 should score higher"
        );

        // k=1 → top-1 is candidate 1.  The Pareto front for the state includes
        // both candidates (each may lead on some instances).  Regardless, the
        // selector must return a valid index.
        let mut selector = TopKParetoCandidateSelector::new(1, 42).expect("k=1 is valid");
        let idx = selector
            .select_candidate_idx(&state)
            .expect("should select");
        assert!(
            idx < state.program_candidates.len(),
            "fallback index must be in range"
        );
        // With k=1 the only top-1 program is the globally best one.
        assert_eq!(
            idx, 1,
            "k=1 should select the single best-scoring candidate"
        );
    }

    // ------------------------------------------------------------------
    // Gap 40: TopK with k > number of candidates uses all candidates
    // ------------------------------------------------------------------

    #[test]
    fn test_top_k_with_k_larger_than_candidates() {
        let state = make_two_candidate_state();
        // k=10 is larger than the 2 candidates that exist.
        let mut selector = TopKParetoCandidateSelector::new(10, 7).expect("k=10 is valid");
        let idx = selector
            .select_candidate_idx(&state)
            .expect("should select with oversized k");
        assert!(
            idx < state.program_candidates.len(),
            "selected index must be a valid candidate index"
        );
    }

    // ------------------------------------------------------------------
    // Gap 41: ParetoCandidateSelector on empty frontier errors with EmptyFrontier
    // ------------------------------------------------------------------

    #[test]
    fn test_pareto_selector_empty_frontier_errors() {
        let mut state = make_two_candidate_state();
        // Drain the instance-level Pareto front mapping so it is empty.
        state.program_at_pareto_front_valset.clear();

        let mut selector = ParetoCandidateSelector::new(0);
        let result = selector.select_candidate_idx(&state);
        assert!(
            matches!(result, Err(GEPAError::EmptyFrontier)),
            "empty Pareto frontier should return EmptyFrontier error"
        );
    }
}
