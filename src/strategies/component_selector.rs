/// Component selection strategies for reflective mutation.
///
/// A component selector answers: *which named components of the current program
/// candidate should the reflective mutation proposer update this iteration?*
///
/// Two strategies provided:
/// - [`RoundRobinSelector`] (paper default) — cycles through components one at a time.
/// - [`AllComponentSelector`] — returns all component names every call.
///
/// Mirrors `gepa.strategies.component_selector`.
use serde_json::Value;

use crate::core::adapter::Candidate;
use crate::core::data_loader::DataId;
use crate::core::state::GEPAState;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Strategy for selecting which components to update in a given iteration.
///
/// The proposer calls this once it has evaluated the current candidate on the
/// minibatch.  The selector returns a non-empty list of component names.
pub trait ComponentSelector<Id: DataId>: Send + Sync {
    /// Return the component names to update.
    ///
    /// # Arguments
    /// * `state`            — current optimizer state (provides round-robin counters).
    /// * `trajectories`     — per-example execution traces from the current eval.
    /// * `subsample_scores` — per-example scores from the minibatch evaluation.
    /// * `candidate_idx`    — index of the candidate being mutated.
    /// * `candidate`        — the current candidate dictionary.
    ///
    /// # Errors
    /// Returns `Err` when no components are available (empty candidate).
    fn select_components(
        &self,
        state: &mut GEPAState<Id>,
        trajectories: &[Value],
        subsample_scores: &[f64],
        candidate_idx: usize,
        candidate: &Candidate,
    ) -> Result<Vec<String>>;
}

// ---------------------------------------------------------------------------
// RoundRobinSelector
// ---------------------------------------------------------------------------

/// Cycles through named components in round-robin order per candidate.
///
/// For each candidate, an independent round-robin pointer is maintained in
/// `state.named_predictor_id_to_update_next`.  On each call the selector
/// reads the current pointer, returns the name at that position in
/// `state.list_of_named_predictors`, and advances the pointer (mod count).
///
/// This is the default strategy from the GEPA paper: one component at a time
/// receives reflective mutations, keeping each LM call focused.
pub struct RoundRobinSelector;

impl<Id: DataId> ComponentSelector<Id> for RoundRobinSelector {
    fn select_components(
        &self,
        state: &mut GEPAState<Id>,
        _trajectories: &[Value],
        _subsample_scores: &[f64],
        candidate_idx: usize,
        _candidate: &Candidate,
    ) -> Result<Vec<String>> {
        let n = state.list_of_named_predictors.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let pid = state.named_predictor_id_to_update_next[candidate_idx] % n;
        // Advance the pointer for the next call.
        state.named_predictor_id_to_update_next[candidate_idx] = (pid + 1) % n;
        Ok(vec![state.list_of_named_predictors[pid].clone()])
    }
}

// ---------------------------------------------------------------------------
// AllComponentSelector
// ---------------------------------------------------------------------------

/// Returns all component names on every call.
///
/// Suitable when the compound AI system has a single component (where it is
/// equivalent to `RoundRobinSelector`) or when joint updates are desired.
pub struct AllComponentSelector;

impl<Id: DataId> ComponentSelector<Id> for AllComponentSelector {
    fn select_components(
        &self,
        _state: &mut GEPAState<Id>,
        _trajectories: &[Value],
        _subsample_scores: &[f64],
        _candidate_idx: usize,
        candidate: &Candidate,
    ) -> Result<Vec<String>> {
        // IndexMap preserves insertion order; return keys in that order.
        Ok(candidate.keys().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::{FrontierType, ValsetEvaluation};

    fn make_multi_component_state() -> GEPAState<usize> {
        let mut seed = Candidate::new();
        seed.insert("alpha".into(), "first".into());
        seed.insert("beta".into(), "second".into());
        seed.insert("gamma".into(), "third".into());

        let eval = ValsetEvaluation::from_vecs(
            vec![0usize],
            vec![serde_json::json!("out")],
            vec![0.5],
            None,
        );
        GEPAState::new(seed, eval, FrontierType::Instance, None)
            .expect("construction should succeed")
    }

    #[test]
    fn round_robin_cycles_through_all_components() {
        let mut state = make_multi_component_state();
        let candidate = state.program_candidates[0].clone();
        let selector = RoundRobinSelector;

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for _ in 0..3 {
            let names = selector
                .select_components(&mut state, &[], &[], 0, &candidate)
                .expect("should select");
            assert_eq!(names.len(), 1);
            seen.insert(names[0].clone());
        }
        // After 3 calls all 3 components should have been selected.
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn round_robin_wraps_around() {
        let mut state = make_multi_component_state();
        let candidate = state.program_candidates[0].clone();
        let selector = RoundRobinSelector;

        let mut calls: Vec<String> = Vec::new();
        for _ in 0..6 {
            let names = selector
                .select_components(&mut state, &[], &[], 0, &candidate)
                .expect("should select");
            calls.push(names[0].clone());
        }
        // First 3 should match the next 3 (round-robin wraps).
        assert_eq!(calls[0], calls[3]);
        assert_eq!(calls[1], calls[4]);
        assert_eq!(calls[2], calls[5]);
    }

    #[test]
    fn all_component_selector_returns_all_components() {
        let mut state = make_multi_component_state();
        let candidate = state.program_candidates[0].clone();
        let selector = AllComponentSelector;

        let names = selector
            .select_components(&mut state, &[], &[], 0, &candidate)
            .expect("should select");

        assert_eq!(names.len(), 3);
        // IndexMap preserves insertion order — all three must be present.
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert!(names.contains(&"gamma".to_string()));
    }

    // ------------------------------------------------------------------
    // Gap 43: round-robin on a single-component candidate always returns same
    // ------------------------------------------------------------------

    #[test]
    fn test_round_robin_single_component() {
        // Build a state with only one component.
        let mut seed = Candidate::new();
        seed.insert("only".into(), "value".into());

        let eval = ValsetEvaluation::from_vecs(
            vec![0usize],
            vec![serde_json::json!("out")],
            vec![0.5],
            None,
        );
        let mut state = GEPAState::new(seed, eval, FrontierType::Instance, None)
            .expect("construction should succeed");

        let candidate = state.program_candidates[0].clone();
        let selector = RoundRobinSelector;

        let first = selector
            .select_components(&mut state, &[], &[], 0, &candidate)
            .expect("should select");
        let second = selector
            .select_components(&mut state, &[], &[], 0, &candidate)
            .expect("should select again");
        let third = selector
            .select_components(&mut state, &[], &[], 0, &candidate)
            .expect("should select third time");

        // With a single component, round-robin always returns the same one.
        assert_eq!(first, vec!["only".to_string()]);
        assert_eq!(second, vec!["only".to_string()]);
        assert_eq!(third, vec!["only".to_string()]);
    }

    #[test]
    fn round_robin_per_candidate_independence() {
        // Two candidates should have independent counters.
        let mut state = make_multi_component_state();

        // Add a second candidate.
        let mut cand2 = Candidate::new();
        cand2.insert("alpha".into(), "a2".into());
        cand2.insert("beta".into(), "b2".into());
        cand2.insert("gamma".into(), "g2".into());
        let eval2 = ValsetEvaluation::from_vecs(
            vec![0usize],
            vec![serde_json::json!("out2")],
            vec![0.8],
            None,
        );
        state
            .update_state_with_new_program(vec![0], cand2, eval2, 1)
            .expect("update should succeed");

        let candidate0 = state.program_candidates[0].clone();
        let candidate1 = state.program_candidates[1].clone();
        let selector = RoundRobinSelector;

        // Advance candidate 0's pointer twice.
        selector
            .select_components(&mut state, &[], &[], 0, &candidate0)
            .unwrap();
        selector
            .select_components(&mut state, &[], &[], 0, &candidate0)
            .unwrap();

        // Candidate 1's pointer should still be at 0 (first component).
        let next_for_1 = selector
            .select_components(&mut state, &[], &[], 1, &candidate1)
            .expect("should select for candidate 1");
        // Candidate 1's counter was never advanced, so it returns the first component.
        assert_eq!(next_for_1.len(), 1);
    }
}
