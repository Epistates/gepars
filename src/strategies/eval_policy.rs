/// Validation evaluation policies for the GEPA engine.
///
/// An evaluation policy controls:
/// 1. **Which validation examples to evaluate** a candidate on (`get_eval_ids`).
/// 2. **Which candidate to report as best** given all validation results
///    accumulated so far (`get_best_program`).
/// 3. **The validation-set aggregate score** for a specific program
///    (`get_valset_score`).
///
/// Mirrors `gepa.strategies.eval_policy`.
use crate::core::data_loader::DataId;
use crate::core::state::{GEPAState, ProgramIdx};
use crate::error::{GEPAError, Result};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Strategy for choosing validation examples and identifying the best program.
pub trait EvalPolicy<Id: DataId>: Send + Sync {
    /// Return the validation example IDs to evaluate for a candidate.
    ///
    /// `target_program_idx` is optional; implementations that do incremental
    /// evaluation (evaluating only unseen examples) can use it to filter.
    /// `FullEvalPolicy` ignores it.
    fn get_eval_ids(
        &self,
        all_val_ids: &[Id],
        state: &GEPAState<Id>,
        target_program_idx: Option<ProgramIdx>,
    ) -> Vec<Id>;

    /// Return the index of the best program given all validation results.
    ///
    /// # Errors
    /// Returns `Err` when there are no candidates.
    fn get_best_program(&self, state: &GEPAState<Id>) -> Result<ProgramIdx>;

    /// Return the validation-set aggregate score for `program_idx`.
    ///
    /// # Errors
    /// Returns `Err` when `program_idx` is out of range.
    fn get_valset_score(&self, program_idx: ProgramIdx, state: &GEPAState<Id>) -> Result<f64>;
}

// ---------------------------------------------------------------------------
// FullEvalPolicy
// ---------------------------------------------------------------------------

/// Policy that evaluates all validation instances every time.
///
/// The simplest and most accurate policy: every candidate is scored on the
/// entire validation set, and the program with the highest average score across
/// all evaluated examples is returned as the best.
///
/// **Tie-breaking**: when two programs have the same average score, the one
/// with greater coverage (more evaluated validation examples) is preferred.
/// This avoids favouring programs that have only been partially evaluated.
pub struct FullEvalPolicy;

impl<Id: DataId> EvalPolicy<Id> for FullEvalPolicy {
    fn get_eval_ids(
        &self,
        all_val_ids: &[Id],
        _state: &GEPAState<Id>,
        _target_program_idx: Option<ProgramIdx>,
    ) -> Vec<Id> {
        all_val_ids.to_vec()
    }

    fn get_best_program(&self, state: &GEPAState<Id>) -> Result<ProgramIdx> {
        if state.program_candidates.is_empty() {
            return Err(GEPAError::NoCandidates);
        }

        let mut best_idx: ProgramIdx = 0;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_coverage: usize = 0;

        for (program_idx, scores) in state.prog_candidate_val_subscores.iter().enumerate() {
            let coverage = scores.len();
            let avg = if coverage == 0 {
                f64::NEG_INFINITY
            } else {
                scores.values().sum::<f64>() / coverage as f64
            };

            if avg > best_score || (avg == best_score && coverage > best_coverage) {
                best_score = avg;
                best_idx = program_idx;
                best_coverage = coverage;
            }
        }

        Ok(best_idx)
    }

    fn get_valset_score(&self, program_idx: ProgramIdx, state: &GEPAState<Id>) -> Result<f64> {
        if program_idx >= state.prog_candidate_val_subscores.len() {
            return Err(GEPAError::Config(format!(
                "program_idx {program_idx} out of range (len={})",
                state.prog_candidate_val_subscores.len()
            )));
        }
        Ok(state.get_program_average_val_subset(program_idx).0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adapter::Candidate;
    use crate::core::state::{FrontierType, GEPAState, ValsetEvaluation};

    fn make_state_with_two_programs() -> GEPAState<usize> {
        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "seed".into());

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
        let mut state =
            GEPAState::new(seed, eval, FrontierType::Instance, None).expect("should construct");

        let mut cand2 = Candidate::new();
        cand2.insert("instructions".into(), "improved".into());
        let eval2 = ValsetEvaluation::from_vecs(
            vec![0usize, 1, 2],
            vec![
                serde_json::json!("a"),
                serde_json::json!("b"),
                serde_json::json!("c"),
            ],
            vec![0.8, 0.9, 1.0],
            None,
        );
        state
            .update_state_with_new_program(vec![0], cand2, eval2, 3)
            .expect("update should succeed");
        state
    }

    #[test]
    fn full_eval_policy_get_eval_ids_returns_all() {
        let state = make_state_with_two_programs();
        let policy = FullEvalPolicy;
        let all_ids: Vec<usize> = vec![0, 1, 2, 3, 4];
        let ids = policy.get_eval_ids(&all_ids, &state, None);
        assert_eq!(ids, all_ids);
    }

    #[test]
    fn full_eval_policy_best_program_picks_highest_scorer() {
        let state = make_state_with_two_programs();
        let policy = FullEvalPolicy;
        let best = policy.get_best_program(&state).expect("should find best");
        // Candidate 1 has avg score (0.8+0.9+1.0)/3 ≈ 0.9.
        assert_eq!(best, 1);
    }

    #[test]
    fn full_eval_policy_valset_score_correct() {
        let state = make_state_with_two_programs();
        let policy = FullEvalPolicy;

        let score0 = policy
            .get_valset_score(0, &state)
            .expect("should get score for candidate 0");
        let score1 = policy
            .get_valset_score(1, &state)
            .expect("should get score for candidate 1");

        assert!((score0 - 0.5).abs() < 1e-9, "seed avg should be ~0.5");
        assert!(
            (score1 - 0.9).abs() < 1e-9,
            "improved avg should be ~0.9, got {score1}"
        );
    }

    #[test]
    fn full_eval_policy_out_of_range_errors() {
        let state = make_state_with_two_programs();
        let policy = FullEvalPolicy;
        assert!(policy.get_valset_score(99, &state).is_err());
    }

    #[test]
    fn full_eval_policy_no_candidates_errors() {
        let policy = FullEvalPolicy;
        // Build a state then manually truncate (not a real use-case but
        // exercises the error path).
        let mut state = make_state_with_two_programs();
        state.program_candidates.clear();
        state.prog_candidate_val_subscores.clear();
        assert!(policy.get_best_program(&state).is_err());
    }

    // ------------------------------------------------------------------
    // Gap 42: tie-breaking prefers higher coverage
    // ------------------------------------------------------------------

    #[test]
    fn test_tie_breaking_prefers_higher_coverage() {
        // Build a state where two candidates have the same average score
        // but different coverage.  The one with more evaluated examples wins.
        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "seed".into());

        // Candidate 0: evaluated on 3 examples, average = 0.5.
        let eval0 = ValsetEvaluation::from_vecs(
            vec![0usize, 1, 2],
            vec![
                serde_json::json!("a"),
                serde_json::json!("b"),
                serde_json::json!("c"),
            ],
            vec![0.5, 0.5, 0.5],
            None,
        );
        let mut state =
            GEPAState::new(seed, eval0, FrontierType::Instance, None).expect("should construct");

        // Candidate 1: evaluated on only 1 example but same average = 0.5.
        let mut cand1 = Candidate::new();
        cand1.insert("instructions".into(), "candidate1".into());
        let eval1 = ValsetEvaluation::from_vecs(
            vec![3usize],
            vec![serde_json::json!("d")],
            vec![0.5],
            None,
        );
        state
            .update_state_with_new_program(vec![0], cand1, eval1, 1)
            .expect("update should succeed");

        let policy = FullEvalPolicy;
        let best = policy.get_best_program(&state).expect("should find best");

        // Both have avg=0.5 but candidate 0 has coverage=3 vs candidate 1 has coverage=1.
        // Tie-breaking should prefer candidate 0 (higher coverage).
        assert_eq!(
            best, 0,
            "tie-breaking should prefer the candidate with higher coverage"
        );
    }
}
