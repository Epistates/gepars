/// Pareto-front utilities implementing Algorithm 2 from the GEPA paper.
///
/// This module provides the core mathematical machinery for sampling program
/// candidates from the Pareto frontier in a frequency-weighted, non-dominated
/// fashion.  All functions operate on a `pareto_front_mapping`:
/// a `HashMap<Key, HashSet<ProgramIdx>>` where each key is a frontier key
/// (e.g. a validation example ID) and each value is the set of program
/// indices that are currently best on that key.
///
/// Mirrors the Python `gepa.gepa_utils` module.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use rand::Rng;
use rand::rngs::StdRng;

use crate::core::state::ProgramIdx;
use crate::error::{GEPAError, Result};

// ---------------------------------------------------------------------------
// is_dominated
// ---------------------------------------------------------------------------

/// Check whether program `y` is dominated by at least one program in `programs`.
///
/// A program is dominated on a given frontier key when there exists another
/// program in `programs` that also appears in that key's Pareto front (i.e.,
/// achieves an equal-or-better score on that key).  A program is considered
/// globally dominated only when it is dominated on **every** frontier key where
/// it appears.
///
/// # Arguments
/// * `y`                          — the program index to test.
/// * `programs`                   — the pool of candidate programs (must exclude
///   `y` and any already-dominated programs).
/// * `pareto_front_mapping`       — mapping from frontier key to the set of
///   program indices Pareto-optimal on that key.
pub fn is_dominated<Key>(
    y: ProgramIdx,
    programs: &HashSet<ProgramIdx>,
    pareto_front_mapping: &HashMap<Key, HashSet<ProgramIdx>>,
) -> bool
where
    Key: Eq + Hash,
{
    // Collect the frontier keys where y appears.
    let y_fronts: Vec<&HashSet<ProgramIdx>> = pareto_front_mapping
        .values()
        .filter(|front| front.contains(&y))
        .collect();

    // If y doesn't appear anywhere, it cannot be meaningfully dominated.
    // Treat it as undominated so it's not silently removed.
    if y_fronts.is_empty() {
        return false;
    }

    for front in y_fronts {
        let dominator_present = front.iter().any(|&other| programs.contains(&other));
        if !dominator_present {
            // y has unique coverage on this key — it is not globally dominated.
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// remove_dominated_programs
// ---------------------------------------------------------------------------

/// Iteratively remove dominated programs from the Pareto front mapping.
///
/// Programs are eliminated in ascending score order (weakest first).  This
/// greedy approach mirrors Algorithm 2 in the GEPA paper: it removes programs
/// that are *fully dominated* — i.e., for every frontier key they appear on,
/// there is at least one other non-dominated program also present on that key.
///
/// # Arguments
/// * `pareto_front_mapping` — mapping from frontier key to the set of program
///   indices Pareto-optimal on that key.
/// * `scores`               — aggregate score per program index.  When `None`,
///   all programs are treated as equally scored (`1.0`), so ordering is
///   arbitrary.
///
/// # Returns
/// A filtered copy of `pareto_front_mapping` retaining only non-dominated
/// program indices.
pub fn remove_dominated_programs<Key>(
    pareto_front_mapping: &HashMap<Key, HashSet<ProgramIdx>>,
    scores: Option<&[f64]>,
) -> HashMap<Key, HashSet<ProgramIdx>>
where
    Key: Eq + Hash + Clone,
{
    // Collect frequency of each program across all frontier keys.
    let mut freq: HashMap<ProgramIdx, usize> = HashMap::new();
    for front in pareto_front_mapping.values() {
        for &p in front {
            *freq.entry(p).or_insert(0) += 1;
        }
    }

    let mut all_programs: Vec<ProgramIdx> = freq.keys().copied().collect();

    // Sort weakest first so we attempt to eliminate them before strong ones.
    all_programs.sort_by(|&a, &b| {
        let sa = scores.and_then(|s| s.get(a)).copied().unwrap_or(1.0);
        let sb = scores.and_then(|s| s.get(b)).copied().unwrap_or(1.0);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut dominated: HashSet<ProgramIdx> = HashSet::new();

    let mut found_to_remove = true;
    while found_to_remove {
        found_to_remove = false;
        for &y in &all_programs {
            if dominated.contains(&y) {
                continue;
            }
            // Pool: all programs except y and already-dominated ones.
            let pool: HashSet<ProgramIdx> = all_programs
                .iter()
                .copied()
                .filter(|&p| p != y && !dominated.contains(&p))
                .collect();

            if is_dominated(y, &pool, pareto_front_mapping) {
                dominated.insert(y);
                found_to_remove = true;
                break; // Restart the pass after any removal.
            }
        }
    }

    let dominators: HashSet<ProgramIdx> = all_programs
        .iter()
        .copied()
        .filter(|p| !dominated.contains(p))
        .collect();

    // Build filtered mapping retaining only non-dominated indices.
    let new_mapping: HashMap<Key, HashSet<ProgramIdx>> = pareto_front_mapping
        .iter()
        .map(|(key, front)| {
            let filtered: HashSet<ProgramIdx> = front
                .iter()
                .copied()
                .filter(|p| dominators.contains(p))
                .collect();
            (key.clone(), filtered)
        })
        .collect();

    // Invariant: every non-empty frontier key must still have at least one
    // surviving program.
    for (key, front) in pareto_front_mapping {
        if !front.is_empty() {
            debug_assert!(
                new_mapping.get(key).is_some_and(|f| !f.is_empty()),
                "Invariant violated: a non-empty frontier key lost all its programs \
                 after domination removal."
            );
        }
    }

    new_mapping
}

// ---------------------------------------------------------------------------
// find_dominator_programs
// ---------------------------------------------------------------------------

/// Return the sorted list of non-dominated program indices.
///
/// Thin wrapper around [`remove_dominated_programs`] that flattens the
/// resulting mapping into a deduplicated, sorted `Vec`.
///
/// # Arguments
/// * `pareto_front_programs` — mapping from frontier key to Pareto-optimal
///   program index sets.
/// * `scores`               — aggregate (e.g. train+val weighted) score per
///   program index.
pub fn find_dominator_programs<Key>(
    pareto_front_programs: &HashMap<Key, HashSet<ProgramIdx>>,
    scores: &[f64],
) -> Vec<ProgramIdx>
where
    Key: Eq + Hash + Clone,
{
    let new_mapping = remove_dominated_programs(pareto_front_programs, Some(scores));

    let mut uniq: HashSet<ProgramIdx> = HashSet::new();
    for front in new_mapping.values() {
        uniq.extend(front.iter().copied());
    }
    let mut result: Vec<ProgramIdx> = uniq.into_iter().collect();
    result.sort_unstable();
    result
}

// ---------------------------------------------------------------------------
// select_program_candidate_from_pareto_front  (Algorithm 2)
// ---------------------------------------------------------------------------

/// Sample a candidate proportionally to its Pareto-frontier frequency.
///
/// Implements Algorithm 2 from the GEPA paper:
///
/// 1. Remove dominated programs from the frontier mapping.
/// 2. Count how many frontier keys each surviving program appears on
///    (its *frequency*).
/// 3. Build a sampling list where each program is repeated proportionally
///    to its frequency.
/// 4. Sample uniformly from that list.
///
/// This naturally balances exploration (favouring programs that are
/// best-on-more-examples) with exploitation of strong candidates.
///
/// # Arguments
/// * `pareto_front_programs`   — mapping from frontier key to Pareto-optimal
///   program index sets.
/// * `scores`                  — aggregate score per program index; used by
///   `remove_dominated_programs` to order elimination (weakest first).
/// * `rng`                     — seeded `StdRng` for reproducibility.
///
/// # Errors
/// Returns `Err(GEPAError::EmptyFrontier)` if the frontier is empty or the
/// sampling list is empty after domination removal.
pub fn select_program_candidate_from_pareto_front<Key>(
    pareto_front_programs: &HashMap<Key, HashSet<ProgramIdx>>,
    scores: &[f64],
    rng: &mut StdRng,
) -> Result<ProgramIdx>
where
    Key: Eq + Hash + Clone,
{
    if pareto_front_programs.is_empty() {
        return Err(GEPAError::EmptyFrontier);
    }

    // Step 1 — remove dominated programs.
    let filtered_mapping = remove_dominated_programs(pareto_front_programs, Some(scores));

    // Step 2 — count per-program frontier coverage frequency.
    let mut program_frequency: HashMap<ProgramIdx, usize> = HashMap::new();
    for front in filtered_mapping.values() {
        for &prog_idx in front {
            *program_frequency.entry(prog_idx).or_insert(0) += 1;
        }
    }

    // Step 3 — build frequency-weighted sampling list.
    let sampling_list: Vec<ProgramIdx> = program_frequency
        .iter()
        .flat_map(|(&prog_idx, &freq)| std::iter::repeat_n(prog_idx, freq))
        .collect();

    if sampling_list.is_empty() {
        return Err(GEPAError::EmptyFrontier);
    }

    // Step 4 — uniform sample over the weighted list.
    let chosen_idx = rng.gen_range(0..sampling_list.len());
    Ok(sampling_list[chosen_idx])
}

// ---------------------------------------------------------------------------
// idxmax helper
// ---------------------------------------------------------------------------

/// Return the index of the maximum value in `lst`.
///
/// When multiple elements share the maximum, the first occurrence is returned.
///
/// # Errors
/// Returns `Err(GEPAError::NoCandidates)` if `lst` is empty.
///
/// # Panics
/// Cannot panic: the early-return guard ensures the iterator is non-empty
/// before calling `unwrap`.
pub fn idxmax(lst: &[f64]) -> Result<usize> {
    if lst.is_empty() {
        return Err(GEPAError::NoCandidates);
    }
    let (idx, _) = lst
        .iter()
        .enumerate()
        .reduce(|(mi, mv), (i, v)| if v > mv { (i, v) } else { (mi, mv) })
        .unwrap();
    Ok(idx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    // Build a frontier mapping from a vec of `(key, [prog_idx...])` pairs.
    fn make_frontier(entries: &[(u32, &[ProgramIdx])]) -> HashMap<u32, HashSet<ProgramIdx>> {
        entries
            .iter()
            .map(|(k, idxs)| (*k, idxs.iter().copied().collect()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // is_dominated
    // -----------------------------------------------------------------------

    #[test]
    fn is_dominated_when_fully_covered() {
        // Program 0 appears on key 0 and key 1.
        // Program 1 also appears on both keys.
        // With programs = {1}, program 0 is dominated.
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[0, 1])]);
        let pool: HashSet<ProgramIdx> = [1].into();
        assert!(is_dominated(0, &pool, &frontier));
    }

    #[test]
    fn not_dominated_when_unique_on_one_key() {
        // Program 0 is the sole occupant of key 1 — pool {1} cannot dominate it.
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[0])]);
        let pool: HashSet<ProgramIdx> = [1].into();
        assert!(!is_dominated(0, &pool, &frontier));
    }

    #[test]
    fn not_dominated_when_not_in_any_front() {
        // Program 99 doesn't appear anywhere in the frontier.
        let frontier = make_frontier(&[(0, &[0, 1])]);
        let pool: HashSet<ProgramIdx> = [0, 1].into();
        assert!(!is_dominated(99, &pool, &frontier));
    }

    // -----------------------------------------------------------------------
    // remove_dominated_programs
    // -----------------------------------------------------------------------

    #[test]
    fn remove_dominated_programs_eliminates_weak_fully_covered_program() {
        // Programs: 0 (0.2), 1 (0.7), 2 (0.9)
        // Key 0: {0, 1}, Key 1: {2}, Key 2: {1}
        // Program 0 is dominated (key 0 also has 1). After removing 0,
        // program 1 has unique coverage on key 2, so it survives.
        // Program 2 has unique coverage on key 1, so it survives.
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[2]), (2, &[1])]);
        let scores = vec![0.2, 0.7, 0.9];
        let result = remove_dominated_programs(&frontier, Some(&scores));

        let surviving: HashSet<ProgramIdx> = result.values().flatten().copied().collect();
        assert!(
            !surviving.contains(&0),
            "program 0 should be dominated and removed"
        );
        assert!(
            surviving.contains(&1),
            "program 1 should survive (unique on key 2)"
        );
        assert!(
            surviving.contains(&2),
            "program 2 should survive (unique on key 1)"
        );
    }

    #[test]
    fn remove_dominated_preserves_all_when_no_one_is_dominated() {
        // Each program is uniquely best on one key.
        let frontier = make_frontier(&[(0, &[0]), (1, &[1]), (2, &[2])]);
        let scores = vec![0.5, 0.6, 0.7];
        let result = remove_dominated_programs(&frontier, Some(&scores));

        let surviving: HashSet<ProgramIdx> = result.values().flatten().copied().collect();
        assert_eq!(surviving, HashSet::from([0, 1, 2]));
    }

    #[test]
    fn remove_dominated_programs_single_program_survives() {
        let frontier = make_frontier(&[(0, &[0])]);
        let scores = vec![0.5];
        let result = remove_dominated_programs(&frontier, Some(&scores));
        let surviving: HashSet<ProgramIdx> = result.values().flatten().copied().collect();
        assert_eq!(surviving, HashSet::from([0]));
    }

    // -----------------------------------------------------------------------
    // find_dominator_programs
    // -----------------------------------------------------------------------

    #[test]
    fn find_dominator_programs_returns_non_dominated_set() {
        // Key 0: {0, 1}, Key 1: {1, 2} — program 1 appears on both keys
        // so it can never be fully dominated. Programs 0 and 2 each appear
        // on only one key that also has 1, so they ARE dominated by 1.
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[1, 2])]);
        let scores = vec![0.2, 0.9, 0.5];
        let dominators = find_dominator_programs(&frontier, &scores);
        assert_eq!(dominators, vec![1], "only program 1 should survive");
    }

    // -----------------------------------------------------------------------
    // select_program_candidate_from_pareto_front
    // -----------------------------------------------------------------------

    #[test]
    fn selection_returns_valid_program_index() {
        // Program 1 covers 2 keys, program 0 covers 1 key.
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[1])]);
        let scores = vec![0.5, 0.9];
        let mut rng = StdRng::seed_from_u64(42);

        let selected = select_program_candidate_from_pareto_front(&frontier, &scores, &mut rng)
            .expect("selection should succeed");

        assert!(
            selected == 0 || selected == 1,
            "selected program must be in frontier"
        );
    }

    #[test]
    fn selection_is_deterministic_with_same_seed() {
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[1])]);
        let scores = vec![0.5, 0.9];

        let mut rng1 = StdRng::seed_from_u64(1234);
        let mut rng2 = StdRng::seed_from_u64(1234);

        let sel1 =
            select_program_candidate_from_pareto_front(&frontier, &scores, &mut rng1).unwrap();
        let sel2 =
            select_program_candidate_from_pareto_front(&frontier, &scores, &mut rng2).unwrap();

        assert_eq!(sel1, sel2, "same seed should produce same selection");
    }

    #[test]
    fn selection_on_empty_frontier_returns_error() {
        let frontier: HashMap<u32, HashSet<ProgramIdx>> = HashMap::new();
        let scores = vec![];
        let mut rng = StdRng::seed_from_u64(0);

        let result = select_program_candidate_from_pareto_front(&frontier, &scores, &mut rng);
        assert!(
            matches!(result, Err(GEPAError::EmptyFrontier)),
            "expected EmptyFrontier error"
        );
    }

    #[test]
    fn frequency_weighting_favours_high_coverage_program() {
        // Program 1 covers 5 keys; program 0 covers 1 key.
        // With enough samples, program 1 should be selected ~5x more often.
        let frontier_entries: Vec<(u32, &[ProgramIdx])> = vec![(0, &[0, 1])];
        // Extra keys covered only by program 1.
        let extra_fronts: Vec<(u32, Vec<ProgramIdx>)> =
            (1..5u32).map(|k| (k, vec![1usize])).collect();
        let frontier: HashMap<u32, HashSet<ProgramIdx>> = frontier_entries
            .iter()
            .map(|(k, idxs)| (*k, idxs.iter().copied().collect::<HashSet<_>>()))
            .chain(
                extra_fronts
                    .iter()
                    .map(|(k, idxs)| (*k, idxs.iter().copied().collect::<HashSet<_>>())),
            )
            .collect();

        let scores = vec![0.5, 0.9]; // program 1 is stronger
        let mut rng = StdRng::seed_from_u64(999);
        let mut count_1 = 0usize;
        let trials = 500;

        for _ in 0..trials {
            if select_program_candidate_from_pareto_front(&frontier, &scores, &mut rng).unwrap()
                == 1
            {
                count_1 += 1;
            }
        }

        // Program 1 has 5× more frequency; it should win >> 50 % of the time.
        // With 500 trials the probability of seeing < 60% is negligible.
        let fraction_1 = count_1 as f64 / f64::from(trials);
        assert!(
            fraction_1 > 0.60,
            "expected program 1 to dominate selection (got {fraction_1:.2})"
        );
    }

    // -----------------------------------------------------------------------
    // idxmax
    // -----------------------------------------------------------------------

    #[test]
    fn idxmax_returns_first_occurrence_of_max() {
        assert_eq!(idxmax(&[0.1, 0.9, 0.9]).unwrap(), 1);
        assert_eq!(idxmax(&[3.0, 1.0, 2.0]).unwrap(), 0);
    }

    #[test]
    fn idxmax_on_empty_returns_error() {
        assert!(matches!(idxmax(&[]), Err(GEPAError::NoCandidates)));
    }

    // ------------------------------------------------------------------
    // Gap 44: cascading domination — multi-pass handles indirect domination
    // ------------------------------------------------------------------

    #[test]
    fn test_cascading_domination() {
        // A is on keys 0 and 1 (score 0.9).
        // B is on keys 0 and 1 (score 0.5) — dominated by A.
        // C is on key 1 only (score 0.3) — dominated only because A covers key 1.
        //
        // After removing B (weakest first), C is still dominated by A on key 1.
        // After removing C, only A survives.
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[0, 1, 2])]);
        // scores: A(0)=0.9, B(1)=0.5, C(2)=0.3 — sorted weakest-first: C, B, A.
        let scores = vec![0.9_f64, 0.5, 0.3];
        let result = remove_dominated_programs(&frontier, Some(&scores));

        let surviving: HashSet<ProgramIdx> = result.values().flatten().copied().collect();
        // Only A (index 0) should survive.
        assert!(
            surviving.contains(&0),
            "A should survive (highest scorer, dominates all)"
        );
        assert!(!surviving.contains(&1), "B should be dominated by A");
        assert!(
            !surviving.contains(&2),
            "C should be dominated by A after B is removed"
        );
    }

    // ------------------------------------------------------------------
    // Gap 45: remove_dominated_programs with scores=None does not panic
    // ------------------------------------------------------------------

    #[test]
    fn test_remove_dominated_with_none_scores() {
        let frontier = make_frontier(&[(0, &[0, 1]), (1, &[0])]);
        // Program 1 is dominated by 0 (both on key 0, and key 1 only has 0).
        // With scores=None all programs are treated as equally scored (1.0).
        let result = remove_dominated_programs(&frontier, None);
        // Should not panic; program 0 must survive (has unique key 1).
        let surviving: HashSet<ProgramIdx> = result.values().flatten().copied().collect();
        assert!(
            surviving.contains(&0),
            "program 0 must survive (unique on key 1)"
        );
    }

    // ------------------------------------------------------------------
    // Gap 46: selection with all-zero scores returns a valid index
    // ------------------------------------------------------------------

    #[test]
    fn test_selection_with_all_zero_scores() {
        // All three programs have score 0.0.
        // select_program_candidate_from_pareto_front must not error and must
        // return a valid index.
        let frontier = make_frontier(&[(0, &[0, 1, 2])]);
        let scores = vec![0.0_f64, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(7);

        let selected = select_program_candidate_from_pareto_front(&frontier, &scores, &mut rng)
            .expect("should select even with all-zero scores");

        assert!(
            selected <= 2,
            "selected program {selected} must be a valid index (0..=2)"
        );
    }
}
