/// Merge Proposer — Algorithm 4 of the GEPA paper.
///
/// Combines two programs on the Pareto front that share a common ancestor.
/// The intuition: when two programs have each independently improved on the
/// ancestor in different ways (on different components), their improvements are
/// often complementary and can be combined.
///
/// Mirrors `gepa.proposer.merge`.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use tracing::debug;

use crate::core::adapter::{Candidate, GEPAAdapter};
use crate::core::data_loader::{DataId, DataLoader};
use crate::core::state::{GEPAState, ProgramIdx};
use crate::error::{GEPAError, Result};
use crate::proposer::base::CandidateProposal;
use crate::utils::pareto::find_dominator_programs;

// ---------------------------------------------------------------------------
// Type aliases (mirror the Python module)
// ---------------------------------------------------------------------------

/// `(i, j, ancestor)` triplet identifying an attempted merge.
type AncestorLog = (ProgramIdx, ProgramIdx, ProgramIdx);

/// `(i, j, new_prog_desc)` — uniquely identifies the *content* of a merged program.
type MergeDescription = (ProgramIdx, ProgramIdx, Vec<ProgramIdx>);

// ---------------------------------------------------------------------------
// Predicate helpers
// ---------------------------------------------------------------------------

/// Returns `true` when the `(ancestor, id1, id2)` triplet is merge-worthy.
///
/// A triplet is merge-worthy when at least one component satisfies the
/// "desirable predictor" condition: one child is identical to the ancestor
/// and the other is different.
fn does_triplet_have_desirable_predictors(
    candidates: &[Candidate],
    ancestor: ProgramIdx,
    id1: ProgramIdx,
    id2: ProgramIdx,
) -> bool {
    let pred_names: Vec<&String> = candidates[ancestor].keys().collect();
    for name in pred_names {
        let anc = candidates[ancestor].get(name).map_or("", String::as_str);
        let c1 = candidates[id1].get(name).map_or("", String::as_str);
        let c2 = candidates[id2].get(name).map_or("", String::as_str);
        // One child kept the ancestor value; the other changed it.
        if (anc == c1 || anc == c2) && c1 != c2 {
            return true;
        }
    }
    false
}

/// Walk the ancestry graph and collect all ancestors of `node`.
fn get_ancestors(node: ProgramIdx, parent_list: &[Vec<Option<ProgramIdx>>]) -> HashSet<ProgramIdx> {
    let mut visited = HashSet::new();
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        for &maybe_parent in &parent_list[n] {
            if let Some(parent) = maybe_parent
                && visited.insert(parent)
            {
                stack.push(parent);
            }
        }
    }
    visited
}

/// Filter `common_ancestors` to only those valid for the `(i, j)` pair.
fn filter_ancestors(
    i: ProgramIdx,
    j: ProgramIdx,
    common_ancestors: &HashSet<ProgramIdx>,
    merges_performed: &(Vec<AncestorLog>, Vec<MergeDescription>),
    agg_scores: &[f64],
    candidates: &[Candidate],
) -> Vec<ProgramIdx> {
    common_ancestors
        .iter()
        .filter(|&&ancestor| {
            // Skip already-attempted triplets.
            if merges_performed.0.contains(&(i, j, ancestor)) {
                return false;
            }
            // Skip if ancestor score exceeds either child (ancestor dominates).
            if agg_scores[ancestor] > agg_scores[i] || agg_scores[ancestor] > agg_scores[j] {
                return false;
            }
            // Skip if no desirable predictors.
            if !does_triplet_have_desirable_predictors(candidates, ancestor, i, j) {
                return false;
            }
            true
        })
        .copied()
        .collect()
}

/// Randomly sample a valid `(i, j, ancestor)` merge triplet.
fn find_common_ancestor_pair(
    rng: &mut StdRng,
    parent_list: &[Vec<Option<ProgramIdx>>],
    program_indexes: &[ProgramIdx],
    merges_performed: &(Vec<AncestorLog>, Vec<MergeDescription>),
    agg_scores: &[f64],
    candidates: &[Candidate],
    max_attempts: usize,
) -> Option<(ProgramIdx, ProgramIdx, ProgramIdx)> {
    if program_indexes.len() < 2 {
        return None;
    }

    for _ in 0..max_attempts {
        let mut pair: Vec<ProgramIdx> = program_indexes.choose_multiple(rng, 2).copied().collect();
        if pair.len() < 2 {
            continue;
        }
        // Canonical ordering.
        if pair[1] < pair[0] {
            pair.swap(0, 1);
        }
        let (i, j) = (pair[0], pair[1]);

        let ancestors_i = get_ancestors(i, parent_list);
        let ancestors_j = get_ancestors(j, parent_list);

        // Skip if one is a direct ancestor of the other.
        if ancestors_i.contains(&j) || ancestors_j.contains(&i) {
            continue;
        }

        let common: HashSet<ProgramIdx> = ancestors_i.intersection(&ancestors_j).copied().collect();
        let valid = filter_ancestors(i, j, &common, merges_performed, agg_scores, candidates);
        if valid.is_empty() {
            continue;
        }

        // Select ancestor weighted by aggregate score.
        let total_score: f64 = valid.iter().map(|&a| agg_scores[a]).sum();
        if total_score <= 0.0 {
            if let Some(&ancestor) = valid.choose(rng) {
                return Some((i, j, ancestor));
            }
            continue;
        }
        let mut cumulative = 0.0;
        let r: f64 = rng.r#gen::<f64>() * total_score;
        let mut chosen = valid[0];
        for &a in &valid {
            cumulative += agg_scores[a];
            if cumulative >= r {
                chosen = a;
                break;
            }
        }
        return Some((i, j, chosen));
    }

    None
}

/// Attempt to construct a merged program from two Pareto-front candidates.
///
/// Returns `(new_program, id1, id2, ancestor)` or `None`.
fn attempt_merge(
    rng: &mut StdRng,
    agg_scores: &[f64],
    merge_candidates: &[ProgramIdx],
    merges_performed: &mut (Vec<AncestorLog>, Vec<MergeDescription>),
    candidates: &[Candidate],
    parent_list: &[Vec<Option<ProgramIdx>>],
    has_val_support_overlap: Option<&dyn Fn(ProgramIdx, ProgramIdx) -> bool>,
    max_attempts: usize,
) -> Option<(Candidate, ProgramIdx, ProgramIdx, ProgramIdx)> {
    if merge_candidates.len() < 2 || parent_list.len() < 3 {
        return None;
    }

    for _ in 0..max_attempts {
        let triplet = find_common_ancestor_pair(
            rng,
            parent_list,
            merge_candidates,
            merges_performed,
            agg_scores,
            candidates,
            max_attempts,
        )?;

        let (id1, id2, ancestor) = triplet;

        if merges_performed.0.contains(&(id1, id2, ancestor)) {
            continue;
        }

        // Component-by-component merge.
        let pred_names: Vec<String> = candidates[ancestor].keys().cloned().collect();
        let mut new_program = candidates[ancestor].clone();
        let mut new_prog_desc: Vec<ProgramIdx> = Vec::new();

        for pred_name in &pred_names {
            let val_anc = candidates[ancestor]
                .get(pred_name)
                .map_or("", String::as_str);
            let val_id1 = candidates[id1].get(pred_name).map_or("", String::as_str);
            let val_id2 = candidates[id2].get(pred_name).map_or("", String::as_str);

            let chosen_idx = if (val_anc == val_id1 || val_anc == val_id2) && val_id1 != val_id2 {
                // One child kept ancestor; use the other's improvement.
                if val_anc == val_id1 { id2 } else { id1 }
            } else if val_anc != val_id1 && val_anc != val_id2 {
                // Both children changed this component; prefer higher-scorer.
                if agg_scores[id1] > agg_scores[id2] {
                    id1
                } else if agg_scores[id2] > agg_scores[id1] {
                    id2
                } else {
                    *[id1, id2].choose(rng).unwrap()
                }
            } else {
                // Both same value.
                id1
            };

            new_program.insert(
                pred_name.clone(),
                candidates[chosen_idx]
                    .get(pred_name)
                    .cloned()
                    .unwrap_or_default(),
            );
            new_prog_desc.push(chosen_idx);
        }

        // De-duplicate by content.
        let desc = (id1, id2, new_prog_desc.clone());
        if merges_performed.1.contains(&desc) {
            continue;
        }

        // Optional: require overlapping validation coverage.
        if let Some(checker) = has_val_support_overlap
            && !checker(id1, id2)
        {
            continue;
        }

        merges_performed.1.push(desc);
        return Some((new_program, id1, id2, ancestor));
    }

    None
}

// ---------------------------------------------------------------------------
// MergeProposer
// ---------------------------------------------------------------------------

/// Propose merged candidates from two Pareto-front programs (Algorithm 4).
pub struct MergeProposer<Id, Item, T, RO>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + 'static,
{
    /// Validation dataset loader (for subsample eval).
    pub valset: Arc<dyn DataLoader<Id, Item>>,
    /// Adapter for evaluating the merged candidate.
    pub adapter: Arc<dyn GEPAAdapter<Item, T, RO>>,
    /// Master switch: when `false`, `propose()` always returns `None`.
    pub use_merge: bool,
    /// Maximum total number of successful merge attempts.
    pub max_merge_invocations: usize,
    /// Minimum number of shared validation IDs between two parents.
    pub val_overlap_floor: usize,
    rng: StdRng,
    /// Pending merge invocations scheduled by the engine.
    pub(crate) merges_due: usize,
    /// Total merge attempts so far.
    pub(crate) total_merges_tested: usize,
    /// Running log of performed merges (ancestor-level and content-level dedup).
    merges_performed: (Vec<AncestorLog>, Vec<MergeDescription>),
    /// Set by the engine after each successful reflective mutation.
    pub(crate) last_iter_found_new_program: bool,
}

impl<Id, Item, T, RO> MergeProposer<Id, Item, T, RO>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + 'static,
{
    /// Construct a new `MergeProposer`.
    ///
    /// # Errors
    /// Returns `Err` when `val_overlap_floor == 0`.
    pub fn new(
        valset: Arc<dyn DataLoader<Id, Item>>,
        adapter: Arc<dyn GEPAAdapter<Item, T, RO>>,
        use_merge: bool,
        max_merge_invocations: usize,
        val_overlap_floor: usize,
        seed: u64,
    ) -> Result<Self> {
        if val_overlap_floor == 0 {
            return Err(GEPAError::Config(
                "val_overlap_floor must be a positive integer".into(),
            ));
        }
        Ok(Self {
            valset,
            adapter,
            use_merge,
            max_merge_invocations,
            val_overlap_floor,
            rng: StdRng::seed_from_u64(seed),
            merges_due: 0,
            total_merges_tested: 0,
            merges_performed: (Vec::new(), Vec::new()),
            last_iter_found_new_program: false,
        })
    }

    /// Increment the merge-due counter if the budget allows.
    ///
    /// Called by the engine after each successful reflective mutation.
    pub fn schedule_if_needed(&mut self) {
        if self.use_merge && self.total_merges_tested < self.max_merge_invocations {
            self.merges_due += 1;
        }
    }

    /// Select a balanced subsample from the shared validation IDs of two parents.
    fn select_eval_subsample(
        &mut self,
        scores1: &HashMap<Id, f64>,
        scores2: &HashMap<Id, f64>,
        num_subsample: usize,
    ) -> Vec<Id> {
        let common_ids: Vec<Id> = scores1
            .keys()
            .filter(|k| scores2.contains_key(k))
            .cloned()
            .collect();

        let bucket_p1: Vec<Id> = common_ids
            .iter()
            .filter(|k| scores1[k] > scores2[k])
            .cloned()
            .collect();
        let bucket_p2: Vec<Id> = common_ids
            .iter()
            .filter(|k| scores2[k] > scores1[k])
            .cloned()
            .collect();
        let bucket_tie: Vec<Id> = common_ids
            .iter()
            .filter(|k| scores1[k] == scores2[k])
            .cloned()
            .collect();

        let n_each = num_subsample.div_ceil(3).max(1);
        let mut selected: Vec<Id> = Vec::new();
        let mut selected_set: HashSet<Id> = HashSet::new();

        for bucket in [&bucket_p1, &bucket_p2, &bucket_tie] {
            if selected.len() >= num_subsample {
                break;
            }
            let available: Vec<Id> = bucket
                .iter()
                .filter(|k| !selected_set.contains(k))
                .cloned()
                .collect();
            let take = available
                .len()
                .min(n_each)
                .min(num_subsample - selected.len());
            if take > 0 {
                let chosen: Vec<Id> = available
                    .choose_multiple(&mut self.rng, take)
                    .cloned()
                    .collect();
                for id in chosen {
                    selected_set.insert(id.clone());
                    selected.push(id);
                }
            }
        }

        // Fill remaining from all common IDs.
        let remaining = num_subsample.saturating_sub(selected.len());
        if remaining > 0 {
            let unused: Vec<Id> = common_ids
                .iter()
                .filter(|k| !selected_set.contains(k))
                .cloned()
                .collect();
            let take = unused.len().min(remaining);
            if take > 0 {
                let fill: Vec<Id> = unused
                    .choose_multiple(&mut self.rng, take)
                    .cloned()
                    .collect();
                for id in fill {
                    selected.push(id);
                }
            }
        }

        selected.truncate(num_subsample);
        selected
    }

    /// Main mutable propose implementation.
    pub async fn propose_mut(
        &mut self,
        state: &mut GEPAState<Id>,
    ) -> Result<Option<CandidateProposal<Id>>> {
        let iter_i = state.i.wrapping_add(1);

        if !(self.use_merge && self.last_iter_found_new_program && self.merges_due > 0) {
            debug!(
                iteration = iter_i,
                "no merge scheduled or preconditions not met"
            );
            return Ok(None);
        }

        let agg_scores = state.program_full_scores_val_set();

        // Build usize-keyed pareto mapping for find_dominator_programs.
        let pareto_front = state.get_pareto_front_mapping();
        let usize_mapping: HashMap<usize, HashSet<usize>> =
            pareto_front.into_values().enumerate().collect();

        let merge_candidates = find_dominator_programs(&usize_mapping, &agg_scores);

        let has_overlap = |id1: ProgramIdx, id2: ProgramIdx| -> bool {
            let common = state.prog_candidate_val_subscores[id1]
                .keys()
                .filter(|k| state.prog_candidate_val_subscores[id2].contains_key(k))
                .count();
            common >= self.val_overlap_floor
        };

        let merge_output = attempt_merge(
            &mut self.rng,
            &agg_scores,
            &merge_candidates,
            &mut self.merges_performed,
            &state.program_candidates,
            &state.parent_program_for_candidate,
            Some(&has_overlap),
            10,
        );

        let Some((new_program, id1, id2, ancestor)) = merge_output else {
            debug!(iteration = iter_i, "no valid merge candidates found");
            return Ok(None);
        };

        // Record this triplet (ancestor-level dedup).
        self.merges_performed.0.push((id1, id2, ancestor));

        debug!(iteration = iter_i, id1, id2, ancestor, "attempting merge");

        // Select subsample from shared validation support.
        let subsample_ids = self.select_eval_subsample(
            &state.prog_candidate_val_subscores[id1],
            &state.prog_candidate_val_subscores[id2],
            5,
        );

        if subsample_ids.is_empty() {
            debug!(
                iteration = iter_i,
                "insufficient overlapping validation coverage — skipping merge"
            );
            return Ok(None);
        }

        let id1_sub_scores: Vec<f64> = subsample_ids
            .iter()
            .map(|k| {
                state.prog_candidate_val_subscores[id1]
                    .get(k)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        let id2_sub_scores: Vec<f64> = subsample_ids
            .iter()
            .map(|k| {
                state.prog_candidate_val_subscores[id2]
                    .get(k)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();

        // Evaluate merged program on subsample.
        let mini_devset = self.valset.fetch(&subsample_ids)?;
        let eval_new = self
            .adapter
            .evaluate(&mini_devset, &new_program, false)
            .await
            .map_err(|e| GEPAError::Evaluation(e.to_string()))?;

        state.increment_evals(subsample_ids.len());

        let new_sub_scores = eval_new.scores.clone();
        let new_sum: f64 = new_sub_scores.iter().sum();
        let p1_sum: f64 = id1_sub_scores.iter().sum();
        let p2_sum: f64 = id2_sub_scores.iter().sum();

        debug!(
            iteration = iter_i,
            new_sum, p1_sum, p2_sum, "evaluated merged candidate"
        );

        // Acceptance criterion: new_sum >= max(parent_sums).
        Ok(Some(
            CandidateProposal::new(new_program, vec![id1, id2])
                .with_tag("merge")
                .with_subsample(subsample_ids, vec![p1_sum, p2_sum], new_sub_scores)
                .with_metadata("ancestor", serde_json::json!(ancestor)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adapter::Candidate;

    // ------------------------------------------------------------------
    // Gap 22: deep chain ancestry (depth 5)
    // ------------------------------------------------------------------

    #[test]
    fn test_find_common_ancestor_deep_chain() {
        // Build chain: 0 → 1 → 2 → 3 → 4
        // parent_list[i] = vec![Some(i-1)] for i>0, vec![None] for 0.
        let parent_list: Vec<Vec<Option<usize>>> = vec![
            vec![None],    // 0: root
            vec![Some(0)], // 1
            vec![Some(1)], // 2
            vec![Some(2)], // 3
            vec![Some(3)], // 4
        ];

        // Candidates 3 and 4 both descend from 0.
        let anc3 = get_ancestors(3, &parent_list);
        let anc4 = get_ancestors(4, &parent_list);

        // 0 must appear in both ancestor sets.
        assert!(anc3.contains(&0), "3's ancestors should include 0");
        assert!(anc4.contains(&0), "4's ancestors should include 0");

        // The common ancestors of 3 and 4 include 0, 1, and 2.
        let common: HashSet<usize> = anc3.intersection(&anc4).copied().collect();
        assert!(
            common.contains(&0),
            "0 should be a common ancestor of 3 and 4"
        );
        assert!(
            common.contains(&1),
            "1 should be a common ancestor of 3 and 4"
        );
        assert!(
            common.contains(&2),
            "2 should be a common ancestor of 3 and 4"
        );
    }

    // ------------------------------------------------------------------
    // Gap 23: ancestor filtered when score higher than children
    // ------------------------------------------------------------------

    #[test]
    fn test_ancestor_filtered_when_score_higher_than_children() {
        // ancestor=0 has score 0.9 (higher than both children 1 and 2 with 0.5).
        // filter_ancestors should return empty because ancestor dominates both.
        let mut anc = Candidate::new();
        anc.insert("a".into(), "v0".into());
        let mut c1 = Candidate::new();
        c1.insert("a".into(), "v1".into());
        let mut c2 = Candidate::new();
        c2.insert("a".into(), "v0".into()); // c2 same as ancestor so desirable predicate is met for i,j
        // Actually make sure desirable predicate IS met so only the score filter removes it.
        c2.insert("b".into(), "w2".into());
        let mut anc2 = anc.clone();
        anc2.insert("b".into(), "w0".into());
        let mut c1b = c1.clone();
        c1b.insert("b".into(), "w0".into()); // c1 same as ancestor on b

        let candidates = vec![anc2, c1b, c2];
        // agg_scores: ancestor(0)=0.9, child1(1)=0.5, child2(2)=0.7.
        let agg_scores = vec![0.9_f64, 0.5, 0.7];

        let common: HashSet<usize> = [0].into();
        let merges_performed = (Vec::new(), Vec::new());
        // i=1, j=2, common ancestor=0 but score[0]=0.9 > score[1]=0.5 → filtered out.
        let valid = filter_ancestors(1, 2, &common, &merges_performed, &agg_scores, &candidates);
        assert!(
            valid.is_empty(),
            "ancestor with higher score than both children should be filtered out"
        );
    }

    // ------------------------------------------------------------------
    // Gap 24: both children changed — higher scorer wins
    // ------------------------------------------------------------------

    #[test]
    fn test_both_children_changed_uses_higher_scorer() {
        // To trigger the "both children changed" branch, both c1 and c2 must
        // differ from the ancestor on component "a".  To also satisfy the
        // desirable-predictor requirement (needed for the merge to proceed),
        // we add a second component "b" where c1 keeps the ancestor value and
        // c2 diverges.  That makes the triplet merge-worthy.
        //
        // ancestor:  a="v0", b="w0"
        // child1:    a="v1", b="w0"   ← a changed, b same as ancestor
        // child2:    a="v2", b="w1"   ← a changed, b changed
        //
        // For component "a":   val_anc(v0) != val_id1(v1) and val_anc(v0) != val_id2(v2)
        //                       → both diverged → pick higher-scoring child (id1, score 0.8).
        // For component "b":   val_anc(w0) == val_id1(w0) → desirable predictor; pick id2.
        let mut rng = StdRng::seed_from_u64(0);

        let mut anc = Candidate::new();
        anc.insert("a".into(), "v0".into());
        anc.insert("b".into(), "w0".into());

        let mut c1 = Candidate::new();
        c1.insert("a".into(), "v1".into()); // changed
        c1.insert("b".into(), "w0".into()); // same as ancestor (desirable predicate for b)

        let mut c2 = Candidate::new();
        c2.insert("a".into(), "v2".into()); // changed
        c2.insert("b".into(), "w1".into()); // changed

        let candidates = vec![anc, c1, c2];
        // agg_scores: ancestor(0)=0.4 (below both children), c1(1)=0.8, c2(2)=0.6.
        let agg_scores = vec![0.4_f64, 0.8, 0.6];

        let parent_list: Vec<Vec<Option<usize>>> = vec![vec![None], vec![Some(0)], vec![Some(0)]];
        let mut merges_performed: (Vec<AncestorLog>, Vec<MergeDescription>) =
            (Vec::new(), Vec::new());

        let checker: &dyn Fn(usize, usize) -> bool = &|_, _| true;

        let result = attempt_merge(
            &mut rng,
            &agg_scores,
            &[1, 2],
            &mut merges_performed,
            &candidates,
            &parent_list,
            Some(checker),
            20,
        );

        let (merged, id1, id2, _ancestor) = result.expect("should produce a merge");
        // id1 and id2 should be 1 and 2 in canonical order.
        assert_eq!((id1, id2), (1, 2));
        // For component "a" (both diverged), higher-scoring child1 (score 0.8) wins.
        assert_eq!(
            merged.get("a").map(String::as_str),
            Some("v1"),
            "higher-scoring child's value (v1) should be used for the both-changed component 'a'"
        );
    }

    // ------------------------------------------------------------------
    // Gap 25: content dedup prevents duplicate merges
    // ------------------------------------------------------------------

    #[test]
    fn test_content_dedup_prevents_duplicate_merges() {
        let mut rng = StdRng::seed_from_u64(42);

        let mut anc = Candidate::new();
        anc.insert("a".into(), "v0".into());
        let mut c1 = Candidate::new();
        c1.insert("a".into(), "v0".into()); // same as ancestor
        let mut c2 = Candidate::new();
        c2.insert("a".into(), "v1".into()); // changed

        let candidates = vec![anc, c1, c2];
        let agg_scores = vec![0.3_f64, 0.5, 0.7];
        let parent_list: Vec<Vec<Option<usize>>> = vec![vec![None], vec![Some(0)], vec![Some(0)]];
        let checker: &dyn Fn(usize, usize) -> bool = &|_, _| true;
        let mut merges_performed: (Vec<AncestorLog>, Vec<MergeDescription>) =
            (Vec::new(), Vec::new());

        // First call should succeed.
        let first = attempt_merge(
            &mut rng,
            &agg_scores,
            &[1, 2],
            &mut merges_performed,
            &candidates,
            &parent_list,
            Some(checker),
            20,
        );
        assert!(first.is_some(), "first merge should succeed");

        // Second call with the same candidates should be deduped (content already recorded).
        // Because the only possible merged content is already in merges_performed.1,
        // attempt_merge should return None.
        let second = attempt_merge(
            &mut rng,
            &agg_scores,
            &[1, 2],
            &mut merges_performed,
            &candidates,
            &parent_list,
            Some(checker),
            20,
        );
        assert!(
            second.is_none(),
            "second identical merge should be deduped and return None"
        );
    }

    // ------------------------------------------------------------------
    // Gap 26: select_eval_subsample with unbalanced buckets
    // ------------------------------------------------------------------

    #[test]
    fn test_select_eval_subsample_unbalanced_buckets() {
        use crate::core::adapter::{EvaluationBatch, GEPAAdapter, ReflectiveDataset};
        use crate::core::data_loader::VecLoader;
        use std::sync::Arc;

        struct DummyAdapter;
        #[async_trait::async_trait]
        impl GEPAAdapter<String, (), String> for DummyAdapter {
            async fn evaluate(
                &self,
                b: &[String],
                _c: &Candidate,
                _t: bool,
            ) -> crate::error::Result<EvaluationBatch<(), String>> {
                Ok(EvaluationBatch::new(b.to_vec(), vec![0.5; b.len()]))
            }
            async fn make_reflective_dataset(
                &self,
                _c: &Candidate,
                _e: &EvaluationBatch<(), String>,
                comps: &[String],
            ) -> crate::error::Result<ReflectiveDataset> {
                Ok(comps.iter().map(|n| (n.clone(), vec![])).collect())
            }
        }

        let valset: Arc<dyn crate::core::data_loader::DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["a".to_string(), "b".to_string()]));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(DummyAdapter);
        let mut proposer =
            MergeProposer::new(valset, adapter, true, 5, 1, 7).expect("should build");

        // scores1 wins on all IDs (bucket_p1 = full, bucket_p2 = empty).
        let mut scores1: HashMap<usize, f64> = HashMap::new();
        scores1.insert(0, 0.9);
        scores1.insert(1, 0.9);
        scores1.insert(2, 0.9);
        scores1.insert(3, 0.9);
        let mut scores2: HashMap<usize, f64> = HashMap::new();
        scores2.insert(0, 0.1);
        scores2.insert(1, 0.1);
        scores2.insert(2, 0.1);
        scores2.insert(3, 0.1);

        let subsample = proposer.select_eval_subsample(&scores1, &scores2, 3);
        // Should return exactly 3 items even though bucket_p2 is empty.
        assert_eq!(
            subsample.len(),
            3,
            "subsample should still return requested count"
        );
        // All items must come from the common keys.
        for id in &subsample {
            assert!(
                scores1.contains_key(id),
                "sampled id {id} must be in common set"
            );
        }
    }

    // ------------------------------------------------------------------
    // Gap 27: select_eval_subsample with no overlapping IDs
    // ------------------------------------------------------------------

    #[test]
    fn test_select_eval_subsample_no_overlap() {
        use crate::core::adapter::{EvaluationBatch, GEPAAdapter, ReflectiveDataset};
        use crate::core::data_loader::VecLoader;
        use std::sync::Arc;

        struct DummyAdapter2;
        #[async_trait::async_trait]
        impl GEPAAdapter<String, (), String> for DummyAdapter2 {
            async fn evaluate(
                &self,
                b: &[String],
                _c: &Candidate,
                _t: bool,
            ) -> crate::error::Result<EvaluationBatch<(), String>> {
                Ok(EvaluationBatch::new(b.to_vec(), vec![0.5; b.len()]))
            }
            async fn make_reflective_dataset(
                &self,
                _c: &Candidate,
                _e: &EvaluationBatch<(), String>,
                comps: &[String],
            ) -> crate::error::Result<ReflectiveDataset> {
                Ok(comps.iter().map(|n| (n.clone(), vec![])).collect())
            }
        }

        let valset: Arc<dyn crate::core::data_loader::DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["x".to_string()]));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(DummyAdapter2);
        let mut proposer =
            MergeProposer::new(valset, adapter, true, 5, 1, 9).expect("should build");

        // Parent 1 covers IDs 0,1; parent 2 covers IDs 2,3 — no overlap.
        let mut scores1: HashMap<usize, f64> = HashMap::new();
        scores1.insert(0, 0.5);
        scores1.insert(1, 0.5);
        let mut scores2: HashMap<usize, f64> = HashMap::new();
        scores2.insert(2, 0.5);
        scores2.insert(3, 0.5);

        let subsample = proposer.select_eval_subsample(&scores1, &scores2, 3);
        // No common IDs → subsample must be empty.
        assert!(
            subsample.is_empty(),
            "disjoint val ID sets should yield empty subsample"
        );
    }

    // ------------------------------------------------------------------
    // Gap 28: propose_mut returns Ok(None) when use_merge=false
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_propose_mut_use_merge_false() {
        use crate::core::adapter::{EvaluationBatch, GEPAAdapter, ReflectiveDataset};
        use crate::core::data_loader::VecLoader;
        use crate::core::state::{FrontierType, ValsetEvaluation};
        use std::sync::Arc;

        struct DummyAdapter3;
        #[async_trait::async_trait]
        impl GEPAAdapter<String, (), String> for DummyAdapter3 {
            async fn evaluate(
                &self,
                b: &[String],
                _c: &Candidate,
                _t: bool,
            ) -> crate::error::Result<EvaluationBatch<(), String>> {
                Ok(EvaluationBatch::new(b.to_vec(), vec![0.5; b.len()]))
            }
            async fn make_reflective_dataset(
                &self,
                _c: &Candidate,
                _e: &EvaluationBatch<(), String>,
                comps: &[String],
            ) -> crate::error::Result<ReflectiveDataset> {
                Ok(comps.iter().map(|n| (n.clone(), vec![])).collect())
            }
        }

        let valset: Arc<dyn crate::core::data_loader::DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["a".to_string()]));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(DummyAdapter3);
        // use_merge=false: propose_mut must always return Ok(None).
        let mut proposer =
            MergeProposer::new(valset, adapter, false, 10, 1, 1).expect("should build");
        proposer.merges_due = 99;
        proposer.last_iter_found_new_program = true;

        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "test".into());
        let eval = ValsetEvaluation::from_vecs(
            vec![0usize],
            vec![serde_json::json!("out")],
            vec![0.5],
            None,
        );
        let mut state =
            crate::core::state::GEPAState::new(seed, eval, FrontierType::Instance, None)
                .expect("should construct");

        let result = proposer
            .propose_mut(&mut state)
            .await
            .expect("should not error");
        assert!(
            result.is_none(),
            "use_merge=false must always return Ok(None)"
        );
    }

    // ------------------------------------------------------------------
    // Gap 29: merge with 3-component system resolves each correctly
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_three_component_system() {
        let mut rng = StdRng::seed_from_u64(99);

        // ancestor: a=v0, b=w0, c=u0
        // child1:   a=v0, b=w1, c=u0  (changed only b)
        // child2:   a=v1, b=w0, c=u1  (changed only a and c)
        // Expected merge: a=v1 (from child2, only child2 changed it),
        //                 b=w1 (from child1, only child1 changed it),
        //                 c=u1 (from child2, only child2 changed it).
        let mut anc = Candidate::new();
        anc.insert("a".into(), "v0".into());
        anc.insert("b".into(), "w0".into());
        anc.insert("c".into(), "u0".into());

        let mut c1 = Candidate::new();
        c1.insert("a".into(), "v0".into()); // same as ancestor
        c1.insert("b".into(), "w1".into()); // changed
        c1.insert("c".into(), "u0".into()); // same as ancestor

        let mut c2 = Candidate::new();
        c2.insert("a".into(), "v1".into()); // changed
        c2.insert("b".into(), "w0".into()); // same as ancestor
        c2.insert("c".into(), "u1".into()); // changed

        let candidates = vec![anc, c1, c2];
        let agg_scores = vec![0.3_f64, 0.6, 0.7];
        let parent_list: Vec<Vec<Option<usize>>> = vec![vec![None], vec![Some(0)], vec![Some(0)]];
        let checker: &dyn Fn(usize, usize) -> bool = &|_, _| true;
        let mut merges_performed: (Vec<AncestorLog>, Vec<MergeDescription>) =
            (Vec::new(), Vec::new());

        let result = attempt_merge(
            &mut rng,
            &agg_scores,
            &[1, 2],
            &mut merges_performed,
            &candidates,
            &parent_list,
            Some(checker),
            20,
        );

        let (merged, _id1, _id2, _ancestor) = result.expect("3-component merge should succeed");
        assert_eq!(
            merged.get("a").map(String::as_str),
            Some("v1"),
            "component 'a' should come from child2 (only child that changed it)"
        );
        assert_eq!(
            merged.get("b").map(String::as_str),
            Some("w1"),
            "component 'b' should come from child1 (only child that changed it)"
        );
        assert_eq!(
            merged.get("c").map(String::as_str),
            Some("u1"),
            "component 'c' should come from child2 (only child that changed it)"
        );
    }

    #[test]
    fn desirable_predictors_detected() {
        let mut anc = Candidate::new();
        anc.insert("a".into(), "v1".into());
        anc.insert("b".into(), "w1".into());

        let mut c1 = Candidate::new();
        c1.insert("a".into(), "v1".into()); // same as ancestor
        c1.insert("b".into(), "w2".into()); // changed

        let mut c2 = Candidate::new();
        c2.insert("a".into(), "v2".into()); // changed
        c2.insert("b".into(), "w1".into()); // same as ancestor

        let candidates = vec![anc, c1, c2];
        assert!(does_triplet_have_desirable_predictors(&candidates, 0, 1, 2));
    }

    #[test]
    fn no_desirable_predictors_when_both_changed() {
        let mut anc = Candidate::new();
        anc.insert("a".into(), "v1".into());

        let mut c1 = Candidate::new();
        c1.insert("a".into(), "v2".into()); // both changed

        let mut c2 = Candidate::new();
        c2.insert("a".into(), "v3".into()); // both changed

        let candidates = vec![anc, c1, c2];
        // Both children changed 'a' from the ancestor, but they differ from
        // each other — this actually counts as "both diverged", not desirable.
        // With c1="v2" and c2="v3": anc!= c1, anc != c2, c1 != c2 → not desirable.
        assert!(!does_triplet_have_desirable_predictors(
            &candidates,
            0,
            1,
            2
        ));
    }

    #[test]
    fn get_ancestors_walks_chain() {
        // 0 → 1 → 2 (0 is the seed)
        let parent_list: Vec<Vec<Option<usize>>> = vec![
            vec![None],    // 0: seed
            vec![Some(0)], // 1: child of 0
            vec![Some(1)], // 2: child of 1
        ];
        let ancestors = get_ancestors(2, &parent_list);
        assert!(ancestors.contains(&1));
        assert!(ancestors.contains(&0));
        assert_eq!(ancestors.len(), 2);
    }

    #[test]
    fn merge_proposer_zero_overlap_floor_errors() {
        use crate::core::adapter::{EvaluationBatch, GEPAAdapter, ReflectiveDataset};
        use crate::core::data_loader::VecLoader;
        use std::sync::Arc;

        struct DummyAdapter;
        #[async_trait::async_trait]
        impl GEPAAdapter<String, (), String> for DummyAdapter {
            async fn evaluate(
                &self,
                b: &[String],
                _c: &Candidate,
                _t: bool,
            ) -> Result<EvaluationBatch<(), String>> {
                Ok(EvaluationBatch::new(b.to_vec(), vec![0.5; b.len()]))
            }
            async fn make_reflective_dataset(
                &self,
                _c: &Candidate,
                _e: &EvaluationBatch<(), String>,
                comps: &[String],
            ) -> Result<ReflectiveDataset> {
                Ok(comps.iter().map(|n| (n.clone(), vec![])).collect())
            }
        }

        let valset: Arc<dyn DataLoader<usize, String>> =
            Arc::new(VecLoader::new(vec!["a".to_string()]));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(DummyAdapter);
        let result = MergeProposer::new(valset, adapter, true, 5, 0, 42);
        assert!(result.is_err(), "val_overlap_floor=0 should error");
    }
}
