/// Minibatch sampling strategies for GEPA's reflective mutation step.
///
/// Each iteration of the reflective mutation proposer draws a small minibatch
/// from the training set to evaluate the current candidate with trace capture.
///
/// The default [`EpochShuffledSampler`] shuffles all training IDs once per
/// epoch and serves consecutive non-overlapping windows of size
/// `minibatch_size`.  This guarantees every training example is seen roughly
/// equally over time while keeping each batch small enough for fast feedback.
///
/// Mirrors `gepa.strategies.batch_sampler`.
use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::core::data_loader::DataId;
use crate::core::state::BEFORE_FIRST_ITERATION;
use crate::core::state::GEPAState;
use crate::error::{GEPAError, Result};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Strategy for drawing a minibatch of training IDs each GEPA iteration.
pub trait BatchSampler<Id: DataId>: Send + Sync {
    /// Return the next minibatch of example IDs.
    ///
    /// # Arguments
    /// * `all_ids`   — ordered list of all training IDs.
    /// * `state`     — current optimizer state (provides `state.i`).
    ///
    /// # Errors
    /// Returns [`GEPAError::EmptyDataset`] when `all_ids` is empty.
    fn next_batch_ids(&mut self, all_ids: &[Id], state: &GEPAState<Id>) -> Result<Vec<Id>>;
}

// ---------------------------------------------------------------------------
// EpochShuffledSampler
// ---------------------------------------------------------------------------

/// Epoch-shuffled minibatch sampler (paper default).
///
/// At the start of each epoch the full training set is shuffled and padded to
/// an exact multiple of `minibatch_size` by appending the least-frequently
/// seen example(s).  Consecutive calls serve successive non-overlapping
/// windows.
///
/// This ensures that over a full epoch every training example is evaluated at
/// least once, while still keeping individual batches small.
pub struct EpochShuffledSampler<Id: DataId> {
    minibatch_size: usize,
    rng: StdRng,
    shuffled_ids: Vec<Id>,
    epoch: Option<usize>,
    id_freqs: HashMap<Id, usize>,
    last_trainset_size: usize,
}

impl<Id: DataId> EpochShuffledSampler<Id> {
    /// Construct with the given `minibatch_size` and RNG `seed`.
    ///
    /// # Errors
    /// Returns `Err` when `minibatch_size == 0`.
    pub fn new(minibatch_size: usize, seed: u64) -> Result<Self> {
        if minibatch_size == 0 {
            return Err(GEPAError::Config(
                "minibatch_size must be at least 1".into(),
            ));
        }
        Ok(Self {
            minibatch_size,
            rng: StdRng::seed_from_u64(seed),
            shuffled_ids: Vec::new(),
            epoch: None,
            id_freqs: HashMap::new(),
            last_trainset_size: 0,
        })
    }

    /// Shuffle and pad the full ID list for a new epoch.
    fn update_shuffled(&mut self, all_ids: &[Id]) {
        let trainset_size = all_ids.len();
        self.last_trainset_size = trainset_size;

        if trainset_size == 0 {
            self.shuffled_ids.clear();
            self.id_freqs.clear();
            return;
        }

        self.shuffled_ids = all_ids.to_vec();
        self.shuffled_ids.shuffle(&mut self.rng);

        self.id_freqs = self
            .shuffled_ids
            .iter()
            .fold(HashMap::new(), |mut acc, id| {
                *acc.entry(id.clone()).or_insert(0) += 1;
                acc
            });

        // Pad to an exact multiple of `minibatch_size` using least-frequent IDs.
        let remainder = trainset_size % self.minibatch_size;
        if remainder != 0 {
            let num_to_pad = self.minibatch_size - remainder;
            for _ in 0..num_to_pad {
                // Pick the least-frequently selected ID.
                let least = self
                    .id_freqs
                    .iter()
                    .min_by_key(|(_, count)| **count)
                    .map(|(id, _)| id.clone())
                    .expect("id_freqs is non-empty");
                self.shuffled_ids.push(least.clone());
                *self.id_freqs.entry(least).or_insert(0) += 1;
            }
        }
    }
}

impl<Id: DataId> BatchSampler<Id> for EpochShuffledSampler<Id> {
    fn next_batch_ids(&mut self, all_ids: &[Id], state: &GEPAState<Id>) -> Result<Vec<Id>> {
        let trainset_size = all_ids.len();
        if trainset_size == 0 {
            return Err(GEPAError::EmptyDataset(
                "Cannot sample a minibatch from an empty loader".into(),
            ));
        }

        // Convert the `usize::MAX` sentinel to 0 so epoch math is well-defined.
        let iter_i = if state.i == BEFORE_FIRST_ITERATION {
            0
        } else {
            state.i
        };

        let base_idx = iter_i.saturating_mul(self.minibatch_size);
        let curr_epoch = if self.epoch.is_none() || self.shuffled_ids.is_empty() {
            0
        } else {
            let len = self.shuffled_ids.len().max(1);
            base_idx / len
        };

        let needs_refresh = self.epoch.is_none()
            || self.shuffled_ids.is_empty()
            || trainset_size != self.last_trainset_size
            || curr_epoch > self.epoch.unwrap_or(0);

        if needs_refresh {
            self.epoch = Some(curr_epoch);
            self.update_shuffled(all_ids);
        }

        let len = self.shuffled_ids.len();
        debug_assert!(len >= self.minibatch_size);
        debug_assert_eq!(len % self.minibatch_size, 0);

        let window_start = base_idx % len;
        let window_end = window_start + self.minibatch_size;
        debug_assert!(window_end <= len);

        Ok(self.shuffled_ids[window_start..window_end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adapter::Candidate;
    use crate::core::state::{FrontierType, ValsetEvaluation};

    fn make_state_at(i: usize) -> GEPAState<usize> {
        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "test".into());
        let eval = ValsetEvaluation::from_vecs(
            vec![0usize],
            vec![serde_json::json!("out")],
            vec![0.5],
            None,
        );
        let mut state =
            GEPAState::new(seed, eval, FrontierType::Instance, None).expect("should construct");
        // Simulate the engine advancing the iteration counter.
        state.i = i;
        state
    }

    #[test]
    fn sampler_returns_correct_batch_size() {
        let state = make_state_at(0);
        let all_ids: Vec<usize> = (0..10).collect();
        let mut sampler = EpochShuffledSampler::new(3, 42).expect("construction ok");
        let batch = sampler
            .next_batch_ids(&all_ids, &state)
            .expect("should sample");
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn sampler_empty_dataset_errors() {
        let state = make_state_at(0);
        let mut sampler = EpochShuffledSampler::new(3, 0).expect("construction ok");
        let result = sampler.next_batch_ids(&[], &state);
        assert!(result.is_err());
    }

    #[test]
    fn sampler_covers_all_ids_over_one_epoch() {
        let all_ids: Vec<usize> = (0..6).collect();
        let mut sampler = EpochShuffledSampler::new(2, 99).expect("construction ok");
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for iter in 0..3 {
            let state = make_state_at(iter);
            let batch = sampler
                .next_batch_ids(&all_ids, &state)
                .expect("should sample");
            seen.extend(batch);
        }
        // All 6 IDs should have been seen in 3 batches of size 2.
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn sampler_non_divisible_size_still_returns_full_batch() {
        // trainset_size=5, minibatch_size=3 → padded to 6.
        let all_ids: Vec<usize> = (0..5).collect();
        let mut sampler = EpochShuffledSampler::new(3, 7).expect("construction ok");
        let state = make_state_at(0);
        let batch = sampler
            .next_batch_ids(&all_ids, &state)
            .expect("should sample");
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn sampler_construction_rejects_zero_size() {
        assert!(EpochShuffledSampler::<usize>::new(0, 0).is_err());
    }

    // ------------------------------------------------------------------
    // Gap 36: epoch boundary reshuffles when window crosses epoch boundary
    // ------------------------------------------------------------------

    #[test]
    fn test_epoch_boundary_reshuffles() {
        // minibatch_size=2, 4 IDs → one epoch = 2 batches (windows at 0 and 2).
        // Requesting at iteration 0, 1, 2 (iter 2 is a new epoch).
        let all_ids: Vec<usize> = (0..4).collect();
        let mut sampler = EpochShuffledSampler::new(2, 13).expect("construction ok");

        let state0 = make_state_at(0);
        let state1 = make_state_at(1);
        let state2 = make_state_at(2); // crosses epoch boundary (2 * 2 = 4 == len of shuffled)

        let batch0 = sampler.next_batch_ids(&all_ids, &state0).expect("batch 0");
        let batch1 = sampler.next_batch_ids(&all_ids, &state1).expect("batch 1");
        let batch2 = sampler.next_batch_ids(&all_ids, &state2).expect("batch 2");

        // All batches must have the correct size.
        assert_eq!(batch0.len(), 2);
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch2.len(), 2);

        // batch0 and batch1 should be non-overlapping (same epoch, different windows).
        let set0: std::collections::HashSet<usize> = batch0.iter().copied().collect();
        let set1: std::collections::HashSet<usize> = batch1.iter().copied().collect();
        assert!(
            set0.is_disjoint(&set1),
            "batch0 and batch1 must be non-overlapping within the same epoch"
        );

        // batch2 is from a new epoch — its items must be valid IDs.
        for id in &batch2 {
            assert!(
                all_ids.contains(id),
                "batch2 id {id} should be a valid training ID"
            );
        }
    }

    // ------------------------------------------------------------------
    // Gap 37: determinism — two samplers with same seed produce identical batches
    // ------------------------------------------------------------------

    #[test]
    fn test_determinism_same_seed_same_sequence() {
        let all_ids: Vec<usize> = (0..8).collect();
        let seed = 12345u64;
        let mut sampler_a = EpochShuffledSampler::new(2, seed).expect("a ok");
        let mut sampler_b = EpochShuffledSampler::new(2, seed).expect("b ok");

        for iter in 0..4 {
            let state = make_state_at(iter);
            let batch_a = sampler_a
                .next_batch_ids(&all_ids, &state)
                .expect("a should sample");
            let batch_b = sampler_b
                .next_batch_ids(&all_ids, &state)
                .expect("b should sample");
            assert_eq!(
                batch_a, batch_b,
                "samplers with same seed must produce identical batch at iter {iter}"
            );
        }
    }
}
