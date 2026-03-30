/// Public top-level API for running GEPA optimisation.
///
/// The primary entry point is [`optimize`], which builds a fully-configured
/// [`GEPAEngine`] from an [`OptimizeConfig`] and runs it to completion.
///
/// ### Quick start
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use gepa::api::{OptimizeConfig, optimize};
/// use gepa::core::data_loader::VecLoader;
///
/// # async fn run() -> gepa::Result<()> {
/// // (Implement your GEPAAdapter and supply a seed candidate here.)
/// # Ok(())
/// # }
/// ```
use std::sync::Arc;
use std::time::Duration;

use tracing::info;

use crate::core::adapter::{Candidate, GEPAAdapter};
use crate::core::callbacks::GEPACallback;
use crate::core::data_loader::{DataId, DataLoader};
use crate::core::engine::GEPAEngine;
use crate::core::result::GEPAResult;
use crate::core::state::FrontierType;
use crate::error::{GEPAError, Result};
use crate::lm::OpenAICompatibleLM;
use crate::proposer::merge::MergeProposer;
use crate::proposer::reflective_mutation::ReflectiveMutationProposer;
use crate::strategies::batch_sampler::EpochShuffledSampler;
use crate::strategies::candidate_selector::{
    CurrentBestSelector, EpsilonGreedySelector, ParetoCandidateSelector,
};
use crate::strategies::component_selector::{AllComponentSelector, RoundRobinSelector};
use crate::strategies::eval_policy::FullEvalPolicy;
use crate::tracking::NoopTracker;
use crate::utils::stop_condition::{
    CompositeMode, CompositeStopper, MaxIterationsStopper, MaxMetricCallsStopper, StopCondition,
    TimeoutStopper,
};

// ---------------------------------------------------------------------------
// Selector strategy selector
// ---------------------------------------------------------------------------

/// Which Pareto-front candidate selection strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CandidateSelectorKind {
    /// Select from the Pareto front weighted by frequency (Algorithm 2).
    #[default]
    Pareto,
    /// Always choose the highest average-score candidate.
    CurrentBest,
    /// ε-greedy: exploit `CurrentBest` with probability `1 − ε`, explore
    /// randomly otherwise.
    EpsilonGreedy,
}

/// Which component selection strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComponentSelectorKind {
    /// Update all components each iteration.
    All,
    /// Round-robin — advance one component at a time.
    #[default]
    RoundRobin,
}

// ---------------------------------------------------------------------------
// StopConditionConfig
// ---------------------------------------------------------------------------

/// Declarative stop-condition configuration.
///
/// All limits are additive via `CompositeMode::Any` — the first condition
/// to fire stops the engine.
#[derive(Debug, Clone)]
pub struct StopConditionConfig {
    /// Maximum number of adapter `evaluate()` calls.  `None` = unlimited.
    pub max_metric_calls: Option<usize>,
    /// Maximum number of optimisation iterations.  `None` = unlimited.
    pub max_iterations: Option<usize>,
    /// Wall-clock timeout.  `None` = unlimited.
    pub timeout: Option<Duration>,
}

impl Default for StopConditionConfig {
    fn default() -> Self {
        Self {
            max_metric_calls: Some(500),
            max_iterations: None,
            timeout: None,
        }
    }
}

// ---------------------------------------------------------------------------
// LMConfig
// ---------------------------------------------------------------------------

/// Configuration for the reflection language model.
#[derive(Clone)]
pub struct LMConfig {
    /// LiteLLM-style model identifier (e.g., `"gpt-4o-mini"`).
    pub model: String,
    /// API key (pass `""` for unauthenticated / local servers). Redacted in Debug.
    pub api_key: String,
    /// API base URL without trailing slash.
    pub base_url: String,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Number of HTTP retries on transient failures.
    pub max_retries: u32,
}

impl std::fmt::Debug for LMConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LMConfig")
            .field("model", &self.model)
            .field("api_key", &"***REDACTED***")
            .field("base_url", &self.base_url)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o-mini".into(),
            api_key: String::new(),
            base_url: "https://api.openai.com".into(),
            temperature: Some(1.0),
            max_tokens: Some(4096),
            max_retries: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// OptimizeConfig
// ---------------------------------------------------------------------------

/// All user-facing configuration for a single GEPA optimisation run.
///
/// Build this struct directly and pass it to [`optimize`].  Defaults are
/// sensible for most tasks.
pub struct OptimizeConfig<Id, Item, T, RO>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + 'static,
{
    // ---- Required -----------------------------------------------------------
    /// The starting candidate to optimise.
    pub seed_candidate: Candidate,
    /// Training data loader.
    pub trainset: Arc<dyn DataLoader<Id, Item>>,
    /// Validation data loader.
    pub valset: Arc<dyn DataLoader<Id, Item>>,
    /// Adapter: evaluates candidates and builds reflective datasets.
    pub adapter: Arc<dyn GEPAAdapter<Item, T, RO>>,
    /// Language model used for instruction mutation.
    pub lm_config: LMConfig,

    // ---- Stop condition -----------------------------------------------------
    /// When and why to stop optimising.
    pub stop_condition: StopConditionConfig,

    // ---- Strategy -----------------------------------------------------------
    /// Candidate selection strategy.
    pub candidate_selector: CandidateSelectorKind,
    /// ε value for `EpsilonGreedy` selector (ignored for other kinds).
    pub epsilon: f64,
    /// Component selection strategy.
    pub component_selector: ComponentSelectorKind,
    /// Training minibatch size.
    pub minibatch_size: usize,

    // ---- Merge --------------------------------------------------------------
    /// Enable the merge proposer (Algorithm 4).
    pub use_merge: bool,
    /// Maximum number of merge attempts over the whole run.
    pub max_merge_invocations: usize,
    /// Minimum shared validation IDs required before attempting a merge.
    pub val_overlap_floor: usize,

    // ---- Pareto front -------------------------------------------------------
    /// Frontier strategy.
    pub frontier_type: FrontierType,

    // ---- Misc ---------------------------------------------------------------
    /// Optional `perfect_score` threshold — iterations where all minibatch
    /// scores are already at this value may be skipped.
    pub perfect_score: Option<f64>,
    /// Whether to skip iterations where all minibatch scores equal `perfect_score`.
    pub skip_perfect_score: bool,
    /// Prompt template override for the reflection LM.  `None` uses the
    /// built-in Appendix C template.
    pub reflection_prompt_template:
        Option<crate::proposer::reflective_mutation::PromptTemplateConfig>,
    /// Callbacks for observing the optimisation run.
    pub callbacks: Vec<Box<dyn GEPACallback<Id>>>,
    /// RNG seed (stored in result metadata).
    pub rng_seed: Option<u64>,
    /// Artefact directory (stored in result metadata).
    pub run_dir: Option<String>,
    /// When set, `GEPAResult::best_candidate` returns the string stored under
    /// this key rather than the full dict.
    pub str_candidate_key: Option<String>,
    /// Whether to track the per-val-id best output seen so far (Gap 5).
    pub track_best_outputs: bool,
    /// Whether to enable the evaluation cache (Gap 10).
    pub cache_evaluation: bool,
}

impl<Id, Item, T, RO> OptimizeConfig<Id, Item, T, RO>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + 'static,
{
    /// Construct a config with the four required parameters and sensible defaults.
    pub fn new(
        seed_candidate: Candidate,
        trainset: Arc<dyn DataLoader<Id, Item>>,
        valset: Arc<dyn DataLoader<Id, Item>>,
        adapter: Arc<dyn GEPAAdapter<Item, T, RO>>,
        lm_config: LMConfig,
    ) -> Self {
        Self {
            seed_candidate,
            trainset,
            valset,
            adapter,
            lm_config,
            stop_condition: StopConditionConfig::default(),
            candidate_selector: CandidateSelectorKind::default(),
            epsilon: 0.1,
            component_selector: ComponentSelectorKind::default(),
            minibatch_size: 3,
            use_merge: false,
            max_merge_invocations: 5,
            val_overlap_floor: 5,
            frontier_type: FrontierType::Instance,
            perfect_score: Some(1.0),
            skip_perfect_score: true,
            reflection_prompt_template: None,
            callbacks: vec![],
            rng_seed: None,
            run_dir: None,
            str_candidate_key: None,
            track_best_outputs: true,
            cache_evaluation: false,
        }
    }
}

// ---------------------------------------------------------------------------
// optimize
// ---------------------------------------------------------------------------

/// Run GEPA optimisation with the given configuration.
///
/// Constructs and runs a [`GEPAEngine`] assembled from the components
/// described in `config`.
///
/// # Errors
/// Returns `Err` on LM client construction failures, adapter errors, or when
/// the validation set is empty.
pub async fn optimize<Id, Item, T, RO>(
    config: OptimizeConfig<Id, Item, T, RO>,
) -> Result<GEPAResult<Id>>
where
    Id: DataId,
    Item: Clone + Send + Sync + 'static,
    T: Send + Sync + 'static,
    RO: Send + Sync + 'static,
{
    // ── Build the language model client ──────────────────────────────────────
    let lm_cfg = &config.lm_config;
    let lm = OpenAICompatibleLM::new(
        lm_cfg.model.clone(),
        lm_cfg.api_key.clone(),
        lm_cfg.base_url.clone(),
        lm_cfg.temperature,
        lm_cfg.max_tokens,
    )
    .map_err(|e| GEPAError::Config(format!("Failed to construct LM client: {e}")))?
    .with_max_retries(lm_cfg.max_retries);

    let lm = Arc::new(lm);

    // ── Build the stop condition ──────────────────────────────────────────────
    let stop: Box<dyn StopCondition<Id>> = build_stop_condition(&config.stop_condition);

    // ── Build the candidate selector ─────────────────────────────────────────
    let rng_seed = config.rng_seed.unwrap_or(0);

    let candidate_selector: Box<dyn crate::strategies::candidate_selector::CandidateSelector<Id>> =
        match config.candidate_selector {
            CandidateSelectorKind::Pareto => Box::new(ParetoCandidateSelector::new(rng_seed)),
            CandidateSelectorKind::CurrentBest => Box::new(CurrentBestSelector),
            CandidateSelectorKind::EpsilonGreedy => Box::new(
                EpsilonGreedySelector::new(config.epsilon, rng_seed)
                    .map_err(|e| GEPAError::Config(format!("Invalid epsilon: {e}")))?,
            ),
        };

    // ── Build the component selector ─────────────────────────────────────────
    let component_selector: Box<dyn crate::strategies::component_selector::ComponentSelector<Id>> =
        match config.component_selector {
            ComponentSelectorKind::All => Box::new(AllComponentSelector),
            ComponentSelectorKind::RoundRobin => Box::new(RoundRobinSelector),
        };

    // ── Build the batch sampler ───────────────────────────────────────────────
    let batch_sampler = Box::new(
        EpochShuffledSampler::new(config.minibatch_size, rng_seed)
            .map_err(|e| GEPAError::Config(format!("Invalid minibatch_size: {e}")))?,
    );

    // ── Build the mutation proposer ───────────────────────────────────────────
    let mutation_proposer = ReflectiveMutationProposer {
        trainset: config.trainset.clone(),
        adapter: config.adapter.clone(),
        candidate_selector,
        component_selector,
        batch_sampler,
        reflection_lm: lm.clone(),
        reflection_prompt_template: config.reflection_prompt_template.clone(),
        perfect_score: config.perfect_score,
        skip_perfect_score: config.skip_perfect_score,
    };

    // ── Build the merge proposer ──────────────────────────────────────────────
    let merge_proposer = MergeProposer::new(
        config.valset.clone(),
        config.adapter.clone(),
        config.use_merge,
        config.max_merge_invocations,
        config.val_overlap_floor,
        rng_seed,
    )
    .map_err(|e| GEPAError::Config(format!("Failed to construct MergeProposer: {e}")))?;

    // ── Assemble the engine ───────────────────────────────────────────────────
    let trainset_len = config.trainset.all_ids().len();
    let valset_len = config.valset.all_ids().len();

    info!(
        trainset_size = trainset_len,
        valset_size = valset_len,
        stop = %stop.description(),
        "Starting GEPA optimisation"
    );

    let mut engine = GEPAEngine {
        trainset: config.trainset,
        valset: config.valset,
        adapter: config.adapter,
        seed_candidate: config.seed_candidate,
        mutation_proposer,
        merge_proposer,
        eval_policy: Box::new(FullEvalPolicy),
        stop_condition: stop,
        frontier_type: config.frontier_type,
        callbacks: config.callbacks,
        rng_seed: config.rng_seed,
        run_dir: config.run_dir,
        str_candidate_key: config.str_candidate_key,
        track_best_outputs: config.track_best_outputs,
        cache_evaluation: config.cache_evaluation,
        tracker: Box::new(NoopTracker),
    };

    engine.run().await
}

// ---------------------------------------------------------------------------
// Internal: build a composite stop condition from config
// ---------------------------------------------------------------------------

fn build_stop_condition<Id: DataId>(cfg: &StopConditionConfig) -> Box<dyn StopCondition<Id>> {
    let mut composite: CompositeStopper<Id> = CompositeStopper::new(CompositeMode::Any);
    let mut has_any = false;

    if let Some(max_calls) = cfg.max_metric_calls {
        composite = composite.push_condition(MaxMetricCallsStopper::new(max_calls));
        has_any = true;
    }
    if let Some(max_iters) = cfg.max_iterations {
        composite = composite.push_condition(MaxIterationsStopper::new(max_iters));
        has_any = true;
    }
    if let Some(timeout) = cfg.timeout {
        composite = composite.push_condition(TimeoutStopper::new(timeout));
        has_any = true;
    }

    if !has_any {
        // Safety valve: if no condition is specified, stop after a large but
        // finite number of iterations to prevent infinite loops.
        composite = composite.push_condition(MaxIterationsStopper::new(10_000));
    }

    Box::new(composite)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::core::adapter::{Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset};
    use crate::core::data_loader::VecLoader;
    use crate::error::Result;

    struct ConstantAdapter;

    #[async_trait]
    impl GEPAAdapter<String, (), String> for ConstantAdapter {
        async fn evaluate(
            &self,
            batch: &[String],
            _candidate: &Candidate,
            _capture_traces: bool,
        ) -> Result<EvaluationBatch<(), String>> {
            Ok(EvaluationBatch::new(batch.to_vec(), vec![0.8; batch.len()]))
        }

        async fn make_reflective_dataset(
            &self,
            _candidate: &Candidate,
            _eval_batch: &EvaluationBatch<(), String>,
            components: &[String],
        ) -> Result<ReflectiveDataset> {
            Ok(components.iter().map(|k| (k.clone(), vec![])).collect())
        }
    }

    fn make_config(max_iters: usize) -> OptimizeConfig<usize, String, (), String> {
        let trainset = Arc::new(VecLoader::new(vec![
            "train0".to_string(),
            "train1".to_string(),
        ]));
        let valset = Arc::new(VecLoader::new(vec!["val0".to_string(), "val1".to_string()]));
        let adapter: Arc<dyn GEPAAdapter<String, (), String>> = Arc::new(ConstantAdapter);

        let mut seed = Candidate::new();
        seed.insert("instructions".into(), "Be helpful.".into());

        // Use a dummy LM endpoint — the test adapter returns empty reflective
        // datasets so the LM will not actually be called.
        let lm_config = LMConfig {
            model: "test-model".into(),
            api_key: String::new(),
            base_url: "http://localhost:11434".into(),
            temperature: None,
            max_tokens: Some(64),
            max_retries: 0,
        };

        let mut cfg = OptimizeConfig::new(seed, trainset, valset, adapter, lm_config);
        cfg.stop_condition = StopConditionConfig {
            max_metric_calls: None,
            max_iterations: Some(max_iters),
            timeout: None,
        };
        cfg.use_merge = false;
        cfg
    }

    #[test]
    fn stop_condition_builds_correctly() {
        let cfg = StopConditionConfig {
            max_metric_calls: Some(100),
            max_iterations: Some(10),
            timeout: None,
        };
        let stop: Box<dyn StopCondition<usize>> = build_stop_condition(&cfg);
        assert!(stop.description().contains("Any"));
    }

    #[test]
    fn stop_condition_empty_uses_safety_valve() {
        let cfg = StopConditionConfig {
            max_metric_calls: None,
            max_iterations: None,
            timeout: None,
        };
        let stop: Box<dyn StopCondition<usize>> = build_stop_condition(&cfg);
        // Should not panic; the safety valve is added automatically.
        assert!(!stop.description().is_empty());
    }

    #[test]
    fn optimize_config_default_stop_condition() {
        let cfg: StopConditionConfig = StopConditionConfig::default();
        assert_eq!(cfg.max_metric_calls, Some(500));
    }

    #[test]
    fn lm_config_default_values() {
        let cfg = LMConfig::default();
        assert_eq!(cfg.model, "gpt-4o-mini");
        assert_eq!(cfg.max_retries, 3);
    }

    /// Smoke-test: optimize() with 0 iterations returns the seed candidate.
    #[tokio::test]
    async fn optimize_zero_iterations_returns_seed() {
        let mut config = make_config(0);
        // Override with a local mock LM server that won't be called.
        config.lm_config.base_url = "http://127.0.0.1:19999".into();
        config.lm_config.max_retries = 0;

        let result = optimize(config).await.expect("should succeed with 0 iters");
        assert_eq!(result.num_candidates(), 1);
        assert_eq!(
            result.candidates[0].get("instructions").unwrap(),
            "Be helpful."
        );
    }
}
