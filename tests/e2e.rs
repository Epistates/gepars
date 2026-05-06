use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use gepa::core::callbacks::{StateSavedEvent, ValsetEvaluatedEvent};
use gepa::{
    Candidate, CandidateAcceptedEvent, CandidateSelectorKind, ComponentSelectorKind,
    EvaluationBatch, GEPAAdapter, GEPACallback, GEPAError, LMConfig, OptimizeConfig,
    ReflectiveDataset, StopConditionConfig, VecLoader, optimize,
};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Default)]
struct ScoringAdapter {
    evaluate_calls: AtomicUsize,
    evaluated_examples: AtomicUsize,
}

impl ScoringAdapter {
    fn score(candidate: &Candidate) -> f64 {
        candidate.get("instructions").map_or(0.2, |instructions| {
            if instructions.contains("improved rubric") {
                1.0
            } else {
                0.2
            }
        })
    }
}

#[async_trait]
impl GEPAAdapter<String, String, String> for ScoringAdapter {
    async fn evaluate(
        &self,
        batch: &[String],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> gepa::Result<EvaluationBatch<String, String>> {
        self.evaluate_calls.fetch_add(1, Ordering::SeqCst);
        self.evaluated_examples
            .fetch_add(batch.len(), Ordering::SeqCst);

        let score = Self::score(candidate);
        let instructions = candidate.get("instructions").cloned().unwrap_or_default();
        let outputs = batch
            .iter()
            .map(|item| format!("{item}::{instructions}"))
            .collect::<Vec<_>>();
        let mut evaluation = EvaluationBatch::new(outputs, vec![score; batch.len()]);

        if capture_traces {
            evaluation = evaluation.with_trajectories(
                batch
                    .iter()
                    .map(|item| format!("trace for {item}: score={score}"))
                    .collect(),
            );
        }

        Ok(evaluation)
    }

    async fn make_reflective_dataset(
        &self,
        _candidate: &Candidate,
        eval_batch: &EvaluationBatch<String, String>,
        components: &[String],
    ) -> gepa::Result<ReflectiveDataset> {
        if eval_batch.trajectories.is_none() {
            return Err(GEPAError::AdapterError(
                "e2e adapter expected trace-capturing evaluation".into(),
            ));
        }

        let record = json!({
            "Inputs": {"examples": eval_batch.outputs.len()},
            "Generated Outputs": eval_batch.outputs,
            "Feedback": "Use the improved rubric to satisfy every validation example."
        });

        Ok(components
            .iter()
            .map(|component| (component.clone(), vec![record.clone()]))
            .collect::<HashMap<_, _>>())
    }
}

#[derive(Default)]
struct RunEvents {
    accepted: AtomicUsize,
    saved: AtomicUsize,
    valset_evaluated: AtomicUsize,
}

struct RecordingCallback(Arc<RunEvents>);

impl GEPACallback<usize> for RecordingCallback {
    fn on_candidate_accepted(&self, _event: &CandidateAcceptedEvent) {
        self.0.accepted.fetch_add(1, Ordering::SeqCst);
    }

    fn on_state_saved(&self, _event: &StateSavedEvent) {
        self.0.saved.fetch_add(1, Ordering::SeqCst);
    }

    fn on_valset_evaluated(&self, _event: &ValsetEvaluatedEvent) {
        self.0.valset_evaluated.fetch_add(1, Ordering::SeqCst);
    }
}

fn unique_run_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!("gepa-e2e-{}-{nanos}", std::process::id()))
}

#[tokio::test]
#[ignore = "optional e2e; run with `cargo test --test e2e -- --ignored`"]
async fn optional_e2e_public_optimize_accepts_http_lm_mutation_and_saves_state() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"content": "```\nimproved rubric\n```"},
                "finish_reason": "stop"
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let adapter = Arc::new(ScoringAdapter::default());
    let trainset = Arc::new(VecLoader::new(vec![
        "train-0".to_string(),
        "train-1".to_string(),
        "train-2".to_string(),
    ]));
    let valset = Arc::new(VecLoader::new(vec![
        "val-0".to_string(),
        "val-1".to_string(),
        "val-2".to_string(),
    ]));
    let events = Arc::new(RunEvents::default());
    let run_dir = unique_run_dir();
    let _ = std::fs::remove_dir_all(&run_dir);

    let mut seed = Candidate::new();
    seed.insert("instructions".into(), "baseline rubric".into());

    let lm_config = LMConfig {
        model: "e2e-model".into(),
        api_key: String::new(),
        base_url: server.uri(),
        temperature: Some(0.0),
        max_tokens: Some(64),
        max_retries: 0,
    };

    let mut config = OptimizeConfig::new(seed, trainset, valset, adapter.clone(), lm_config);
    config.stop_condition = StopConditionConfig {
        max_metric_calls: None,
        max_iterations: Some(1),
        timeout: None,
    };
    config.candidate_selector = CandidateSelectorKind::CurrentBest;
    config.component_selector = ComponentSelectorKind::All;
    config.minibatch_size = 2;
    config.use_merge = false;
    config.skip_perfect_score = false;
    config.perfect_score = None;
    config.cache_evaluation = true;
    config.track_best_outputs = true;
    config.rng_seed = Some(7);
    config.run_dir = Some(run_dir.to_string_lossy().into_owned());
    config.str_candidate_key = Some("instructions".into());
    config.callbacks = vec![Box::new(RecordingCallback(events.clone()))];

    let result = optimize(config)
        .await
        .expect("optional e2e optimization should succeed");

    assert_eq!(result.num_candidates(), 2);
    assert_eq!(result.best_idx().expect("best candidate exists"), 1);
    assert_eq!(result.best_candidate_str(), Some("improved rubric"));
    // 3 seed validation examples + 2 before/2 after minibatch examples +
    // 3 accepted-candidate validation examples.
    assert_eq!(result.total_metric_calls, Some(10));
    assert_eq!(result.num_full_val_evals, Some(2));
    assert_eq!(adapter.evaluate_calls.load(Ordering::SeqCst), 4);
    assert_eq!(adapter.evaluated_examples.load(Ordering::SeqCst), 10);
    assert_eq!(events.accepted.load(Ordering::SeqCst), 1);
    assert_eq!(events.saved.load(Ordering::SeqCst), 1);
    assert_eq!(events.valset_evaluated.load(Ordering::SeqCst), 1);

    let state_path = run_dir.join("gepa_state.json");
    assert!(state_path.exists(), "state file should be saved");
    let state_json = std::fs::read_to_string(&state_path).expect("state file should be readable");
    assert!(state_json.contains("improved rubric"));
    assert!(state_json.contains("evaluation_cache"));

    let _ = std::fs::remove_dir_all(run_dir);
}
