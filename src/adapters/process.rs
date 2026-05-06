/// Subprocess-based adapter for evaluating candidates via external binaries.
///
/// This adapter spawns a child process for each evaluation, passes candidate
/// components as environment variables or a config file, captures stdout/stderr,
/// and parses metrics from the output.
///
/// This is the primary integration point for systems like oxideresearch where
/// evaluation means running a training loop and parsing metrics from stdout.
///
/// # Example: oxideresearch integration
///
/// ```rust,no_run
/// use gepa::adapters::ProcessAdapter;
/// use gepa::core::component::{ComponentMeta, ComponentMetaMap};
///
/// let mut meta = ComponentMetaMap::new();
/// meta.insert("hyperparams".into(), ComponentMeta::config("Training hyperparameters"));
/// meta.insert("architecture".into(), ComponentMeta::code("Model architecture", "rust"));
///
/// let adapter = ProcessAdapter::new("cargo")
///     .args(["run", "--release", "--", "train"])
///     .timeout(std::time::Duration::from_secs(330))  // 5 min + 30s overhead
///     .metric_parser(|stdout| {
///         // Parse "val_bpb: 0.997900" from stdout
///         gepa::adapters::process::parse_key_value_metrics(stdout)
///     })
///     .score_key("val_bpb")
///     .score_direction(gepa::adapters::process::ScoreDirection::Lower);
/// ```
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::core::adapter::{Candidate, EvaluationBatch, GEPAAdapter, ReflectiveDataset};
use crate::error::Result;

/// How to interpret the score: higher-is-better or lower-is-better.
///
/// GEPA internally uses higher-is-better. When `Lower` is selected, the
/// adapter negates the parsed metric so that GEPA's Pareto tracking works
/// correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoreDirection {
    /// Higher values are better (e.g., accuracy).
    #[default]
    Higher,
    /// Lower values are better (e.g., loss, BPB). Scores are negated internally.
    Lower,
}

/// Type alias for the metric parser function.
type MetricParserFn = Box<dyn Fn(&str) -> HashMap<String, f64> + Send + Sync>;

static NEXT_JSON_CONFIG_ID: AtomicU64 = AtomicU64::new(0);

/// Subprocess-based adapter.
///
/// Evaluates candidates by spawning an external process, passing candidate
/// components as environment variables, and parsing metrics from stdout.
pub struct ProcessAdapter {
    /// Binary to execute.
    command: String,
    /// Additional arguments.
    args: Vec<String>,
    /// Working directory for the subprocess.
    working_dir: Option<PathBuf>,
    /// Per-evaluation timeout.
    timeout: Option<Duration>,
    /// How to pass candidate components to the subprocess.
    pass_mode: PassMode,
    /// Function that parses stdout into named metrics.
    metric_parser: MetricParserFn,
    /// Which metric key to use as the primary score.
    score_key: String,
    /// Score direction (higher or lower is better).
    score_direction: ScoreDirection,
    /// Additional environment variables to set.
    extra_env: HashMap<String, String>,
    /// Multi-objective keys (optional).
    objective_keys: Vec<String>,
}

/// How candidate components are passed to the subprocess.
#[derive(Debug, Clone, Default)]
pub enum PassMode {
    /// Pass each component as an environment variable: `GEPA_<COMPONENT_NAME>=<value>`.
    #[default]
    EnvVars,
    /// Write components to a JSON file and pass the path as an env var.
    JsonFile {
        /// Path to write the JSON config.
        path: PathBuf,
    },
}

impl ProcessAdapter {
    /// Create a new process adapter with the given command.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            working_dir: None,
            timeout: None,
            pass_mode: PassMode::EnvVars,
            metric_parser: Box::new(parse_key_value_metrics),
            score_key: "score".into(),
            score_direction: ScoreDirection::Higher,
            extra_env: HashMap::new(),
            objective_keys: Vec::new(),
        }
    }

    /// Set additional command-line arguments.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Set the working directory for the subprocess.
    pub fn working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set a per-evaluation timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set how components are passed to the subprocess.
    pub fn pass_mode(mut self, mode: PassMode) -> Self {
        self.pass_mode = mode;
        self
    }

    /// Set the metric parser function.
    pub fn metric_parser(
        mut self,
        parser: impl Fn(&str) -> HashMap<String, f64> + Send + Sync + 'static,
    ) -> Self {
        self.metric_parser = Box::new(parser);
        self
    }

    /// Set the primary score key to extract from parsed metrics.
    pub fn score_key(mut self, key: impl Into<String>) -> Self {
        self.score_key = key.into();
        self
    }

    /// Set the score direction (higher or lower is better).
    pub fn score_direction(mut self, direction: ScoreDirection) -> Self {
        self.score_direction = direction;
        self
    }

    /// Add an extra environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_env.insert(key.into(), value.into());
        self
    }

    /// Set multi-objective score keys.
    pub fn objective_keys(mut self, keys: Vec<String>) -> Self {
        self.objective_keys = keys;
        self
    }

    /// Run the subprocess with the given candidate and return parsed metrics.
    async fn run_process(&self, candidate: &Candidate) -> Result<(String, String, i32)> {
        let mut cmd = tokio::process::Command::new(&self.command);
        cmd.args(&self.args);
        let mut cleanup_path: Option<PathBuf> = None;

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        // Pass candidate components
        match &self.pass_mode {
            PassMode::EnvVars => {
                for (name, value) in candidate {
                    let env_name = format!("GEPA_{}", name.to_uppercase().replace('-', "_"));
                    cmd.env(&env_name, value);
                }
            }
            PassMode::JsonFile { path } => {
                let config_path = unique_json_config_path(path);
                if let Some(parent) = config_path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|e| crate::error::GEPAError::AdapterError(e.to_string()))?;
                }
                let json = serde_json::to_string_pretty(candidate)
                    .map_err(|e| crate::error::GEPAError::AdapterError(e.to_string()))?;
                tokio::fs::write(&config_path, &json)
                    .await
                    .map_err(|e| crate::error::GEPAError::AdapterError(e.to_string()))?;
                cmd.env("GEPA_CONFIG_PATH", config_path.to_string_lossy().as_ref());
                cleanup_path = Some(config_path);
            }
        }

        // Extra env vars
        for (k, v) in &self.extra_env {
            cmd.env(k, v);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let output_result = if let Some(timeout) = self.timeout {
            cmd.kill_on_drop(true);
            let child = cmd
                .spawn()
                .map_err(|e| crate::error::GEPAError::AdapterError(format!("spawn failed: {e}")))?;

            match tokio::time::timeout(timeout, child.wait_with_output()).await {
                Ok(Ok(output)) => Ok(output),
                Ok(Err(e)) => Err(crate::error::GEPAError::AdapterError(format!(
                    "process error: {e}"
                ))),
                Err(_) => Err(crate::error::GEPAError::AdapterError(
                    "evaluation timed out".into(),
                )),
            }
        } else {
            cmd.output()
                .await
                .map_err(|e| crate::error::GEPAError::AdapterError(format!("exec failed: {e}")))
        };

        let output = match output_result {
            Ok(output) => output,
            Err(e) => {
                cleanup_temp_file(cleanup_path).await;
                return Err(e);
            }
        };

        cleanup_temp_file(cleanup_path).await;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        Ok((stdout, stderr, exit_code))
    }
}

async fn cleanup_temp_file(path: Option<PathBuf>) {
    if let Some(path) = path {
        let _ = tokio::fs::remove_file(path).await;
    }
}

fn unique_json_config_path(base: &Path) -> PathBuf {
    let id = NEXT_JSON_CONFIG_ID.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();

    if base.extension().is_some() {
        let parent = base.parent().unwrap_or_else(|| Path::new(""));
        let stem = base
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("gepa-config");
        let extension = base
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json");
        parent.join(format!("{stem}.gepa-{pid}-{id}.{extension}"))
    } else {
        base.join(format!("gepa-config-{pid}-{id}.json"))
    }
}

impl std::fmt::Debug for ProcessAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessAdapter")
            .field("command", &self.command)
            .field("args", &self.args)
            .field("score_key", &self.score_key)
            .field("score_direction", &self.score_direction)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// Execution trace from a process evaluation.
#[derive(Debug, Clone)]
pub struct ProcessTrace {
    /// Full stdout from the process.
    pub stdout: String,
    /// Full stderr from the process.
    pub stderr: String,
    /// Exit code.
    pub exit_code: i32,
    /// Parsed metrics.
    pub metrics: HashMap<String, f64>,
}

#[async_trait]
impl GEPAAdapter<(), ProcessTrace, HashMap<String, f64>> for ProcessAdapter {
    async fn evaluate(
        &self,
        batch: &[()],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> Result<EvaluationBatch<ProcessTrace, HashMap<String, f64>>> {
        // ProcessAdapter runs one evaluation per candidate (not per-example)
        if batch.len() != 1 {
            return Err(crate::error::GEPAError::AdapterError(format!(
                "ProcessAdapter expects exactly one unit item per evaluation batch, got {}. \
                 Use VecLoader::new(vec![()]) and minibatch_size=1 for candidate-level processes.",
                batch.len()
            )));
        }

        let (stdout, stderr, exit_code) = self.run_process(candidate).await?;

        let metrics = (self.metric_parser)(&stdout);

        // Extract primary score
        let score = if exit_code == 0 {
            let raw_score = metrics.get(&self.score_key).copied().ok_or_else(|| {
                crate::error::GEPAError::AdapterError(format!(
                    "score key '{}' was not found in process metrics; available keys: {:?}",
                    self.score_key,
                    sorted_metric_keys(&metrics)
                ))
            })?;
            match self.score_direction {
                ScoreDirection::Higher => raw_score,
                ScoreDirection::Lower => -raw_score, // negate so higher = better for GEPA
            }
        } else {
            -f64::MAX
        };

        let mut batch = EvaluationBatch::new(vec![metrics.clone()], vec![score]);

        // Multi-objective scores
        if !self.objective_keys.is_empty() {
            let mut obj_scores = HashMap::new();
            for key in &self.objective_keys {
                let val = if exit_code == 0 {
                    metrics.get(key).copied().ok_or_else(|| {
                        crate::error::GEPAError::AdapterError(format!(
                            "objective key '{key}' was not found in process metrics; available keys: {:?}",
                            sorted_metric_keys(&metrics)
                        ))
                    })?
                } else {
                    -f64::MAX
                };
                obj_scores.insert(key.clone(), val);
            }
            batch = batch.with_objective_scores(vec![obj_scores]);
        }

        if capture_traces {
            batch = batch.with_trajectories(vec![ProcessTrace {
                stdout,
                stderr,
                exit_code,
                metrics,
            }]);
        }

        Ok(batch)
    }

    async fn make_reflective_dataset(
        &self,
        candidate: &Candidate,
        eval_batch: &EvaluationBatch<ProcessTrace, HashMap<String, f64>>,
        components_to_update: &[String],
    ) -> Result<ReflectiveDataset> {
        let mut dataset = ReflectiveDataset::new();

        // Build reflective records from the process output
        let metrics = eval_batch.outputs.first().cloned().unwrap_or_default();
        let trace = eval_batch.trajectories.as_ref().and_then(|t| t.first());

        for component_name in components_to_update {
            let current_value = candidate.get(component_name).cloned().unwrap_or_default();

            let mut record = serde_json::Map::new();
            record.insert("Current Value".into(), Value::String(current_value));
            record.insert(
                "Metrics".into(),
                serde_json::to_value(&metrics).unwrap_or(Value::Null),
            );
            record.insert(
                "Score".into(),
                Value::Number(
                    serde_json::Number::from_f64(eval_batch.scores.first().copied().unwrap_or(0.0))
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );

            if let Some(trace) = trace {
                // Include relevant stdout lines as feedback
                let feedback_lines: Vec<&str> = trace
                    .stdout
                    .lines()
                    .filter(|line| {
                        line.contains("loss")
                            || line.contains("bpb")
                            || line.contains("mfu")
                            || line.contains("FAIL")
                            || line.starts_with("---")
                    })
                    .collect();
                record.insert("Feedback".into(), Value::String(feedback_lines.join("\n")));

                if trace.exit_code != 0 {
                    record.insert(
                        "Error".into(),
                        Value::String(format!(
                            "Process exited with code {}. stderr: {}",
                            trace.exit_code,
                            trace.stderr.chars().take(500).collect::<String>()
                        )),
                    );
                }
            }

            dataset.insert(component_name.clone(), vec![Value::Object(record)]);
        }

        Ok(dataset)
    }
}

// ---------------------------------------------------------------------------
// Metric parsing utilities
// ---------------------------------------------------------------------------

/// Parse `key: value` or `key=value` lines from stdout into a metric map.
///
/// Handles formats like:
/// ```text
/// val_bpb:          0.997900
/// training_seconds: 300.1
/// mfu_percent:      39.80
/// ```
///
/// Also handles `key=value` and `key: value` with arbitrary whitespace.
pub fn parse_key_value_metrics(stdout: &str) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();

    for line in stdout.lines() {
        for segment in line.split('|') {
            parse_metric_segment(segment.trim(), &mut metrics);
        }
    }

    metrics
}

fn parse_metric_segment(segment: &str, metrics: &mut HashMap<String, f64>) {
    let Some((key, value)) = segment.split_once(':').or_else(|| segment.split_once('=')) else {
        return;
    };

    let key = key.trim();
    if key.is_empty() {
        return;
    }

    if let Some(value) = parse_metric_value(value) {
        metrics.insert(key.to_string(), value);
    }
}

fn parse_metric_value(raw: &str) -> Option<f64> {
    let token = raw
        .split_whitespace()
        .next()?
        .trim_end_matches([',', ';', '%']);
    token.parse::<f64>().ok()
}

fn sorted_metric_keys(metrics: &HashMap<String, f64>) -> Vec<&str> {
    let mut keys: Vec<&str> = metrics.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_key_value_colon() {
        let stdout = "\
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8";

        let metrics = parse_key_value_metrics(stdout);
        assert!((metrics["val_bpb"] - 0.9979).abs() < 1e-4);
        assert!((metrics["training_seconds"] - 300.1).abs() < 0.1);
        assert!((metrics["mfu_percent"] - 39.80).abs() < 0.1);
        assert_eq!(metrics["num_steps"], 953.0);
        assert_eq!(metrics["depth"], 8.0);
    }

    #[test]
    fn parse_key_value_equals() {
        let stdout = "accuracy=0.95\nloss=0.123";
        let metrics = parse_key_value_metrics(stdout);
        assert!((metrics["accuracy"] - 0.95).abs() < 1e-6);
        assert!((metrics["loss"] - 0.123).abs() < 1e-6);
    }

    #[test]
    fn parse_ignores_non_metric_lines() {
        let stdout = "\
step 00042 (15.3%) | loss: 3.14159 | remaining: 250s
---
val_bpb:          0.997900
some random text without numbers";

        let metrics = parse_key_value_metrics(stdout);
        assert!(metrics.contains_key("val_bpb"));
        assert_eq!(metrics.get("loss"), Some(&3.14159));
        assert!(!metrics.contains_key("remaining"));
    }

    #[test]
    fn parse_key_value_with_units_and_scientific_notation() {
        let stdout = "mfu=39.8%\nlr: 1e-4,\nloss: -2.5e-1;";
        let metrics = parse_key_value_metrics(stdout);
        assert_eq!(metrics.get("mfu"), Some(&39.8));
        assert_eq!(metrics.get("lr"), Some(&1e-4));
        assert_eq!(metrics.get("loss"), Some(&-0.25));
    }

    #[test]
    fn score_direction_negation() {
        // Lower-is-better: GEPA negates so its internal higher-is-better works
        let raw = 0.9979;
        let adjusted = match ScoreDirection::Lower {
            ScoreDirection::Higher => raw,
            ScoreDirection::Lower => -raw,
        };
        assert!(adjusted < 0.0);
    }

    #[test]
    fn process_adapter_builder() {
        let adapter = ProcessAdapter::new("cargo")
            .args(["run", "--release", "--", "train"])
            .timeout(Duration::from_secs(330))
            .score_key("val_bpb")
            .score_direction(ScoreDirection::Lower)
            .env("RUST_LOG", "info");

        assert_eq!(adapter.command, "cargo");
        assert_eq!(adapter.args, vec!["run", "--release", "--", "train"]);
        assert_eq!(adapter.score_key, "val_bpb");
        assert_eq!(adapter.score_direction, ScoreDirection::Lower);
        assert!(adapter.timeout.is_some());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn process_adapter_runs_subprocess_with_env_vars() {
        use crate::core::adapter::GEPAAdapter;

        let adapter = ProcessAdapter::new("sh")
            .args([
                "-c",
                "test \"$GEPA_PROMPT\" = expected && printf 'score: 0.75\\naccuracy=0.80\\n'",
            ])
            .score_key("score")
            .objective_keys(vec!["accuracy".into()]);

        let mut candidate = Candidate::new();
        candidate.insert("prompt".into(), "expected".into());

        let batch = adapter
            .evaluate(&[()], &candidate, false)
            .await
            .expect("subprocess should run");
        assert_eq!(batch.scores, vec![0.75]);
        assert_eq!(
            batch
                .objective_scores
                .as_ref()
                .and_then(|rows| rows[0].get("accuracy")),
            Some(&0.80)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn process_adapter_json_file_uses_unique_temp_path_and_cleans_up() {
        use crate::core::adapter::GEPAAdapter;

        let dir = std::env::temp_dir().join(format!(
            "gepa-process-json-{}-{}",
            std::process::id(),
            NEXT_JSON_CONFIG_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let adapter = ProcessAdapter::new("sh")
            .args([
                "-c",
                "test -f \"$GEPA_CONFIG_PATH\" && grep -q '\"prompt\": \"expected\"' \"$GEPA_CONFIG_PATH\" && printf 'score: 0.61\\n'",
            ])
            .pass_mode(PassMode::JsonFile { path: dir.clone() })
            .score_key("score");

        let mut candidate = Candidate::new();
        candidate.insert("prompt".into(), "expected".into());

        let batch = adapter
            .evaluate(&[()], &candidate, false)
            .await
            .expect("subprocess should read JSON config");
        assert_eq!(batch.scores, vec![0.61]);

        let remaining = std::fs::read_dir(&dir)
            .expect("read temp dir")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("read entries");
        assert!(
            remaining.is_empty(),
            "temporary JSON config should be removed"
        );

        std::fs::remove_dir_all(dir).expect("cleanup temp dir");
    }

    #[tokio::test]
    async fn process_adapter_rejects_non_unit_batch_size() {
        use crate::core::adapter::GEPAAdapter;

        let adapter = ProcessAdapter::new("unused");
        let candidate = Candidate::new();
        let err = adapter
            .evaluate(&[(), ()], &candidate, false)
            .await
            .expect_err("batch size mismatch should error");
        assert!(err.to_string().contains("expects exactly one"));
    }
}
