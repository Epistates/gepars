/// Experiment tracking abstraction (Gap 9).
///
/// Provides a thin trait over experiment-tracking back-ends such as
/// Weights & Biases and MLflow.  A no-op implementation is provided for
/// use when tracking is not required.
///
/// ## Feature flags
/// - `wandb`   — enables `WandbTracker` (stub; requires the `wandb` Python
///               binary on `$PATH` when the actual integration is wired up).
/// - `mlflow`  — enables `MlflowTracker` (stub; requires a running MLflow
///               server).
///
/// Both stubs log via `tracing` so they are useful for development without
/// requiring external services.

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for experiment-tracking back-ends.
///
/// All methods have default no-op implementations so that new back-ends only
/// need to override the methods they support.
pub trait ExperimentTracker: Send + Sync {
    /// Log a scalar metric at a given step.
    ///
    /// # Arguments
    /// * `key`   — metric name.
    /// * `value` — metric value.
    /// * `step`  — optimisation iteration (monotonically increasing).
    fn log_metric(&self, key: &str, value: f64, step: usize);

    /// Log a text artifact (e.g., the best candidate prompt).
    ///
    /// Default: no-op.
    fn log_text(&self, key: &str, value: &str) {
        let _ = (key, value);
    }

    /// Signal that the experiment run has finished.
    ///
    /// Trackers that buffer metrics should flush here.
    ///
    /// Default: no-op.
    fn finish(&self) {}
}

// ---------------------------------------------------------------------------
// NoopTracker — default used when no external tracking is requested
// ---------------------------------------------------------------------------

/// Experiment tracker that discards all events.
///
/// This is the default tracker used by the engine when the caller does not
/// supply an explicit tracker.
pub struct NoopTracker;

impl ExperimentTracker for NoopTracker {
    #[inline]
    fn log_metric(&self, _key: &str, _value: f64, _step: usize) {}

    #[inline]
    fn log_text(&self, _key: &str, _value: &str) {}

    #[inline]
    fn finish(&self) {}
}

// ---------------------------------------------------------------------------
// WandbTracker — stub (behind `wandb` feature flag)
// ---------------------------------------------------------------------------

/// Stub experiment tracker for Weights & Biases.
///
/// In this stub implementation all methods log via `tracing`.  A full
/// implementation would call into the W&B SDK (e.g., via `pyo3` or the
/// `wandb` REST API).
///
/// Enabled with the `wandb` Cargo feature.
#[cfg(feature = "wandb")]
pub struct WandbTracker {
    /// W&B project name.
    pub project: String,
    /// Optional run name.
    pub run_name: Option<String>,
}

#[cfg(feature = "wandb")]
impl ExperimentTracker for WandbTracker {
    fn log_metric(&self, key: &str, value: f64, step: usize) {
        tracing::info!(
            tracker = "wandb",
            project = %self.project,
            key,
            value,
            step,
            "log_metric"
        );
    }

    fn log_text(&self, key: &str, value: &str) {
        tracing::info!(
            tracker = "wandb",
            project = %self.project,
            key,
            value,
            "log_text"
        );
    }

    fn finish(&self) {
        tracing::info!(
            tracker = "wandb",
            project = %self.project,
            "finish"
        );
    }
}

// ---------------------------------------------------------------------------
// MlflowTracker — stub (behind `mlflow` feature flag)
// ---------------------------------------------------------------------------

/// Stub experiment tracker for MLflow.
///
/// In this stub implementation all methods log via `tracing`.  A full
/// implementation would POST to the MLflow tracking REST API.
///
/// Enabled with the `mlflow` Cargo feature.
#[cfg(feature = "mlflow")]
pub struct MlflowTracker {
    /// MLflow tracking server URI (e.g., `http://localhost:5000`).
    pub tracking_uri: String,
    /// Experiment name to log under.
    pub experiment_name: String,
}

#[cfg(feature = "mlflow")]
impl ExperimentTracker for MlflowTracker {
    fn log_metric(&self, key: &str, value: f64, step: usize) {
        tracing::info!(
            tracker = "mlflow",
            uri = %self.tracking_uri,
            experiment = %self.experiment_name,
            key,
            value,
            step,
            "log_metric"
        );
    }

    fn log_text(&self, key: &str, value: &str) {
        tracing::info!(
            tracker = "mlflow",
            uri = %self.tracking_uri,
            experiment = %self.experiment_name,
            key,
            value,
            "log_text"
        );
    }

    fn finish(&self) {
        tracing::info!(
            tracker = "mlflow",
            uri = %self.tracking_uri,
            experiment = %self.experiment_name,
            "finish"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A test tracker that records every `log_metric` call.
    struct RecordingTracker {
        calls: std::sync::Mutex<Vec<(String, f64, usize)>>,
    }

    impl RecordingTracker {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn recorded(&self) -> Vec<(String, f64, usize)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl ExperimentTracker for RecordingTracker {
        fn log_metric(&self, key: &str, value: f64, step: usize) {
            self.calls
                .lock()
                .unwrap()
                .push((key.to_owned(), value, step));
        }
    }

    #[test]
    fn noop_tracker_compiles_and_is_silent() {
        let t = NoopTracker;
        t.log_metric("loss", 0.5, 0);
        t.log_text("prompt", "hello");
        t.finish();
        // No assertion needed — just ensure it compiles and does not panic.
    }

    #[test]
    fn recording_tracker_captures_metrics() {
        let t = RecordingTracker::new();
        t.log_metric("accuracy", 0.8, 1);
        t.log_metric("loss", 0.2, 2);

        let calls = t.recorded();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], ("accuracy".to_owned(), 0.8, 1));
        assert_eq!(calls[1], ("loss".to_owned(), 0.2, 2));
    }

    #[test]
    fn experiment_tracker_is_object_safe() {
        // Verifies that `Box<dyn ExperimentTracker>` is valid.
        let t: Box<dyn ExperimentTracker> = Box::new(NoopTracker);
        t.log_metric("x", 1.0, 0);
        t.finish();
    }

    #[test]
    fn default_log_text_and_finish_are_no_ops() {
        // Use the recording tracker which only overrides log_metric.
        let t = RecordingTracker::new();
        t.log_text("key", "val");
        t.finish();
        // No metrics should have been recorded.
        assert!(t.recorded().is_empty());
    }
}
