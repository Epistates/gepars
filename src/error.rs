/// Top-level error type for the GEPA optimizer.
///
/// Every subsystem converts its local errors into one of these variants before
/// propagating up to the engine loop.  The intent is to give callers a single
/// import path for pattern-matching while still embedding the root cause.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GEPAError {
    /// The user-supplied adapter returned an error during program evaluation.
    #[error("Adapter evaluation failed: {0}")]
    AdapterEvaluation(String),

    /// The adapter raised an error while building the reflective dataset for
    /// mutation proposals.
    #[error("Adapter reflective dataset build failed: {0}")]
    AdapterReflectiveDataset(String),

    /// A candidate evaluation step failed (adapter or subsample eval).
    #[error("Evaluation error: {0}")]
    Evaluation(String),

    /// The proposer subsystem could not generate a new candidate.
    #[error("Proposal failed: {0}")]
    ProposalFailed(String),

    /// The proposer could not produce improved candidate text.
    #[error("Proposal error: {0}")]
    Proposal(String),

    /// An upstream language-model call failed (network, timeout, etc.).
    #[error("LM call failed: {0}")]
    LMError(String),

    /// A language-model API call returned an error response.
    #[error("LM API error: {0}")]
    LmApi(String),

    /// All retry attempts for an LM API call have been exhausted.
    #[error("LM retries exhausted: {0}")]
    RetriesExhausted(String),

    /// The optimizer has no candidates to work with.  This normally indicates
    /// the seed candidate evaluation failed before any state was initialised.
    #[error("No candidates available")]
    NoCandidates,

    /// The Pareto frontier mapping is empty — cannot sample from it.
    #[error("Empty Pareto frontier")]
    EmptyFrontier,

    /// The training or validation dataset is empty.
    #[error("Empty dataset: {0}")]
    EmptyDataset(String),

    /// A configuration value is invalid or missing.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Transparent wrapper around [`std::io::Error`].
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Transparent wrapper around [`serde_json::Error`].
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Transparent wrapper around [`reqwest::Error`].
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T, E = GEPAError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_readable() {
        let err = GEPAError::AdapterEvaluation("timeout".into());
        assert_eq!(err.to_string(), "Adapter evaluation failed: timeout");

        let err = GEPAError::NoCandidates;
        assert_eq!(err.to_string(), "No candidates available");

        let err = GEPAError::Config("missing api_key".into());
        assert_eq!(err.to_string(), "Configuration error: missing api_key");
    }

    #[test]
    fn from_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let gepa_err = GEPAError::from(io_err);
        assert!(gepa_err.to_string().contains("IO error"));
    }

    #[test]
    fn from_serde_error_conversion() {
        let serde_err: serde_json::Error =
            serde_json::from_str::<serde_json::Value>("{invalid}").unwrap_err();
        let gepa_err = GEPAError::from(serde_err);
        assert!(gepa_err.to_string().contains("Serialization error"));
    }
}
