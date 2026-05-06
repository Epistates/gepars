/// Language-model abstraction for calling OpenAI-compatible APIs.
///
/// Provides:
/// - [`LanguageModel`] — the async trait that all LM implementations satisfy.
/// - [`OpenAICompatibleLM`] — a production-grade implementation that POSTs to
///   `/v1/chat/completions`, handles streaming responses, and retries on
///   transient failures with exponential back-off.
///
/// Mirrors `gepa.lm.LM` and `gepa.proposer.reflective_mutation.base.LanguageModel`.
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, warn};

use crate::error::{GEPAError, Result};

const MAX_ERROR_BODY_CHARS: usize = 256;

// ---------------------------------------------------------------------------
// LanguageModel trait
// ---------------------------------------------------------------------------

/// Async trait satisfied by every language-model implementation.
///
/// The trait is object-safe so it can be stored as `Arc<dyn LanguageModel>`.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// Call the language model with a plain-text `prompt` and return the
    /// model's response as a `String`.
    ///
    /// # Errors
    /// Returns `Err` on network failures, API errors, or when all retries are
    /// exhausted.
    async fn complete(&self, prompt: &str) -> Result<String>;
}

// ---------------------------------------------------------------------------
// Request / response types for the OpenAI chat-completions API
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // fields used by serde deserialization
struct Choice {
    message: Option<MessageContent>,
    delta: Option<DeltaContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // fields used by serde deserialization
struct DeltaContent {
    content: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAICompatibleLM
// ---------------------------------------------------------------------------

/// Language-model client that calls any OpenAI-compatible `/v1/chat/completions`
/// endpoint.
///
/// Compatible with:
/// - `OpenAI` API (`https://api.openai.com/v1/chat/completions`)
/// - Anthropic Messages API via the OpenAI-compatible shim
/// - `LMStudio` local server (`http://localhost:1234/v1/chat/completions`)
/// - Ollama's OpenAI-compatible endpoint
/// - Any other OpenAI-spec compliant server
///
/// ### Retry policy
/// On 429 (rate-limited) or 5xx responses, requests are retried up to
/// `max_retries` times with exponential back-off starting at 1 second.
#[derive(Clone)]
pub struct OpenAICompatibleLM {
    /// LiteLLM-style model identifier (e.g., `"gpt-4o-mini"`, `"llama3"`).
    pub model: String,
    /// Bearer token for the API (omitted when empty). Redacted in Debug output.
    pub api_key: String,
    /// Full base URL for the API, e.g. `"https://api.openai.com"`.
    pub base_url: String,
    /// Sampling temperature forwarded to the model.
    pub temperature: Option<f64>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Whether to use streaming mode to receive incremental output.
    pub use_streaming: bool,
    /// Maximum number of retry attempts on transient failures.
    pub max_retries: u32,
    client: Client,
}

impl std::fmt::Debug for OpenAICompatibleLM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatibleLM")
            .field("model", &self.model)
            .field("api_key", &"***REDACTED***")
            .field("base_url", &self.base_url)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("use_streaming", &self.use_streaming)
            .field("max_retries", &self.max_retries)
            .field("client", &"<reqwest::Client>")
            .finish()
    }
}

impl OpenAICompatibleLM {
    /// Construct a new client.
    ///
    /// # Arguments
    /// * `model`       — Model identifier forwarded in the request body.
    /// * `api_key`     — Bearer token (pass `""` for unauthenticated servers).
    /// * `base_url`    — API base URL without trailing slash.
    /// * `temperature` — Optional sampling temperature.
    /// * `max_tokens`  — Optional maximum tokens.
    ///
    /// # Errors
    /// Returns `Err` when the `reqwest` client cannot be built.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_mins(2))
            .build()
            .map_err(|e| GEPAError::Config(format!("Failed to build HTTP client: {e}")))?;

        Ok(Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: base_url.into(),
            temperature,
            max_tokens,
            use_streaming: false,
            max_retries: 3,
            client,
        })
    }

    /// Enable or disable streaming responses.
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.use_streaming = enabled;
        self
    }

    /// Override the default retry count (3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Build the chat-completions endpoint URL.
    fn completions_url(&self) -> String {
        format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        )
    }

    /// Perform a single non-streaming chat-completions request.
    async fn complete_non_streaming(&self, prompt: &str) -> Result<String> {
        let request_body = ChatCompletionRequest {
            model: &self.model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: None,
        };

        let url = self.completions_url();
        debug!(url = %url, model = %self.model, "sending chat completion request");

        let mut req = self.client.post(&url).json(&request_body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let truncated = truncate_error_body(&body);
            return Err(GEPAError::LmApi(format!(
                "API returned HTTP {status}: {truncated}"
            )));
        }

        let completion: ChatCompletionResponse = response.json().await?;

        // Check for truncation.
        if let Some(choice) = completion.choices.first()
            && let Some(ref reason) = choice.finish_reason
            && reason == "length"
        {
            warn!(
                model = %self.model,
                max_tokens = ?self.max_tokens,
                "LM response was truncated (finish_reason=length). \
                 Consider increasing max_tokens."
            );
        }

        let content = completion
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message)
            .and_then(|m| m.content)
            .ok_or_else(|| GEPAError::LmApi("API returned an empty message content".into()))?;

        Ok(content)
    }

    /// Perform a streaming chat-completions request and accumulate all chunks.
    async fn complete_streaming(&self, prompt: &str) -> Result<String> {
        let request_body = ChatCompletionRequest {
            model: &self.model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: Some(true),
        };

        let url = self.completions_url();
        debug!(url = %url, model = %self.model, "sending streaming chat completion request");

        let mut req = self.client.post(&url).json(&request_body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let response = req.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let truncated = truncate_error_body(&body);
            return Err(GEPAError::LmApi(format!(
                "Streaming API returned HTTP {status}: {truncated}"
            )));
        }

        // Accumulate SSE data chunks.
        let mut accumulated = String::new();
        let text = response.text().await?;

        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    break;
                }
                if let Ok(chunk) = serde_json::from_str::<Value>(data)
                    && let Some(delta_content) = chunk
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                {
                    accumulated.push_str(delta_content);
                }
            }
        }

        if accumulated.is_empty() {
            return Err(GEPAError::LmApi(
                "Streaming response produced no content".into(),
            ));
        }

        Ok(accumulated)
    }
}

fn truncate_error_body(body: &str) -> String {
    let mut chars = body.chars();
    let prefix = chars
        .by_ref()
        .take(MAX_ERROR_BODY_CHARS)
        .collect::<String>();
    if chars.next().is_some() {
        format!("{prefix}...[truncated]")
    } else {
        body.to_string()
    }
}

#[async_trait]
impl LanguageModel for OpenAICompatibleLM {
    async fn complete(&self, prompt: &str) -> Result<String> {
        let mut last_err: Option<GEPAError> = None;

        for attempt in 0..=self.max_retries {
            let result = if self.use_streaming {
                self.complete_streaming(prompt).await
            } else {
                self.complete_non_streaming(prompt).await
            };

            match result {
                Ok(content) => return Ok(content),
                Err(e) => {
                    let is_retryable = match &e {
                        GEPAError::Http(req_err) => {
                            // Retry on connection errors, timeouts, and 5xx status codes.
                            req_err.is_connect()
                                || req_err.is_timeout()
                                || req_err.status().is_some_and(|s| s.is_server_error())
                        }
                        GEPAError::LmApi(msg) => {
                            // Retry on rate-limit (429) and server errors (5xx).
                            msg.contains("HTTP 429")
                                || msg.contains("HTTP 500")
                                || msg.contains("HTTP 502")
                                || msg.contains("HTTP 503")
                                || msg.contains("HTTP 504")
                        }
                        _ => false,
                    };

                    if is_retryable && attempt < self.max_retries {
                        let backoff = Duration::from_secs(2u64.pow(attempt));
                        warn!(
                            attempt = attempt + 1,
                            max = self.max_retries,
                            backoff_secs = backoff.as_secs(),
                            error = %e,
                            "LM request failed, retrying"
                        );
                        tokio::time::sleep(backoff).await;
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(GEPAError::RetriesExhausted(format!(
            "All {} retries exhausted: {}",
            self.max_retries,
            last_err.map_or("unknown error".into(), |e| e.to_string())
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Unit tests that don't require a live API
    // ---------------------------------------------------------------------------

    #[test]
    fn openai_lm_builds_correct_url() {
        let lm = OpenAICompatibleLM::new(
            "gpt-4o-mini",
            "sk-test",
            "https://api.openai.com",
            Some(0.7),
            Some(2048),
        )
        .expect("should build");
        assert_eq!(
            lm.completions_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn openai_lm_trailing_slash_stripped() {
        let lm = OpenAICompatibleLM::new("gpt-4o-mini", "", "http://localhost:1234/", None, None)
            .expect("should build");
        assert_eq!(
            lm.completions_url(),
            "http://localhost:1234/v1/chat/completions"
        );
    }

    #[test]
    fn openai_lm_builder_methods_chain() {
        let lm = OpenAICompatibleLM::new("model", "key", "http://host", None, None)
            .expect("should build")
            .with_streaming(true)
            .with_max_retries(5);
        assert!(lm.use_streaming);
        assert_eq!(lm.max_retries, 5);
    }

    /// Verify that a mock LM implementing the trait works end-to-end.
    #[tokio::test]
    async fn trait_object_completes_successfully() {
        struct MockLM;
        #[async_trait]
        impl LanguageModel for MockLM {
            async fn complete(&self, _prompt: &str) -> Result<String> {
                Ok("Mock response".into())
            }
        }

        let lm: Box<dyn LanguageModel> = Box::new(MockLM);
        let result = lm.complete("hello").await.expect("mock should succeed");
        assert_eq!(result, "Mock response");
    }

    // ------------------------------------------------------------------
    // Gap 47: error body truncation produces output <= 256 + overhead chars
    // ------------------------------------------------------------------

    #[test]
    fn test_error_body_truncation() {
        let long_body = "x".repeat(600);
        let truncated = truncate_error_body(&long_body);

        // The truncated string must start with the first 256 chars.
        assert_eq!(
            truncated.chars().take(256).collect::<String>(),
            long_body.chars().take(256).collect::<String>()
        );
        // And must end with the sentinel.
        assert!(
            truncated.ends_with("...[truncated]"),
            "truncated body must end with '...[truncated]'"
        );
        // Total length must be 256 + len("...[truncated]") = 256 + 14 = 270.
        assert_eq!(truncated.len(), 256 + "...[truncated]".len());

        // A body that fits within 256 chars must NOT be truncated.
        let short_body = "y".repeat(100);
        let not_truncated = truncate_error_body(&short_body);
        assert_eq!(
            not_truncated, short_body,
            "short body should be passed through unchanged"
        );
        assert!(!not_truncated.ends_with("...[truncated]"));
    }

    #[test]
    fn test_error_body_truncation_handles_multibyte_text() {
        let body = "é".repeat(300);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("...[truncated]"));
        assert_eq!(truncated.chars().take(256).count(), 256);
    }

    // ------------------------------------------------------------------
    // Gap 48: completions URL construction with various base_url inputs
    // ------------------------------------------------------------------

    #[test]
    fn test_completions_url_construction() {
        struct Case {
            base_url: &'static str,
            expected: &'static str,
        }

        let cases = [
            Case {
                base_url: "https://api.openai.com",
                expected: "https://api.openai.com/v1/chat/completions",
            },
            Case {
                base_url: "https://api.openai.com/",
                expected: "https://api.openai.com/v1/chat/completions",
            },
            Case {
                base_url: "http://localhost:1234",
                expected: "http://localhost:1234/v1/chat/completions",
            },
            Case {
                base_url: "http://localhost:1234/",
                expected: "http://localhost:1234/v1/chat/completions",
            },
            Case {
                base_url: "http://host:8080//",
                expected: "http://host:8080/v1/chat/completions",
            },
        ];

        for case in &cases {
            let lm = OpenAICompatibleLM::new("model", "", case.base_url, None, None)
                .expect("should build");
            assert_eq!(
                lm.completions_url(),
                case.expected,
                "base_url='{}' should produce correct URL",
                case.base_url
            );
        }
    }

    #[tokio::test]
    async fn mock_non_streaming_completion_returns_message() {
        use serde_json::json;
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {"content": "world"},
                    "finish_reason": "stop"
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let lm = OpenAICompatibleLM::new("test-model", "", server.uri(), None, Some(32))
            .expect("should build")
            .with_max_retries(0);
        let response = lm.complete("hello").await.expect("mock should succeed");
        assert_eq!(response, "world");
    }

    #[tokio::test]
    async fn mock_completion_retries_transient_server_error() {
        use serde_json::json;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("try again"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {"content": "recovered"},
                    "finish_reason": "stop"
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let lm = OpenAICompatibleLM::new("test-model", "", server.uri(), None, Some(32))
            .expect("should build")
            .with_max_retries(1);
        let response = lm.complete("hello").await.expect("retry should recover");
        assert_eq!(response, "recovered");
    }

    #[tokio::test]
    async fn mock_streaming_completion_accumulates_sse_chunks() {
        use serde_json::json;
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"stream": true})))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .expect(1)
            .mount(&server)
            .await;

        let lm = OpenAICompatibleLM::new("test-model", "", server.uri(), None, Some(32))
            .expect("should build")
            .with_streaming(true)
            .with_max_retries(0);
        let response = lm
            .complete("hello")
            .await
            .expect("streaming mock should succeed");
        assert_eq!(response, "hello");
    }

    /// Integration test (requires a live server — skipped in CI).
    #[tokio::test]
    #[ignore = "requires a live OpenAI-compatible API"]
    async fn integration_complete() {
        let lm = OpenAICompatibleLM::new(
            "gpt-4o-mini",
            std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            "https://api.openai.com",
            Some(0.0),
            Some(50),
        )
        .expect("should build");

        let response = lm
            .complete("Reply with exactly the word PONG.")
            .await
            .expect("API call should succeed");
        assert!(
            response.to_uppercase().contains("PONG"),
            "unexpected response: {response}"
        );
    }
}
