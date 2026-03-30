//! Instruction proposal signature implementing the GEPA meta-prompt (Appendix C).
//!
//! Provides the meta-prompt template and helpers for rendering and extracting
//! improved instructions from an LM response.
//!
//! Mirrors `gepa.strategies.instruction_proposal.InstructionProposalSignature`.
use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// Meta-prompt (exact text from Appendix C of the GEPA paper)
// ---------------------------------------------------------------------------

/// The exact meta-prompt template from Appendix C of the GEPA paper.
///
/// Replace `<curr_param>` with the current instruction text and `<side_info>`
/// with the formatted reflective dataset.
pub const META_PROMPT_TEMPLATE: &str = "\
I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant \
along with the assistant's response for each of them, and some feedback on how \
the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task \
description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all \
niche and domain specific factual information about the task and include it in \
the instruction, as a lot of it may not be available to the assistant in the \
future. The assistant may have utilized a generalizable strategy to solve the \
task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.";

// ---------------------------------------------------------------------------
// Prompt rendering
// ---------------------------------------------------------------------------

/// Render the instruction-proposal meta-prompt.
///
/// Replaces `<curr_param>` with `current_instruction` and `<side_info>` with
/// `reflective_dataset` in [`META_PROMPT_TEMPLATE`].
///
/// Returns the rendered prompt string.
pub fn render_prompt(current_instruction: &str, reflective_dataset: &str) -> String {
    META_PROMPT_TEMPLATE
        .replace("<curr_param>", current_instruction)
        .replace("<side_info>", reflective_dataset)
}

/// Render using a custom template.
///
/// The template must contain both `<curr_param>` and `<side_info>` placeholders.
///
/// # Errors
/// Returns an error message when either placeholder is absent.
pub fn render_prompt_with_template(
    template: &str,
    current_instruction: &str,
    reflective_dataset: &str,
) -> Result<String, String> {
    validate_prompt_template(template)?;
    Ok(template
        .replace("<curr_param>", current_instruction)
        .replace("<side_info>", reflective_dataset))
}

/// Check that `template` contains both required placeholders.
///
/// # Errors
/// Returns a descriptive error when either placeholder is missing.
pub fn validate_prompt_template(template: &str) -> Result<(), String> {
    let missing: Vec<&str> = ["<curr_param>", "<side_info>"]
        .into_iter()
        .filter(|p| !template.contains(p))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Missing placeholder(s) in prompt template: {}",
            missing.join(", ")
        ))
    }
}

// ---------------------------------------------------------------------------
// Output extraction
// ---------------------------------------------------------------------------

/// Extract the proposed instruction from the LM response.
///
/// Searches for the first and last triple-backtick delimiters (` ``` `) in
/// `lm_response` and returns the text between them, stripping any optional
/// language specifier immediately after the opening fence.
///
/// Handles incomplete / malformed responses gracefully:
/// - Only an opening fence  → strips the fence and returns the rest.
/// - Only a closing fence   → strips the fence and returns the rest.
/// - No fences at all       → returns the stripped raw output.
///
/// Returns `None` when `lm_response` is empty or contains only whitespace.
pub fn extract_output(lm_response: &str) -> Option<String> {
    let trimmed = lm_response.trim();
    if trimmed.is_empty() {
        return None;
    }

    let fence = "```";
    let start_pos = trimmed.find(fence);
    let end_pos = trimmed.rfind(fence);

    match (start_pos, end_pos) {
        (Some(s), Some(e)) if s != e => {
            // Both fences found at different positions.
            let after_open = &trimmed[s + fence.len()..e];
            // Strip optional language specifier (e.g., "python\n").
            let content = strip_lang_specifier(after_open);
            Some(content.trim().to_string())
        }
        (Some(_s), _) => {
            // Only an opening fence (or open == close), strip it and return the rest.
            let stripped = if let Some(after) = trimmed.strip_prefix(fence) {
                // Strip the opening fence and optional lang specifier.
                strip_lang_specifier(after)
            } else {
                // Fence not at start; just strip it.
                trimmed.replacen(fence, "", 1)
            };
            Some(stripped.trim().to_string())
        }
        (None, Some(_)) => {
            // Only a closing fence.
            let stripped = trimmed.trim_end_matches(fence);
            Some(stripped.trim().to_string())
        }
        (None, None) => Some(trimmed.to_string()),
    }
}

/// Strip an optional language specifier (e.g., `python\n`) from the start of
/// a string.  Returns the string unchanged when no specifier is present.
fn strip_lang_specifier(s: &str) -> String {
    // A lang specifier is a run of non-whitespace chars followed by a newline.
    let mut chars = s.chars().peekable();
    let mut specifier_end = 0;
    for ch in &mut chars {
        if ch == '\n' {
            specifier_end += 1; // consume the newline
            break;
        }
        if ch.is_whitespace() {
            // Whitespace before newline means no lang specifier.
            specifier_end = 0;
            break;
        }
        specifier_end += ch.len_utf8();
    }
    if specifier_end > 0 {
        s[specifier_end..].to_string()
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Reflective dataset formatting
// ---------------------------------------------------------------------------

/// Format a slice of JSON sample records as a markdown string for use as
/// `<side_info>` in the meta-prompt.
///
/// Each record is rendered as a top-level markdown section numbered from 1.
/// Values are recursively rendered as nested headers.
pub fn format_samples_as_markdown(samples: &[serde_json::Value]) -> String {
    samples
        .iter()
        .enumerate()
        .map(|(i, sample)| convert_sample_to_markdown(sample, i + 1))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn convert_sample_to_markdown(sample: &serde_json::Value, num: usize) -> String {
    let mut s = format!("# Example {num}\n");
    if let Some(obj) = sample.as_object() {
        for (key, val) in obj {
            let _ = writeln!(s, "## {key}");
            s.push_str(&render_value(val, 3));
        }
    } else {
        s.push_str(&render_value(sample, 2));
    }
    s
}

fn render_value(value: &serde_json::Value, level: usize) -> String {
    let level = level.min(6);
    let hashes = "#".repeat(level);
    match value {
        serde_json::Value::Object(map) => {
            let mut s = String::new();
            for (k, v) in map {
                let _ = writeln!(s, "{hashes} {k}");
                s.push_str(&render_value(v, level + 1));
            }
            if map.is_empty() {
                s.push('\n');
            }
            s
        }
        serde_json::Value::Array(arr) => {
            let mut s = String::new();
            for (i, item) in arr.iter().enumerate() {
                let _ = writeln!(s, "{hashes} Item {}", i + 1);
                s.push_str(&render_value(item, level + 1));
            }
            if arr.is_empty() {
                s.push('\n');
            }
            s
        }
        serde_json::Value::String(text) => format!("{text}\n\n"),
        other => format!("{other}\n\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_prompt_substitutes_placeholders() {
        let prompt = render_prompt("Do the task.", "Example 1:\nInput: hi");
        assert!(prompt.contains("Do the task."));
        assert!(prompt.contains("Example 1:\nInput: hi"));
        assert!(!prompt.contains("<curr_param>"));
        assert!(!prompt.contains("<side_info>"));
    }

    #[test]
    fn validate_prompt_template_accepts_valid() {
        assert!(validate_prompt_template("Instruction: <curr_param>\nData: <side_info>").is_ok());
    }

    #[test]
    fn validate_prompt_template_rejects_missing_placeholders() {
        let err = validate_prompt_template("Missing both").unwrap_err();
        assert!(err.contains("<curr_param>"));
        assert!(err.contains("<side_info>"));
    }

    #[test]
    fn validate_prompt_template_rejects_one_missing() {
        let err = validate_prompt_template("Has <curr_param> but not the other").unwrap_err();
        assert!(err.contains("<side_info>"));
        assert!(!err.contains("<curr_param>"));
    }

    #[test]
    fn extract_output_normal_fenced_block() {
        let response = "Here is the instruction:\n```\nDo the task carefully.\n```\nDone.";
        let extracted = extract_output(response).expect("should extract");
        assert_eq!(extracted, "Do the task carefully.");
    }

    #[test]
    fn extract_output_fenced_block_with_lang_spec() {
        let response = "```python\nprint('hello')\n```";
        let extracted = extract_output(response).expect("should extract");
        assert_eq!(extracted, "print('hello')");
    }

    #[test]
    fn extract_output_no_fences_returns_stripped() {
        let response = "  Just plain text.  ";
        let extracted = extract_output(response).expect("should extract");
        assert_eq!(extracted, "Just plain text.");
    }

    #[test]
    fn extract_output_empty_returns_none() {
        assert!(extract_output("").is_none());
        assert!(extract_output("   ").is_none());
    }

    #[test]
    fn extract_output_only_opening_fence() {
        let response = "```\nInstruction text here.";
        let extracted = extract_output(response).expect("should extract");
        assert_eq!(extracted, "Instruction text here.");
    }

    #[test]
    fn extract_output_only_closing_fence() {
        let response = "Instruction text here.\n```";
        let extracted = extract_output(response).expect("should extract");
        assert_eq!(extracted, "Instruction text here.");
    }

    #[test]
    fn format_samples_as_markdown_produces_numbered_sections() {
        let samples = vec![
            serde_json::json!({
                "Inputs": "What is 2+2?",
                "Generated Outputs": "5",
                "Feedback": "Incorrect. The answer is 4."
            }),
            serde_json::json!({
                "Inputs": "What is the capital of France?",
                "Generated Outputs": "London",
                "Feedback": "Wrong. Paris is the capital of France."
            }),
        ];
        let md = format_samples_as_markdown(&samples);
        assert!(md.contains("# Example 1"));
        assert!(md.contains("# Example 2"));
        assert!(md.contains("## Inputs"));
        assert!(md.contains("## Feedback"));
    }

    #[test]
    fn meta_prompt_template_contains_placeholders() {
        assert!(META_PROMPT_TEMPLATE.contains("<curr_param>"));
        assert!(META_PROMPT_TEMPLATE.contains("<side_info>"));
        // Validate the template itself.
        assert!(validate_prompt_template(META_PROMPT_TEMPLATE).is_ok());
    }
}
