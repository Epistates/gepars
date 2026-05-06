/// Component metadata for candidate components.
///
/// When GEPA optimizes candidates, each component (key in the `Candidate` map)
/// can be annotated with metadata that influences how mutations are proposed.
/// This is especially important when evolving structured artifacts like code
/// or configuration rather than natural language instructions.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The kind of content a candidate component contains.
///
/// This influences which meta-prompt template is used for reflective mutation
/// and how the reflective dataset is structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ComponentKind {
    /// Free-form natural language (instructions, prompts).
    /// Uses the standard GEPA meta-prompt from Appendix C.
    #[default]
    Text,

    /// Source code (Python, Rust, etc.).
    /// Uses a code-aware meta-prompt that preserves syntax and asks for
    /// targeted edits rather than full rewrites.
    Code,

    /// Structured configuration (hyperparameters, YAML/TOML/JSON).
    /// Uses a config-aware meta-prompt that understands key-value pairs
    /// and numeric ranges.
    Config,
}

/// Metadata for a single candidate component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMeta {
    /// What kind of content this component contains.
    pub kind: ComponentKind,

    /// Human-readable description of what this component controls.
    /// Included in the meta-prompt to give the reflection LM context.
    pub description: String,

    /// Optional language hint for code components (e.g., "python", "rust", "toml").
    pub language: Option<String>,

    /// Optional constraints for config components.
    /// e.g., `{"DEPTH": "integer, 1-32", "LEARNING_RATE": "float, 1e-5 to 1.0"}`
    pub constraints: Option<HashMap<String, String>>,
}

impl ComponentMeta {
    /// Create metadata for a text component.
    pub fn text(description: impl Into<String>) -> Self {
        Self {
            kind: ComponentKind::Text,
            description: description.into(),
            language: None,
            constraints: None,
        }
    }

    /// Create metadata for a code component.
    pub fn code(description: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            kind: ComponentKind::Code,
            description: description.into(),
            language: Some(language.into()),
            constraints: None,
        }
    }

    /// Create metadata for a config component.
    pub fn config(description: impl Into<String>) -> Self {
        Self {
            kind: ComponentKind::Config,
            description: description.into(),
            language: None,
            constraints: None,
        }
    }

    /// Add constraints (for config components).
    pub fn with_constraints(mut self, constraints: HashMap<String, String>) -> Self {
        self.constraints = Some(constraints);
        self
    }
}

/// Map from component name to its metadata.
///
/// Not every component needs metadata — components without entries use
/// `ComponentKind::Text` (the default, matching original GEPA behavior).
pub type ComponentMetaMap = HashMap<String, ComponentMeta>;

/// Look up the kind for a component, defaulting to `Text`.
pub fn component_kind(meta: &ComponentMetaMap, name: &str) -> ComponentKind {
    meta.get(name).map_or(ComponentKind::Text, |m| m.kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_kind_is_text() {
        assert_eq!(ComponentKind::default(), ComponentKind::Text);
    }

    #[test]
    fn component_kind_lookup_defaults_to_text() {
        let meta = ComponentMetaMap::new();
        assert_eq!(component_kind(&meta, "unknown"), ComponentKind::Text);
    }

    #[test]
    fn component_kind_lookup_returns_configured() {
        let mut meta = ComponentMetaMap::new();
        meta.insert(
            "model".into(),
            ComponentMeta::code("Model architecture", "python"),
        );
        meta.insert(
            "config".into(),
            ComponentMeta::config("Training hyperparameters"),
        );

        assert_eq!(component_kind(&meta, "model"), ComponentKind::Code);
        assert_eq!(component_kind(&meta, "config"), ComponentKind::Config);
        assert_eq!(component_kind(&meta, "instructions"), ComponentKind::Text);
    }

    #[test]
    fn config_constraints() {
        let meta = ComponentMeta::config("Hyperparams").with_constraints(HashMap::from([
            ("DEPTH".into(), "integer, 1-32".into()),
            ("LR".into(), "float, 1e-5 to 1.0".into()),
        ]));

        assert_eq!(meta.kind, ComponentKind::Config);
        assert!(meta.constraints.is_some());
        assert_eq!(meta.constraints.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn serde_round_trip() {
        let meta = ComponentMeta::code("Architecture", "rust");
        let json = serde_json::to_string(&meta).unwrap();
        let back: ComponentMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind, ComponentKind::Code);
        assert_eq!(back.language.as_deref(), Some("rust"));
    }
}
