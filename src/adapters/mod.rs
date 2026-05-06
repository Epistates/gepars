/// Built-in adapter implementations.
///
/// These adapters provide ready-made integration patterns for common use cases.
/// The [`ProcessAdapter`](process::ProcessAdapter) is particularly useful for
/// evaluating candidates that require running an external binary (e.g., a
/// training script, benchmark suite, or compiled model).
pub mod process;

pub use process::ProcessAdapter;
