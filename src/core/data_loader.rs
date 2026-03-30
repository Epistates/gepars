/// Data loader traits and concrete helpers.
///
/// Defines the [`DataLoader`] / [`MutableDataLoader`] traits and the
/// [`VecLoader`] in-memory reference implementation.  The [`ensure_loader`]
/// helper normalises raw `Vec` values into proper loader instances so engine
/// code never needs to handle both forms.
///
/// Mirrors the Python `gepa.core.data_loader` module.
use std::fmt::Debug;
use std::hash::Hash;

use serde::{Serialize, de::DeserializeOwned};

// ---------------------------------------------------------------------------
// DataId bound
// ---------------------------------------------------------------------------

/// Bound required for all data-example identifiers.
///
/// Concrete types are typically `usize` (index-based) or `String`
/// (named / UUID-based) IDs.
pub trait DataId:
    Eq + Hash + Clone + Ord + Send + Sync + Serialize + DeserializeOwned + Debug + 'static
{
}

/// Blanket implementation: every type that satisfies the bound is a `DataId`.
impl<T> DataId for T where
    T: Eq + Hash + Clone + Ord + Send + Sync + Serialize + DeserializeOwned + Debug + 'static
{
}

// ---------------------------------------------------------------------------
// DataLoader trait
// ---------------------------------------------------------------------------

/// Minimal interface for retrieving validation examples keyed by opaque IDs.
///
/// The engine uses this to fetch specific subsets of the validation set
/// (e.g., a mini-batch) without materialising the full dataset.
pub trait DataLoader<Id: DataId, Item>: Send + Sync {
    /// Return the ordered universe of IDs currently available.
    fn all_ids(&self) -> Vec<Id>;

    /// Materialise the payloads corresponding to `ids`, preserving order.
    ///
    /// # Errors
    /// Returns an error if any ID is not present in the loader.
    fn fetch(&self, ids: &[Id]) -> crate::error::Result<Vec<Item>>;

    /// Return the current number of items in the loader.
    fn len(&self) -> usize;

    /// Returns `true` when the loader contains no items.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// MutableDataLoader trait
// ---------------------------------------------------------------------------

/// A [`DataLoader`] that supports appending new items at runtime.
///
/// Useful for curricula and online-learning scenarios where the validation
/// set grows incrementally.
pub trait MutableDataLoader<Id: DataId, Item>: DataLoader<Id, Item> {
    /// Append `items` to the loader, assigning new IDs automatically.
    fn add_items(&mut self, items: Vec<Item>);
}

// ---------------------------------------------------------------------------
// VecLoader
// ---------------------------------------------------------------------------

/// In-memory reference implementation backed by a `Vec<T>`.
///
/// IDs are zero-based `usize` indices.  [`MutableDataLoader::add_items`]
/// extends the underlying `Vec` and the new items are accessible via their
/// new contiguous indices.
///
/// ```rust
/// use gepa::core::data_loader::{VecLoader, DataLoader};
///
/// let loader = VecLoader::new(vec!["apple", "banana", "cherry"]);
/// assert_eq!(loader.all_ids(), vec![0, 1, 2]);
/// assert_eq!(loader.fetch(&[0, 2]).unwrap(), vec!["apple", "cherry"]);
/// assert_eq!(loader.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct VecLoader<T> {
    items: Vec<T>,
}

impl<T> VecLoader<T> {
    /// Construct a loader from an existing `Vec`.
    pub fn new(items: Vec<T>) -> Self {
        Self { items }
    }

    /// Return a reference to the underlying items.
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }
}

impl<T> DataLoader<usize, T> for VecLoader<T>
where
    T: Clone + Send + Sync,
{
    fn all_ids(&self) -> Vec<usize> {
        (0..self.items.len()).collect()
    }

    fn fetch(&self, ids: &[usize]) -> crate::error::Result<Vec<T>> {
        ids.iter()
            .map(|&idx| {
                self.items.get(idx).cloned().ok_or_else(|| {
                    crate::error::GEPAError::Config(format!(
                        "VecLoader: index {idx} out of range (len={})",
                        self.items.len()
                    ))
                })
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl<T> MutableDataLoader<usize, T> for VecLoader<T>
where
    T: Clone + Send + Sync,
{
    fn add_items(&mut self, items: Vec<T>) {
        self.items.extend(items);
    }
}

// ---------------------------------------------------------------------------
// ensure_loader
// ---------------------------------------------------------------------------

/// Wrap a `Vec<T>` into a [`VecLoader`] if needed.
///
/// When the caller already has a `VecLoader` they can pass a reference; this
/// function simply returns a new `VecLoader` wrapping the provided `Vec`.
///
/// This is the Rust analogue of Python's `ensure_loader`.
pub fn ensure_loader<T: Clone + Send + Sync>(items: Vec<T>) -> VecLoader<T> {
    VecLoader::new(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_loader_all_ids_and_fetch() {
        let loader = VecLoader::new(vec!["alpha", "beta", "gamma"]);
        assert_eq!(loader.all_ids(), vec![0, 1, 2]);
        assert_eq!(loader.fetch(&[0, 2]).unwrap(), vec!["alpha", "gamma"]);
        assert_eq!(loader.len(), 3);
        assert!(!loader.is_empty());
    }

    #[test]
    fn vec_loader_out_of_range_returns_error() {
        let loader = VecLoader::new(vec![10_i32, 20, 30]);
        let result = loader.fetch(&[5]);
        assert!(result.is_err(), "expected an error for index 5");
    }

    #[test]
    fn mutable_loader_add_items() {
        let mut loader = VecLoader::new(vec!["a", "b"]);
        loader.add_items(vec!["c", "d"]);
        assert_eq!(loader.len(), 4);
        assert_eq!(loader.all_ids(), vec![0, 1, 2, 3]);
        assert_eq!(loader.fetch(&[2, 3]).unwrap(), vec!["c", "d"]);
    }

    #[test]
    fn ensure_loader_wraps_vec() {
        let loader = ensure_loader(vec![1_u32, 2, 3]);
        assert_eq!(loader.len(), 3);
        assert_eq!(loader.fetch(&[1]).unwrap(), vec![2_u32]);
    }

    #[test]
    fn vec_loader_preserves_order() {
        let loader = VecLoader::new(vec![10_i32, 20, 30, 40, 50]);
        // Fetch in non-sequential order.
        let result = loader.fetch(&[4, 2, 0]).unwrap();
        assert_eq!(result, vec![50, 30, 10]);
    }
}
