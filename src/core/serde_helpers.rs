//! Shared serde helpers for serialising `HashMap`-based fields that cannot
//! use JSON object keys directly (e.g., when the key type is not `String`).
//!
//! Used by both [`super::state`] and [`super::result`].

// ---------------------------------------------------------------------------
// serde_vec_of_maps
// ---------------------------------------------------------------------------

/// Serialise/deserialise `Vec<HashMap<K, V>>` as a list of entry-pair lists.
pub mod serde_vec_of_maps {
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;
    use std::hash::Hash;

    pub fn serialize<K, V, S>(vec: &[HashMap<K, V>], ser: S) -> Result<S::Ok, S::Error>
    where
        K: Serialize + Eq + Hash,
        V: Serialize,
        S: Serializer,
    {
        let nested: Vec<Vec<(&K, &V)>> = vec.iter().map(|m| m.iter().collect()).collect();
        nested.serialize(ser)
    }

    pub fn deserialize<'de, K, V, D>(de: D) -> Result<Vec<HashMap<K, V>>, D::Error>
    where
        K: DeserializeOwned + Eq + Hash,
        V: DeserializeOwned,
        D: Deserializer<'de>,
    {
        let nested: Vec<Vec<(K, V)>> = Vec::deserialize(de)?;
        Ok(nested
            .into_iter()
            .map(|pairs| pairs.into_iter().collect())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// serde_map_set_as_vec
// ---------------------------------------------------------------------------

/// Serialise/deserialise `HashMap<K, HashSet<V>>` as a list of `(K, Vec<V>)` pairs.
pub mod serde_map_set_as_vec {
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::{HashMap, HashSet};
    use std::hash::Hash;

    pub fn serialize<K, V, S>(map: &HashMap<K, HashSet<V>>, ser: S) -> Result<S::Ok, S::Error>
    where
        K: Serialize + Eq + Hash,
        V: Serialize + Eq + Hash,
        S: Serializer,
    {
        let pairs: Vec<(&K, Vec<&V>)> = map
            .iter()
            .map(|(k, set)| (k, set.iter().collect()))
            .collect();
        pairs.serialize(ser)
    }

    pub fn deserialize<'de, K, V, D>(de: D) -> Result<HashMap<K, HashSet<V>>, D::Error>
    where
        K: DeserializeOwned + Eq + Hash,
        V: DeserializeOwned + Eq + Hash,
        D: Deserializer<'de>,
    {
        let pairs: Vec<(K, Vec<V>)> = Vec::deserialize(de)?;
        Ok(pairs
            .into_iter()
            .map(|(k, vec)| (k, vec.into_iter().collect()))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::{HashMap, HashSet};

    // ------------------------------------------------------------------
    // Helper: round-trip a value through JSON using a wrapper struct so we
    // can attach the custom serde modules via #[serde(with = "...")].
    // ------------------------------------------------------------------

    // -- serde_vec_of_maps round-trip wrapper --

    #[derive(Serialize, Deserialize, Debug)]
    struct VecOfMapsWrapper {
        #[serde(with = "serde_vec_of_maps")]
        data: Vec<HashMap<String, f64>>,
    }

    // -- serde_map_set_as_vec round-trip wrapper --

    #[derive(Serialize, Deserialize, Debug)]
    struct MapSetWrapper {
        #[serde(with = "serde_map_set_as_vec")]
        data: HashMap<String, HashSet<usize>>,
    }

    // ------------------------------------------------------------------
    // Gap 58a: Vec<HashMap<String, f64>> round-trip including empty maps
    // ------------------------------------------------------------------

    #[test]
    fn test_serde_vec_of_maps_round_trip() {
        let mut m1: HashMap<String, f64> = HashMap::new();
        m1.insert("alpha".into(), 1.0);
        m1.insert("beta".into(), 2.5);

        // m2 is deliberately empty to exercise that code path.
        let m2: HashMap<String, f64> = HashMap::new();

        let mut m3: HashMap<String, f64> = HashMap::new();
        m3.insert("gamma".into(), 0.0);

        let original = VecOfMapsWrapper {
            data: vec![m1.clone(), m2.clone(), m3.clone()],
        };

        let json = serde_json::to_string(&original).expect("serialisation must succeed");
        let recovered: VecOfMapsWrapper =
            serde_json::from_str(&json).expect("deserialisation must succeed");

        assert_eq!(
            recovered.data.len(),
            3,
            "round-trip must preserve the number of maps"
        );

        // Empty map must survive.
        assert!(
            recovered.data[1].is_empty(),
            "empty map at index 1 must survive round-trip"
        );

        // Non-empty maps must have correct content.
        assert_eq!(
            recovered.data[0].get("alpha").copied(),
            Some(1.0),
            "alpha key must survive round-trip"
        );
        assert_eq!(
            recovered.data[2].get("gamma").copied(),
            Some(0.0),
            "gamma=0.0 must survive round-trip"
        );
    }

    // ------------------------------------------------------------------
    // Gap 58b: HashMap<String, HashSet<usize>> round-trip including empty sets
    // ------------------------------------------------------------------

    #[test]
    fn test_serde_map_set_as_vec_round_trip() {
        let mut original_data: HashMap<String, HashSet<usize>> = HashMap::new();
        original_data.insert("key_a".into(), [1, 2, 3].iter().copied().collect());
        // Empty set.
        original_data.insert("key_b".into(), HashSet::new());
        original_data.insert("key_c".into(), [42].iter().copied().collect());

        let wrapper = MapSetWrapper {
            data: original_data.clone(),
        };

        let json = serde_json::to_string(&wrapper).expect("serialisation must succeed");
        let recovered: MapSetWrapper =
            serde_json::from_str(&json).expect("deserialisation must succeed");

        // All three keys must survive.
        assert_eq!(
            recovered.data.len(),
            3,
            "round-trip must preserve key count"
        );

        // Non-empty set "key_a" must have correct members.
        let set_a = recovered.data.get("key_a").expect("key_a must exist");
        assert_eq!(
            set_a,
            &[1usize, 2, 3].iter().copied().collect::<HashSet<_>>()
        );

        // Empty set "key_b" must still be present and empty.
        let set_b = recovered.data.get("key_b").expect("key_b must exist");
        assert!(set_b.is_empty(), "empty set must survive round-trip");

        // Singleton set "key_c".
        let set_c = recovered.data.get("key_c").expect("key_c must exist");
        assert!(
            set_c.contains(&42),
            "key_c must contain 42 after round-trip"
        );
    }
}
