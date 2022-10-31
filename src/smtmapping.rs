use novasmt::{ContentAddrStore, FullProof};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use tmelcrypt::HashVal;

use crate::stats::{STAT_SMT_GET_SECS, STAT_SMT_INSERT_SECS};

/// SmtMapping is a type-safe, constant-time cloneable, imperative-style interface to a sparse Merkle tree.
pub struct SmtMapping<C: ContentAddrStore, K: Serialize, V: Serialize + DeserializeOwned> {
    pub mapping: novasmt::Tree<C>,
    _phantom_k: PhantomData<K>,
    _phantom_v: PhantomData<V>,
}

impl<C: ContentAddrStore, K: Serialize, V: Serialize + DeserializeOwned> Debug
    for SmtMapping<C, K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.mapping.root_hash().fmt(f)
    }
}

impl<C: ContentAddrStore, K: Serialize, V: Serialize + DeserializeOwned> Clone
    for SmtMapping<C, K, V>
{
    fn clone(&self) -> Self {
        SmtMapping::new(self.mapping.clone())
    }
}

impl<C: ContentAddrStore, K: Serialize, V: Serialize + DeserializeOwned> SmtMapping<C, K, V> {
    /// Clears a mapping.
    pub fn clear(&mut self) {
        self.mapping.clear()
    }

    /// Returns true iff the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.root_hash().0 == [0; 32]
    }

    /// new converts a type-unsafe SMT to a SmtMapping
    pub fn new(tree: novasmt::Tree<C>) -> Self {
        SmtMapping {
            mapping: tree,
            _phantom_k: PhantomData,
            _phantom_v: PhantomData,
        }
    }

    /// Get obtains a mapping
    pub fn get(&self, key: &K) -> Option<V> {
        let _timer = STAT_SMT_GET_SECS.timer_secs("smt get");

        let key = tmelcrypt::hash_single(&stdcode::serialize(key).unwrap());
        let v_bytes = self.mapping.get(key.0);
        match v_bytes.len() {
            0 => None,
            _ => {
                let res: V = stdcode::deserialize(&v_bytes).expect("SmtMapping saw invalid data");
                Some(res)
            }
        }
    }

    /// Obtains a mapping, along with a proof.
    pub fn get_with_proof(&self, key: &K) -> (Option<V>, FullProof) {
        let _timer = STAT_SMT_GET_SECS.timer_secs("smt get w/ proof");

        let key = tmelcrypt::hash_single(&stdcode::serialize(key).unwrap());
        let (v_bytes, proof) = self.mapping.get_with_proof(key.0);
        match v_bytes.len() {
            0 => (None, proof),
            _ => {
                let res: V = stdcode::deserialize(&v_bytes).expect("SmtMapping saw invalid data");
                (Some(res), proof)
            }
        }
    }

    /// insert inserts a mapping, replacing any existing mapping
    pub fn insert(&mut self, key: K, val: V) {
        let _timer = STAT_SMT_INSERT_SECS.timer_secs("smt insert");

        let key = tmelcrypt::hash_single(&stdcode::serialize(&key).unwrap());
        self.mapping
            .insert(key.0, &stdcode::serialize(&val).unwrap());
    }

    /// delete deletes a mapping, replacing the mapping with a mapping to the empty bytestring
    pub fn delete(&mut self, key: &K) {
        let _timer = STAT_SMT_INSERT_SECS.timer_secs("smt delete");

        let key = tmelcrypt::hash_single(&stdcode::serialize(key).unwrap());
        self.mapping.insert(key.0, Default::default());
    }

    /// root_hash returns the root hash
    pub fn root_hash(&self) -> HashVal {
        HashVal(self.mapping.root_hash())
    }

    /// val_iter returns an iterator over the values.
    pub fn val_iter(&'_ self) -> impl Iterator<Item = V> + '_ {
        self.mapping
            .iter()
            .map(|(_, v)| stdcode::deserialize::<V>(&v).unwrap())
    }
}
