mod consts;
mod executor;
pub mod opcode;
mod value;
use std::sync::Arc;

use bytes::Bytes;
use consts::{HADDR_SPENDER_INDEX, HADDR_SPENDER_TX};
// pub use executor::*;

use executor::Executor;
use opcode::{opcodes_weight, DecodeError, OpCode};
use serde::{Deserialize, Serialize};
use melstructs::{Address, CoinDataHeight, CoinID, Header, Transaction};
use thiserror::Error;
pub use value::*;

/// Weight calculator from bytes, to use in Transaction::weight etc.
pub fn covenant_weight_from_bytes(b: &[u8]) -> u128 {
    Covenant::from_bytes(b).map(|b| b.weight()).unwrap_or(0)
}

#[derive(Clone, Eq, PartialEq, Debug)]
/// A MelVM covenant. Essentially, given a transaction that attempts to spend it, it either allows the transaction through or doesn't.
pub struct Covenant(Arc<Vec<OpCode>>);

#[derive(Error, Debug)]
pub enum AddrParseError {
    #[error("cannot parse covhash address")]
    CannotParse,
}

#[derive(Clone, Eq, PartialEq, Debug, Hash, Serialize, Deserialize)]
/// The execution environment of a covenant. Serializable to make use with yaml/toml/etc more convenient.
pub struct CovenantEnv {
    pub parent_coinid: CoinID,
    pub parent_cdh: CoinDataHeight,
    pub spender_index: u8,
    pub last_header: Header,
}

impl Covenant {
    /// Parses a covenant from its binary representation.
    pub fn from_bytes(mut b: &[u8]) -> Result<Self, DecodeError> {
        let mut opcodes: Vec<OpCode> = Vec::with_capacity(128);

        while !b.is_empty() {
            opcodes.push(OpCode::decode(&mut b)?);
        }

        Ok(Self(opcodes.into()))
    }

    /// Parses a covenant from a list of opcodes.
    pub fn from_ops(ops: &[OpCode]) -> Self {
        Self(Arc::new(ops.to_vec()))
    }

    /// Converts to a list of opcodes.
    pub fn to_ops(&self) -> Vec<OpCode> {
        self.0.as_ref().clone()
    }

    /// Converts to a byte vector.
    pub fn to_bytes(&self) -> Bytes {
        let mut out: Vec<u8> = vec![];
        for op in self.0.iter() {
            op.encode(&mut out).unwrap();
        }
        out.into()
    }

    /// Executes this covenant against a transaction, returning whether or not the transaction is valid.
    ///
    /// The caller must also pass in the [CoinID] and [CoinDataHeight] corresponding to the coin that's being spent, as well as the [Header] of the *previous* block (if this transaction is trying to go into block N, then the header of block N-1). This allows the covenant to access (a commitment to) its execution environment, allowing constructs like timelock contracts and colored-coin-like systems.
    pub fn execute(&self, tx: &Transaction, env: Option<CovenantEnv>) -> Option<Value> {
        Executor::new_from_env(self.0.to_vec(), tx.clone(), env).run_to_end()
    }

    /// Executes this covenant in isolation, with a particular melvm heap.
    pub fn debug_execute(&self, env: &[Value]) -> Option<Value> {
        Executor::new(
            self.0.to_vec(),
            env.iter()
                .enumerate()
                .map(|(i, k)| (i as u16, k.clone()))
                .collect(),
        )
        .run_to_end()
    }

    /// The hash of the covenant.
    pub fn hash(&self) -> Address {
        tmelcrypt::hash_single(&self.to_bytes()).into()
    }

    /// Returns a legacy ed25519 signature checking covenant, which checks the *first* signature.
    pub fn std_ed25519_pk_legacy(pk: tmelcrypt::Ed25519PK) -> Self {
        Covenant::from_ops(&[
            OpCode::PushI(0u32.into()),
            OpCode::PushI(6u32.into()),
            OpCode::LoadImm(HADDR_SPENDER_TX),
            OpCode::VRef,
            OpCode::VRef,
            OpCode::PushB(pk.0.to_vec()),
            OpCode::LoadImm(1),
            OpCode::SigEOk(32),
        ])
    }

    /// Returns a new ed25519 signature checking covenant, which checks the *nth* signature when spent as the nth input.
    pub fn std_ed25519_pk_new(pk: tmelcrypt::Ed25519PK) -> Self {
        Covenant::from_ops(&[
            OpCode::LoadImm(HADDR_SPENDER_INDEX),
            OpCode::PushI(6u32.into()),
            OpCode::LoadImm(HADDR_SPENDER_TX),
            OpCode::VRef,
            OpCode::VRef,
            OpCode::PushB(pk.0.to_vec()),
            OpCode::LoadImm(1),
            OpCode::SigEOk(32),
        ])
    }

    pub fn always_true() -> Self {
        Covenant::from_ops(&[OpCode::PushI(1u32.into())])
    }

    pub fn weight(&self) -> u128 {
        opcodes_weight(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use bytes::Bytes;

    use tap::Tap;
    fn dontcrash(data: &[u8]) {
        let script = Covenant::from_bytes(data).unwrap();
        let ops = script.to_ops();
        println!("{:?}", ops);
        let redone = Covenant::from_ops(&ops);
        assert_eq!(redone, script);
    }

    #[test]
    fn fuzz_crash_0() {
        dontcrash(&hex::decode("b000001010").unwrap())
    }

    #[test]
    fn stack_overflow() {
        let mut data = Vec::new();

        let range = 0..100000;

        range.into_iter().for_each(|_index| data.push(0xb0));

        dontcrash(&data.to_vec())
    }
    #[test]
    fn imbl_bug() {
        use opcode::OpCode::*;
        let ops = vec![
            VEmpty,
            VEmpty,
            VEmpty,
            VEmpty,
            BEmpty,
            BEmpty,
            Hash(29298),
            BAppend,
            BEmpty,
            BAppend,
            BEmpty,
            Hash(11264),
            BEmpty,
            BEmpty,
            BAppend,
            BEmpty,
            Hash(29298),
            BAppend,
            BEmpty,
            BAppend,
            BEmpty,
            BLength,
            BCons,
            LoadImm(0),
            BAppend,
            BEmpty,
            Hash(29298),
            BAppend,
            BEmpty,
            Hash(29298),
            Hash(29298),
            BAppend,
            BEmpty,
            BAppend,
            BEmpty,
            BLength,
            BCons,
            LoadImm(0),
            BAppend,
        ];
        let mut exec = Executor::new(
            ops,
            HashMap::new().tap_mut(|hm| {
                hm.insert(0, Bytes::from(vec![0u8; 4096]).into());
            }),
        );
        exec.run_to_end();
    }

    #[test]
    fn check_sig() {
        let (pk, sk) = tmelcrypt::ed25519_keygen();
        // (SIGEOK (LOAD 1) (PUSH pk) (VREF (VREF (LOAD 0) 6) 0))
        let check_sig_script = Covenant::from_ops(&[
            OpCode::Loop(5, 8),
            OpCode::PushI(0u32.into()),
            OpCode::PushI(6u32.into()),
            OpCode::LoadImm(0),
            OpCode::VRef,
            OpCode::VRef,
            OpCode::PushB(pk.0.to_vec()),
            OpCode::LoadImm(1),
            OpCode::SigEOk(32),
        ]);
        println!("script length is {}", check_sig_script.0.len());
        let mut tx = Transaction::default().signed_ed25519(sk);
        assert!(check_sig_script.execute(&tx, None).unwrap().into_bool());
        let mut sig = tx.sigs[0].to_vec();
        sig[0] ^= 123;
        tx.sigs[0] = sig.into();
        assert!(!check_sig_script.execute(&tx, None).unwrap().into_bool());
    }
}
