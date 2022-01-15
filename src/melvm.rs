mod consts;
mod executor;
pub mod opcode;
mod value;
use std::collections::HashMap;

use arbitrary::Arbitrary;

pub use executor::*;
use serde::{Deserialize, Serialize};

use themelio_structs::{Address, CoinDataHeight, CoinID, Header, Transaction};
use thiserror::Error;
pub use value::*;

use crate::melvm::{
    consts::{HADDR_SPENDER_INDEX, HADDR_SPENDER_TX},
    opcode::{opcodes_weight, DecodeError, EncodeError, OpCode},
};

#[derive(Clone, Eq, PartialEq, Debug, Arbitrary, Serialize, Deserialize, Hash)]
/// A MelVM covenant. Essentially, given a transaction that attempts to spend it, it either allows the transaction through or doesn't.
pub struct Covenant(#[serde(with = "stdcode::hex")] pub Vec<u8>);

#[derive(Error, Debug)]
pub enum AddrParseError {
    #[error("cannot parse covhash address")]
    CannotParse,
}

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
/// The execution environment of a covenant.
pub struct CovenantEnv<'a> {
    pub parent_coinid: &'a CoinID,
    pub parent_cdh: &'a CoinDataHeight,
    pub spender_index: u8,
    pub last_header: &'a Header,
}

impl Covenant {
    /// Converts to a vector of OpCodes.
    pub fn to_ops(&self) -> Result<Vec<OpCode>, DecodeError> {
        let mut opcodes: Vec<OpCode> = Vec::with_capacity(128);

        let mut temporary_slice: &[u8] = self.0.as_slice();

        while !temporary_slice.is_empty() {
            opcodes.push(OpCode::decode(&mut temporary_slice)?);
        }

        Ok(opcodes)
    }

    /// Create a Covenant from a slice of OpCodes.
    pub fn from_ops(ops: &[OpCode]) -> Result<Self, EncodeError> {
        let mut output: Vec<u8> = Vec::new();
        // go through output
        let was_encoding_successful: Result<(), EncodeError> =
            ops.iter().try_for_each(|op| match op.encode() {
                Ok(data) => {
                    output.extend_from_slice(&data);

                    Ok(())
                }
                Err(error) => Err(error),
            });

        match was_encoding_successful {
            Ok(()) => Ok(Covenant(output)),
            Err(error) => Err(error),
        }
    }

    /// Checks a transaction, returning whether or not the transaction is valid.
    ///
    /// The caller must also pass in the [CoinID] and [CoinDataHeight] corresponding to the coin that's being spent, as well as the [Header] of the *previous* block (if this transaction is trying to go into block N, then the header of block N-1). This allows the covenant to access (a committment to) its execution environment, allowing constructs like timelock contracts and colored-coin-like systems.
    pub fn check(&self, tx: &Transaction, env: CovenantEnv) -> bool {
        self.check_opt_env(tx, Some(env))
    }

    /// Execute a transaction in a [CovenantEnv] to completion and return whether the covenant succeeded.
    pub fn check_opt_env(&self, tx: &Transaction, env: Option<CovenantEnv>) -> bool {
        if let Ok(instrs) = self.to_ops() {
            Executor::new_from_env(instrs, tx.clone(), env).run_to_end()
        } else {
            false
        }
    }

    /// Runs to the end, with respect to a manually instantiated initial heap.
    /// This is for testing, when we do not have a transaction.
    pub fn debug_run_without_transaction(&self, args: &[Value]) -> bool {
        let mut hashmap: HashMap<u16, Value> = HashMap::new();

        args.iter().enumerate().for_each(|(index, value)| {
            hashmap.insert(index as u16, value.clone());
        });

        match self.to_ops() {
            Ok(ops) => {
                let mut executor: Executor = Executor::new(ops, hashmap);

                while !executor.at_end() {
                    if executor.stack.is_empty() {
                        eprintln!("Stack (step) is empty.");
                    } else {
                        eprintln!("Stack (step): {:?}", &executor.stack);
                    }

                    if executor.heap.is_empty() {
                        eprintln!("Heap (step) is empty.");
                    } else {
                        eprintln!("Heap (step): {:?}", &executor.heap);
                    }

                    if executor.step().is_none() {
                        return false;
                    }
                }

                if executor.stack.is_empty() {
                    eprintln!("Stack (final) is empty.");
                } else {
                    eprintln!("Stack (final): {:?}", &executor.stack);
                }

                if executor.heap.is_empty() {
                    eprintln!("Heap (final) is empty.");
                } else {
                    eprintln!("Heap (final): {:?}", &executor.heap);
                }

                executor
                    .stack
                    .pop()
                    .map(|f| f.into_bool())
                    .unwrap_or_default()
            }
            Err(error) => {
                eprintln!(
                    "While converting inputs to OpCodes, we hit a decode error: {}",
                    error
                );

                false
            }
        }
    }

    /// Runs to the end, with respect to a manually instantiated initial heap.
    /// This is for testing, when we do not have a transaction.
    /// This method outputs a tuple containing the stack and the heap.
    pub fn debug_run_outputting_stack_and_heap(
        &self,
        args: &[Value],
    ) -> Option<(Vec<Value>, HashMap<u16, Value>)> {
        let mut hashmap: HashMap<u16, Value> = HashMap::new();

        args.iter().enumerate().for_each(|(index, value)| {
            hashmap.insert(index as u16, value.clone());
        });

        match self.to_ops() {
            Ok(ops) => {
                let mut executor: Executor = Executor::new(ops, hashmap);

                while !executor.at_end() {
                    if executor.stack.is_empty() {
                        eprintln!("Stack (step) is empty.");
                    } else {
                        eprintln!("Stack (step): {:?}", &executor.stack);
                    }

                    if executor.heap.is_empty() {
                        eprintln!("Heap (step) is empty.");
                    } else {
                        eprintln!("Heap (step): {:?}", &executor.heap);
                    }

                    if executor.step().is_none() {
                        return None;
                    }
                }

                if executor.stack.is_empty() {
                    eprintln!("Stack (final) is empty.");
                } else {
                    eprintln!("Stack (final): {:?}", &executor.stack);
                }

                if executor.heap.is_empty() {
                    eprintln!("Heap (final) is empty.");
                } else {
                    eprintln!("Heap (final): {:?}", &executor.heap);
                }

                Some((executor.stack, executor.heap))
            }
            Err(error) => {
                eprintln!(
                    "While converting inputs to OpCodes, we hit a decode error: {}",
                    error
                );

                None
            }
        }
    }

    /// The hash of the covenant.
    pub fn hash(&self) -> Address {
        tmelcrypt::hash_single(&self.0).into()
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
        .expect("Could not create a legacy ed25519 signature checking covenant.")
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
        .expect("Could not create a new ed25519 signature checking covenant.")
    }

    pub fn always_true() -> Self {
        Covenant::from_ops(&[OpCode::PushI(1u32.into())]).unwrap()
    }

    pub fn weight(&self) -> Result<u128, DecodeError> {
        let ops = self.to_ops()?;

        Ok(opcodes_weight(&ops))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::*;
    use tap::Tap;
    fn dontcrash(data: &[u8]) {
        let script = Covenant(data.to_vec());
        if let Ok(ops) = script.to_ops() {
            println!("{:?}", ops);
            let redone = Covenant::from_ops(&ops).unwrap();
            assert_eq!(redone, script);
        }
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
                hm.insert(0, vec![0u8; 4096].into());
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
        ])
        .unwrap();
        println!("script length is {}", check_sig_script.0.len());
        let mut tx = Transaction::empty_test().signed_ed25519(sk);
        assert!(check_sig_script.check_opt_env(&tx, None));
        tx.sigs[0][0] ^= 123;
        assert!(!check_sig_script.check_opt_env(&tx, None));
    }

    // #[quickcheck]
    // fn loop_once_is_identity(bitcode: Vec<u8>) -> bool {
    //     let ops = Covenant(bitcode.clone()).to_ops();
    //     let tx = Transaction::empty_test();
    //     match ops {
    //         None => true,
    //         Some(ops) => {
    //             let loop_ops = vec![OpCode::Loop(1, ops.clone())];
    //             let loop_script = Covenant::from_ops(&loop_ops).unwrap();
    //             let orig_script = Covenant::from_ops(&ops).unwrap();
    //             loop_script.check_no_env(&tx) == orig_script.check_no_env(&tx)
    //         }
    //     }
    // }

    #[quickcheck]
    fn deterministic_execution(bitcode: Vec<u8>) -> bool {
        let ops = Covenant(bitcode).to_ops();
        let tx = Transaction::empty_test();
        match ops {
            Err(_) => true,
            Ok(ops) => {
                let orig_script = Covenant::from_ops(&ops).unwrap();
                let first = orig_script.check_opt_env(&tx, None);
                let second = orig_script.check_opt_env(&tx, None);
                first == second
            }
        }
    }
}
