mod consts;
pub mod opcode;

pub use crate::{CoinData, CoinID, Transaction};
use crate::{CoinDataHeight, Denom, Header, HexBytes};

use std::{collections::HashMap, str::FromStr};
use std::{convert::TryInto, fmt::Display};

use arbitrary::Arbitrary;
use catvec::CatVec;
use derive_more::{From, Into};
use ethnum::U256;
use serde::{Deserialize, Serialize};
use tap::Tap;
use thiserror::Error;
use tmelcrypt::HashVal;

use crate::melvm::{
    consts::{
        HADDR_LAST_HEADER, HADDR_PARENT_ADDITIONAL_DATA, HADDR_PARENT_DENOM, HADDR_PARENT_HEIGHT,
        HADDR_PARENT_INDEX, HADDR_PARENT_TXHASH, HADDR_PARENT_VALUE, HADDR_SELF_HASH,
        HADDR_SPENDER_INDEX, HADDR_SPENDER_TX, HADDR_SPENDER_TXHASH,
    },
    opcode::{opcodes_weight, DecodeError, EncodeError, OpCode},
};


#[derive(Clone, Eq, PartialEq, Debug, Arbitrary, Serialize, Deserialize, Hash)]
/// A MelVM covenant. Essentially, given a transaction that attempts to spend it, it either allows the transaction through or doesn't.
pub struct Covenant(#[serde(with = "stdcode::hex")] pub Vec<u8>);

/// A pointer to the currently executing instruction.
type ProgramCounter = usize;

/// An address is the hash of a MelVM covenant. In Bitcoin terminology, all Themelio addresses are "pay-to-script-hash".
#[derive(
    Copy,
    Clone,
    Debug,
    Eq,
    PartialEq,
    Hash,
    PartialOrd,
    Ord,
    From,
    Into,
    Serialize,
    Deserialize,
    Arbitrary,
)]
pub struct Address(pub HashVal);

impl Address {
    /// Returns the address that represents destruction of a coin.
    pub fn coin_destroy() -> Self {
        Address(Default::default())
    }
}

impl Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.to_addr().fmt(f)
    }
}

impl FromStr for Address {
    type Err = AddrParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        HashVal::from_addr(s)
            .ok_or(AddrParseError::CannotParse)
            .map(|v| v.into())
    }
}

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
        let was_encoding_successful: Result<(), EncodeError> = ops.iter().try_for_each(|op| {
            match op.encode() {
                Ok(data) => {
                    output.extend_from_slice(&data);

                    Ok(())
                },
                Err(error) => Err(error),
            }
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

        if let Ok(ops) = self.to_ops() {
            let mut executor: Executor = Executor::new(ops, hashmap);

            while executor.pc < executor.instrs.len() {
                if executor.stack.is_empty() {
                    dbg!("Stack (step) is empty.");
                } else {
                    dbg!("Stack (step): {:?}", &executor.stack);
                }

                if executor.heap.is_empty() {
                    dbg!("Heap (step) is empty.");
                } else {
                    dbg!("Heap (step): {:?}", &executor.heap);
                }

                if executor.step().is_none() {
                    return false;
                }
            }

            if executor.stack.is_empty() {
                dbg!("Stack (final) is empty.");
            } else {
                dbg!("Stack (final): {:?}", &executor.stack);
            }

            if executor.heap.is_empty() {
                dbg!("Heap (final) is empty.");
            } else {
                dbg!("Heap (final): {:?}", &executor.heap);
            }

            executor.stack.pop().map(|f| f.into_bool()).unwrap_or_default()
        } else {
            false
        }
    }

    /// Runs to the end, with respect to a manually instantiated initial heap.
    /// This is for testing, when we do not have a transaction.
    /// This method outputs a tuple containing the stack and the heap.
    pub fn debug_run_outputting_stack_and_heap(&self, args: &[Value]) -> Option<(Vec<Value>, HashMap<u16, Value>)> {
        let mut hashmap: HashMap<u16, Value> = HashMap::new();

        args.iter().enumerate().for_each(|(index, value)| {
            hashmap.insert(index as u16, value.clone());
        });

        if let Ok(ops) = self.to_ops() {
            let mut executor: Executor = Executor::new(ops, hashmap);

            while executor.pc < executor.instrs.len() {
                if executor.stack.is_empty() {
                    dbg!("Stack (step) is empty.");
                } else {
                    dbg!("Stack (step): {:?}", &executor.stack);
                }

                if executor.heap.is_empty() {
                    dbg!("Heap (step) is empty.");
                } else {
                    dbg!("Heap (step): {:?}", &executor.heap);
                }

                if executor.step().is_none() {
                    return None;
                }
            }

            if executor.stack.is_empty() {
                dbg!("Stack (final) is empty.");
            } else {
                dbg!("Stack (final): {:?}", &executor.stack);
            }

            if executor.heap.is_empty() {
                dbg!("Heap (final) is empty.");
            } else {
                dbg!("Heap (final): {:?}", &executor.heap);
            }

            Some((executor.stack, executor.heap))
        } else {
            None
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

/// Internal tracking of state during a loop in [Executor].
struct LoopState {
    /// Pointer to first op in loop
    begin: ProgramCounter,
    /// Pointer to last op in loop (inclusive)
    end: ProgramCounter,
    /// Total number of iterations
    iterations_left: u16,
}

pub struct Executor {
    pub stack: Vec<Value>,
    pub heap: HashMap<u16, Value>,
    instrs: Vec<OpCode>,
    /// Program counter
    pc: ProgramCounter,
    /// Marks the (begin, end) of the loop if currently in one
    loop_state: Vec<LoopState>,
}

impl Executor {
    pub fn new(instrs: Vec<OpCode>, heap_init: HashMap<u16, Value>) -> Self {
        Executor {
            stack: Vec::new(),
            heap: heap_init,
            instrs,
            pc: 0,
            loop_state: vec![],
        }
    }

    pub fn new_from_env(instrs: Vec<OpCode>, tx: Transaction, env: Option<CovenantEnv>) -> Self {
        let mut hm = HashMap::new();
        hm.insert(HADDR_SPENDER_TXHASH, Value::from_bytes(&tx.hash_nosigs().0));
        let tx_val = Value::from(tx);
        hm.insert(HADDR_SPENDER_TX, tx_val);
        if let Some(env) = env {
            let CoinID { txhash, index } = &env.parent_coinid;

            hm.insert(HADDR_PARENT_TXHASH, txhash.0.into());
            hm.insert(HADDR_PARENT_INDEX, Value::Int(U256::from(*index)));

            let CoinDataHeight {
                coin_data:
                    CoinData {
                        covhash,
                        value,
                        denom,
                        additional_data,
                    },
                height,
            } = &env.parent_cdh;

            hm.insert(HADDR_SELF_HASH, covhash.0.into());
            hm.insert(HADDR_PARENT_VALUE, (value.0).into());
            hm.insert(HADDR_PARENT_DENOM, (*denom).into());
            hm.insert(HADDR_PARENT_ADDITIONAL_DATA, additional_data.clone().into());
            hm.insert(HADDR_PARENT_HEIGHT, height.0.into());
            hm.insert(HADDR_LAST_HEADER, Value::from(*env.last_header));
            hm.insert(HADDR_SPENDER_INDEX, Value::from(env.spender_index as u64));
        }

        Executor::new(instrs, hm)
    }
    fn do_triop(&mut self, op: impl Fn(Value, Value, Value) -> Option<Value>) -> Option<()> {
        let stack = &mut self.stack;
        let x = stack.pop()?;
        let y = stack.pop()?;
        let z = stack.pop()?;
        stack.push(op(x, y, z)?);
        Some(())
    }
    fn do_binop(&mut self, op: impl Fn(Value, Value) -> Option<Value>) -> Option<()> {
        let stack = &mut self.stack;
        let x = stack.pop()?;
        let y = stack.pop()?;
        stack.push(op(x, y)?);
        // eprintln!("stack at {}", stack.len());
        Some(())
    }
    fn do_monop(&mut self, op: impl Fn(Value) -> Option<Value>) -> Option<()> {
        let stack = &mut self.stack;
        let x = stack.pop()?;
        stack.push(op(x)?);
        Some(())
    }
    pub fn pc(&self) -> ProgramCounter {
        self.pc
    }
    /// Update program pointer state (for loops etc)
    fn update_pc_state(&mut self) {
        while let Some(mut state) = self.loop_state.pop() {
            // If done with body of loop
            if self.pc > state.end {
                // But not finished with all iterations, and did not jump outside the loop
                if state.iterations_left > 0 && self.pc.saturating_sub(state.end) == 1 {
                    log::trace!("{} iterations left", state.iterations_left);
                    // loop again
                    state.iterations_left -= 1;
                    self.pc = state.begin;
                    self.loop_state.push(state);
                    break;
                }
            } else {
                // If not done with loop body, resume
                self.loop_state.push(state);
                break;
            }
        }
    }

    /// Execute to the end
    pub fn run_to_end(&mut self) -> bool {
        while self.pc < self.instrs.len() {
            if self.step().is_none() {
                return false;
            }
        }

        self.stack.pop().map(|f| f.into_bool()).unwrap_or_default()
    }

    /// Execute an instruction, modifying state and program counter.
    #[inline(always)]
    pub fn step(&mut self) -> Option<()> {
        let mut inner = || {
            let op = self.instrs.get(self.pc)?.clone();
            // eprintln!("OPS: {:?}", self.instrs);
            // eprintln!("PC:  {}", self.pc);
            // eprintln!("OP:  {:?}", op);
            // eprintln!("STK: {:?}", self.stack);
            // eprintln!();
            self.pc += 1;
            // eprintln!("running {:?}", op);
            match op {
                OpCode::Noop => {
                    dbg!("NoOp");
                }
                // arithmetic
                OpCode::Add => self.do_binop(|x, y| {
                    dbg!("Addition, First: {}", &x);
                    dbg!("Addition, Second: {}", &y);

                    Some(Value::Int(x.into_int()?.overflowing_add(y.into_int()?).0))
                })?,
                OpCode::Sub => self.do_binop(|x, y| {
                    dbg!("Subtraction, First: {}", &x);
                    dbg!("Subtraction, Second: {}", &y);

                    Some(Value::Int(x.into_int()?.overflowing_sub(y.into_int()?).0))
                })?,
                OpCode::Mul => self.do_binop(|x, y| {
                    dbg!("Multiplication, First: {}", &x);
                    dbg!("Multiplication, Second: {}", &y);

                    Some(Value::Int(x.into_int()?.overflowing_mul(y.into_int()?).0))
                })?,
                OpCode::Div => self
                    .do_binop(|x, y| {
                        dbg!("Division, First: {}", &x);
                        dbg!("Division, Second: {}", &y);

                        Some(Value::Int(x.into_int()?.checked_div(y.into_int()?)?))
                    })?,
                OpCode::Rem => self
                    .do_binop(|x, y| {
                        dbg!("Remainder, First: {}", &x);
                        dbg!("Remainder, Second: {}", &y);

                        Some(Value::Int(x.into_int()?.checked_rem(y.into_int()?)?))
                    })?,
                // logic
                OpCode::And => {
                    self.do_binop(|x, y| {
                        Some(Value::Int(x.into_int()? & y.into_int()?))
                    })?
                }
                OpCode::Or => {
                    self.do_binop(|x, y| {
                        Some(Value::Int(x.into_int()? | y.into_int()?))
                    })?
                }
                OpCode::Xor => {
                    self.do_binop(|x, y| {
                        Some(Value::Int(x.into_int()? ^ y.into_int()?))
                    })?
                }
                OpCode::Not => self.do_monop(|x| {
                    Some(Value::Int(!x.into_int()?))
                })?,
                OpCode::Eql => self.do_binop(|x, y| match (x, y) {
                    (Value::Int(x), Value::Int(y)) => {
                        dbg!("Equality, First: {}", &x);
                        dbg!("Equality, Second: {}", &y);
                        if x == y {
                            Some(Value::Int(1u32.into()))
                        } else {
                            Some(Value::Int(0u32.into()))
                        }
                    }
                    _ => None,
                })?,
                OpCode::Lt => self.do_binop(|x, y| {
                    dbg!("Less than, First: {}", &x);
                    dbg!("Less than, Second: {}", &y);

                    let x = x.into_int()?;
                    let y = y.into_int()?;
                    if x < y {
                        Some(Value::Int(1u32.into()))
                    } else {
                        Some(Value::Int(0u32.into()))
                    }
                })?,
                OpCode::Gt => self.do_binop(|x, y| {
                    dbg!("Greater than, First: {}", &x);
                    dbg!("Greater than, Second: {}", &y);

                    let x = x.into_int()?;
                    let y = y.into_int()?;
                    if x > y {
                        Some(Value::Int(1u32.into()))
                    } else {
                        Some(Value::Int(0u32.into()))
                    }
                })?,
                OpCode::Shl => self.do_binop(|x, offset| {
                    let x = x.into_int()?;
                    let offset = offset.into_int()?;

                    Some(Value::Int(x.wrapping_shl(offset.as_u32())))
                })?,
                OpCode::Shr => self.do_binop(|x, offset| {
                    let x = x.into_int()?;
                    let offset = offset.into_int()?;

                    Some(Value::Int(x.wrapping_shr(offset.as_u32())))
                })?,
                // cryptography
                OpCode::Hash(n) => self.do_monop(|to_hash| {
                    let bytes: CatVec<u8, 256> = to_hash.into_bytes()?;

                    if bytes.len() > n as usize {
                        return None;
                    }

                    let byte_vector: Vec<u8> = bytes.into();
                    let hash: tmelcrypt::HashVal = tmelcrypt::hash_single(&byte_vector);

                    dbg!("Hash: {}", &hash.0);

                    Some(Value::from_bytes(&hash.0))
                })?,
                OpCode::SigEOk(n) => self.do_triop(|message, public_key, signature| {
                    //println!("SIGEOK({:?}, {:?}, {:?})", message, public_key, signature);
                    let public_key_bytes: CatVec<u8, 256> = public_key.into_bytes()?;

                    if public_key_bytes.len() > 32 {
                        return Some(Value::from_bool(false));
                    }

                    let public_key_byte_vector: Vec<u8> = public_key_bytes.into();
                    let public_key: tmelcrypt::Ed25519PK = tmelcrypt::Ed25519PK::from_bytes(&public_key_byte_vector)?;
                    let message_bytes: CatVec<u8, 256> = message.into_bytes()?;

                    if message_bytes.len() > n as usize {
                        return None;
                    }

                    let message_byte_vector: Vec<u8> = message_bytes.into();
                    let signature_bytes: CatVec<u8, 256> = signature.into_bytes()?;

                    if signature_bytes.len() > 64 {
                        return Some(Value::from_bool(false));
                    }

                    let signature_byte_vector: Vec<u8> = signature_bytes.into();

                    Some(Value::from_bool(public_key.verify(&message_byte_vector, &signature_byte_vector)))
                })?,
                // storage access
                OpCode::Store => {
                    let address: u16 = self.stack.pop()?.into_u16()?;
                    let value: Value = self.stack.pop()?;

                    dbg!("Storing {} at address: {} on the heap.", &value, &address);

                    self.heap.insert(address, value);
                }
                OpCode::Load => {
                    let address: u16 = self.stack.pop()?.into_u16()?;
                    let res: Value = self.heap.get(&address)?.clone();

                    dbg!("Loading {} from address: {} from the heap.", &res, &address);

                    self.stack.push(res)
                }
                OpCode::StoreImm(idx) => {
                    let value: Value = self.stack.pop()?;

                    dbg!("Storing {} at index {} immutably on the heap.", &value, &idx);

                    self.heap.insert(idx, value);
                }
                OpCode::LoadImm(idx) => {
                    let res = self.heap.get(&idx)?.clone();

                    dbg!("Loading {} from index {} immutably from the heap.", &res, &idx);

                    self.stack.push(res)
                }
                // vector operations
                OpCode::VRef => self.do_binop(|vec, idx| {
                    let idx: usize = idx.into_u16()? as usize;

                    dbg!("Loading index {} from VM vector containing {} onto the stack.", &idx, &vec);

                    Some(vec.into_vector()?.get(idx)?.clone())
                })?,
                OpCode::VSet => self.do_triop(|vec, idx, value| {
                    let idx: usize = idx.into_u16()? as usize;
                    let mut vec: CatVec<Value, 32> = vec.into_vector()?;

                    dbg!("Overwriting index {} of a VM vector containing {} with {}", &idx, &vec, &value);

                    *vec.get_mut(idx)? = value;

                    Some(Value::Vector(vec))
                })?,
                OpCode::VAppend => self.do_binop(|v1, v2| {
                    let mut v1 = v1.into_vector()?;
                    let v2 = v2.into_vector()?;

                    dbg!("Appending a vector that contains {} to a vector that contains {}", &v2, &v1);

                    v1.append(v2);

                    Some(Value::Vector(v1))
                })?,
                OpCode::VSlice => self.do_triop(|vec, beginning_value, end_value| {
                    let beginning: usize = beginning_value.into_u16()? as usize;
                    let end: usize = end_value.into_u16()? as usize;

                    match vec {
                        Value::Vector(vec) => {
                            let is_end_greater_or_equal_to_vector_length: bool = end >= vec.len();
                            let is_end_less_than_or_equal_to_beginning: bool = end <= beginning;

                            if is_end_greater_or_equal_to_vector_length || is_end_less_than_or_equal_to_beginning {
                                dbg!("Tried to create a VM slice with invalid bounds. Returning an empty VM vector.");

                                Some(Value::Vector(Default::default()))
                            } else {
                                dbg!("Returning a slice from {} to {} from the VM vector containing: {}", beginning, end, &vec);

                                Some(Value::Vector(vec.tap_mut(|vec| vec.slice_into(beginning..end))))
                            }
                        }
                        _ => {
                            dbg!("Tried to call VSlice on something that was not a VM vector (Value::Vector).");

                            None
                        },
                    }
                })?,
                OpCode::VLength => self.do_monop(|vec| match vec {
                    Value::Vector(vec) => {
                        let length: usize = vec.len();

                        dbg!("VM vector is of length: {}", length);

                        Some(Value::Int(U256::from(length as u64)))
                    },
                    _ => {
                        dbg!("Tried to call VLength on something that was not a VM vector (Value::Vector).");

                        None
                    },
                })?,
                OpCode::VEmpty => {
                    dbg!("Creating a new empty vector on the stack.");

                    self.stack.push(Value::Vector(Default::default()))
                },
                OpCode::VPush => self.do_binop(|vec, item| {
                    let mut vec: CatVec<Value, 32> = vec.into_vector()?;

                    dbg!("Pushing: {} into a VM vector that contains: {}.", &item, &vec);

                    vec.push_back(item);

                    Some(Value::Vector(vec))
                })?,
                OpCode::VCons => self.do_binop(|item, vec| {
                    let mut vec: CatVec<Value, 32> = vec.into_vector()?;

                    dbg!("Inserting: {} at index 0 of a VM vector that contains: {}", &item, &vec);

                    vec.insert(0, item);

                    Some(Value::Vector(vec))
                })?,
                // bit stuff
                OpCode::BEmpty => {
                    dbg!("Creating a new empty byte vector on the stack.");

                    self.stack.push(Value::Bytes(Default::default()))
                },
                OpCode::BPush => self.do_binop(|vec, val| {
                    let mut vec: CatVec<u8, 256> = vec.into_bytes()?;
                    let val: U256 = val.into_int()?;

                    dbg!("Pushing: {} into a byte vector containing: {}", &val, &vec);

                    vec.push_back(*val.low() as u8);

                    Some(Value::Bytes(vec))
                })?,
                OpCode::BCons => self.do_binop(|item, vec| {
                    let mut vec: CatVec<u8, 256> = vec.into_bytes()?;

                    dbg!("Inserting: {} at index 0 of a byte vector that contains: {}", &item, &vec);

                    vec.insert(0, item.into_truncated_u8()?);

                    Some(Value::Bytes(vec))
                })?,
                OpCode::BRef => self.do_binop(|vec, idx| {
                    let idx: usize = idx.into_u16()? as usize;

                    dbg!("Loading index {} from a stack byte vector containing {} onto the stack.", &idx, &vec);

                    Some(Value::Int(vec.into_bytes()?.get(idx).copied()?.into()))
                })?,
                OpCode::BSet => self.do_triop(|vec, idx, value| {
                    let idx: usize = idx.into_u16()? as usize;
                    let mut vec: CatVec<u8, 256> = vec.into_bytes()?;

                    dbg!("Overwriting index {} of a byte vector containing {} with {}", &idx, &vec, &value);

                    *vec.get_mut(idx)? = value.into_truncated_u8()?;

                    Some(Value::Bytes(vec))
                })?,
                OpCode::BAppend => self.do_binop(|v1, v2| {
                    let mut v1: CatVec<u8, 256> = v1.into_bytes()?;
                    let v2: CatVec<u8, 256> = v2.into_bytes()?;

                    dbg!("Appending a vector that contains {} to a vector that contains {}", &v2, &v1);

                    v1.append(v2);

                    Some(Value::Bytes(v1))
                })?,
                OpCode::BSlice => self.do_triop(|vec, beginning_value, end_value| {
                    let beginning: usize = beginning_value.into_u16()? as usize;
                    let end: usize = end_value.into_u16()? as usize;

                    match vec {
                        Value::Bytes(mut vec) => {
                            let is_end_greater_or_equal_to_vector_length: bool = end >= vec.len();
                            let is_end_less_than_or_equal_to_beginning: bool = end <= beginning;

                            if is_end_greater_or_equal_to_vector_length || is_end_less_than_or_equal_to_beginning {
                                dbg!("Tried to create a byte slice with invalid bounds. Returning an empty byte vector.");

                                Some(Value::Bytes(Default::default()))
                            } else {
                                dbg!("Returning a byte slice from {} to {} from the byte vector containing: {}", beginning, end, &vec);

                                vec.slice_into(beginning..end);

                                Some(Value::Bytes(vec))
                            }
                        }
                        _ => {
                            dbg!("Tried to call VSlice on something that was not a VM vector (Value::Vector).");

                            None
                        },
                    }
                })?,
                OpCode::BLength => self.do_monop(|vec| match vec {
                    Value::Bytes(vec) => {
                        let length: usize = vec.len();

                        dbg!("Byte vector is of length: {}", length);

                        Some(Value::Int(U256::from(length as u64)))
                    },
                    _ => {
                        dbg!("Tried to call BLength on something that was not a byte vector (Value::Bytes).");

                        None
                    },
                })?,
                // control flow
                OpCode::Bez(jgap) => {
                    let top = self.stack.pop()?;

                    if top.into_int() == Some(0u32.into()) {
                        dbg!("In a call to Bez, the top of the stack was zero. Skipping to {}", &jgap);

                        self.pc += jgap as usize;

                        return Some(());
                    } else {
                        dbg!("In a call to Bez, the top of the stack was not zero. It was {}. Not skipping any operations.", &jgap);
                    }
                }
                OpCode::Bnz(jgap) => {
                    let top = self.stack.pop()?;

                    if top.into_int() != Some(0u32.into()) {
                        dbg!("In a call to Bnz, the top of the stack was not zero. Skipping to {}", &jgap);

                        self.pc += jgap as usize;
                        return Some(());
                    } else {
                        dbg!("In a call to Bnz, the top of the stack was not zero. It was {}. Not skipping any operations.", &jgap);
                    }
                }
                OpCode::Jmp(jgap) => {
                    dbg!("Jumping ahead to instruction number {}", &jgap);

                    self.pc += jgap as usize;
                    return Some(());
                }
                OpCode::Loop(iterations, op_count) => {
                    let is_iterations_positive: bool = iterations > 0;
                    let is_op_count_positive: bool = op_count > 0;

                    if is_iterations_positive && is_op_count_positive {
                        dbg!("In a call to Loop, iterations and op_count were positive. Looping {} times.", iterations);

                        self.loop_state.push(LoopState {
                            // start after loop instruction
                            begin: self.pc,
                            // final op is inclusive
                            end: self.pc + op_count as usize - 1,
                            // dec happens after an iteration so -1 for first loop
                            iterations_left: iterations - 1,
                        });
                    } else {
                        if !is_iterations_positive {
                            dbg!("In a call to Loop, iterations was not positive: {}. Skipping loop.", iterations);
                        } else if !is_op_count_positive {
                            dbg!("In a call to Loop, op_count was not positive: {}. Skipping loop.", iterations);
                        } else {
                            dbg!("In a call to Loop, neither iterations: {}, nor op_count were positive: {}. Skipping loop.", iterations, op_count);
                        }

                        return None;
                    }
                }
                // Conversions
                OpCode::BtoI => self.do_monop(|input_byte_vector| {
                    dbg!("Converting bytes {} into an integer.", &input_byte_vector);

                    let bytes = input_byte_vector.into_bytes()?;
                    let bytes_vector: Vec<u8> = bytes.into();

                    let byte_vector_option: Option<[u8; 32]> = bytes_vector.try_into().ok();

                    match byte_vector_option {
                        Some(byte_vector) => {
                            dbg!("In a call to BtoI, successfully converted input bytes to an integer.");

                            Some(Value::Int(U256::from_be_bytes(byte_vector)))
                        },
                        None => {
                            dbg!("In a call to BtoI, failed to convert input bytes to an integer.");

                            None
                        },
                    }
                })?,
                OpCode::ItoB => self.do_monop(|input_integer| {
                    let number_option: Option<U256> = input_integer.into_int();

                    // I may not need this match, because it may not be able to fail, given that only positive integers are available.

                    match number_option {
                        Some(number) => {
                            dbg!("In a call to ItoB, successfully converted input integer to bytes.");

                            Some(Value::Bytes(number.to_be_bytes().into()))
                        },
                        None => {
                            dbg!("In a call to ItoB, failed to convert input integer to bytes.");

                            None
                        },
                    }
                })?,
                // literals
                OpCode::PushB(bts) => {
                    let bts = Value::from_bytes(&bts);
                    self.stack.push(bts);
                }
                OpCode::PushI(num) => self.stack.push(Value::Int(num)),
                OpCode::PushIC(num) => self.stack.push(Value::Int(num)),
                OpCode::TypeQ => self.do_monop(|x| match x {
                    Value::Int(_) => Some(Value::Int(0u32.into())),
                    Value::Bytes(_) => Some(Value::Int(1u32.into())),
                    Value::Vector(_) => Some(Value::Int(2u32.into())),
                })?,
                // dup
                OpCode::Dup => {
                    let val = self.stack.pop()?;
                    self.stack.push(val.clone());
                    self.stack.push(val);
                }
            }
            Some(())
        };
        let res: Option<()> = inner();
        self.update_pc_state();

        res
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Int(U256),
    Bytes(CatVec<u8, 256>),
    Vector(CatVec<Value, 32>),
}

impl Value {
    fn into_bool(self) -> bool {
        match self {
            Value::Int(v) => v != U256::from(0u32),
            _ => true,
        }
    }

    fn into_int(self) -> Option<U256> {
        match self {
            Value::Int(bi) => Some(bi),
            _ => None,
        }
    }
    fn into_u16(self) -> Option<u16> {
        let num = self.into_int()?;
        if num > U256::from(65535u32) {
            None
        } else {
            Some(*num.low() as u16)
        }
    }
    fn into_truncated_u8(self) -> Option<u8> {
        let num = self.into_int()?;
        Some(*num.low() as u8)
    }
    pub fn from_bytes(bts: &[u8]) -> Self {
        Value::Bytes(bts.into())
    }
    fn from_bool(b: bool) -> Self {
        if b {
            Value::Int(1u32.into())
        } else {
            Value::Int(0u32.into())
        }
    }

    fn into_bytes(self) -> Option<CatVec<u8, 256>> {
        match self {
            Value::Bytes(bts) => Some(bts),
            _ => None,
        }
    }

    fn into_vector(self) -> Option<CatVec<Value, 32>> {
        match self {
            Value::Vector(vec) => Some(vec),
            _ => None,
        }
    }
}

impl From<u128> for Value {
    fn from(n: u128) -> Self {
        Value::Int(U256::from(n))
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Value::Int(U256::from(n))
    }
}

impl From<CoinData> for Value {
    fn from(cd: CoinData) -> Self {
        Value::Vector(
            vec![
                cd.covhash.0.into(),
                cd.value.0.into(),
                cd.denom.into(),
                cd.additional_data.into(),
            ]
            .into(),
        )
    }
}

impl From<Header> for Value {
    fn from(cd: Header) -> Self {
        Value::Vector(
            vec![
                (cd.network as u64).into(),
                cd.previous.into(),
                cd.height.0.into(),
                cd.history_hash.into(),
                cd.coins_hash.into(),
                cd.transactions_hash.into(),
                cd.fee_pool.0.into(),
                cd.fee_multiplier.into(),
                cd.dosc_speed.into(),
                cd.pools_hash.into(),
                cd.stakes_hash.into(),
            ]
            .into(),
        )
    }
}

impl From<CoinDataHeight> for Value {
    fn from(cd: CoinDataHeight) -> Self {
        Value::Vector(vec![cd.coin_data.into(), cd.height.0.into()].into())
    }
}

impl From<CoinID> for Value {
    fn from(c: CoinID) -> Self {
        Value::Vector(vec![c.txhash.0.into(), Value::Int(U256::from(c.index))].into())
    }
}

impl From<Covenant> for Value {
    fn from(c: Covenant) -> Self {
        Value::Bytes(c.0.into())
    }
}

impl From<[u8; 32]> for Value {
    fn from(v: [u8; 32]) -> Self {
        Value::Bytes(v.into())
    }
}

impl From<HashVal> for Value {
    fn from(v: HashVal) -> Self {
        Value::Bytes(v.into())
    }
}

impl From<Denom> for Value {
    fn from(v: Denom) -> Self {
        Value::Bytes(v.to_bytes().into())
    }
}

impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bytes(v.into())
    }
}

impl From<HexBytes> for Value {
    fn from(v: HexBytes) -> Self {
        Value::Bytes(v.0.into())
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Vector(
            v.into_iter()
                .map(|x| x.into())
                .collect::<Vec<Value>>()
                .into(),
        )
    }
}

impl From<Transaction> for Value {
    fn from(tx: Transaction) -> Self {
        Value::Vector(
            vec![
                Value::Int(U256::from(tx.kind as u8)),
                tx.inputs.into(),
                tx.outputs.into(),
                tx.fee.0.into(),
                tx.scripts.into(),
                tx.data.into(),
                tx.sigs.into(),
            ]
            .into(),
        )
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

        range.into_iter().for_each(|_index| {
            data.push(0xb0)
        });

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