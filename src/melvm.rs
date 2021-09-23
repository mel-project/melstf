pub use crate::{CoinData, CoinID, Transaction};
use crate::{CoinDataHeight, Denom, Header, HexBytes};
use arbitrary::Arbitrary;
use derive_more::{From, Into};
use ethnum::U256;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, str::FromStr};
use std::{convert::TryInto, fmt::Display};
use thiserror::Error;
use tmelcrypt::HashVal;

use self::{
    consts::{
        HADDR_LAST_HEADER, HADDR_PARENT_ADDITIONAL_DATA, HADDR_PARENT_DENOM, HADDR_PARENT_HEIGHT,
        HADDR_PARENT_INDEX, HADDR_PARENT_TXHASH, HADDR_PARENT_VALUE, HADDR_SELF_HASH,
        HADDR_SPENDER_INDEX, HADDR_SPENDER_TX, HADDR_SPENDER_TXHASH,
    },
    opcode::{opcodes_weight, DecodeError, EncodeError, OpCode},
};
mod consts;
pub mod opcode;

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

    /// Checks with respect to a manually instantiated initial heap.
    pub fn check_raw(&self, args: &[Value]) -> bool {
        let mut hm = HashMap::new();
        for (i, v) in args.iter().enumerate() {
            hm.insert(i as u16, v.clone());
        }
        if let Ok(ops) = self.to_ops() {
            Executor::new(ops, hm).run_to_end()
        } else {
            false
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
        .unwrap()
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
        .unwrap()
    }

    /// Create a Covenant from a slice of OpCodes.
    pub fn from_ops(ops: &[OpCode]) -> Result<Self, EncodeError> {
        let mut output: Vec<u8> = Vec::new();
        // go through output
        for op in ops {
            output.extend_from_slice(&op.encode()?)
        }
        Ok(Covenant(output))
    }

    pub fn always_true() -> Self {
        Covenant::from_ops(&[OpCode::PushI(1u32.into())]).unwrap()
    }

    /// Converts to a vector of OpCodes.
    pub fn to_ops(&self) -> Result<Vec<OpCode>, DecodeError> {
        let mut collected = Vec::new();
        let mut rdr = self.0.as_slice();
        while !rdr.is_empty() {
            collected.push(OpCode::decode(&mut rdr)?);
        }
        Ok(collected)
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
    pub fn step(&mut self) -> Option<()> {
        let mut inner = || {
            let op = self.instrs.get(self.pc)?.clone();
            // eprintln!("OPS: {:?}", self.instrs);
            // eprintln!("PC:  {}", self.pc);
            // eprintln!("OP:  {:?}", op);
            // eprintln!("STK: {:?}", self.stack);
            // eprintln!();
            self.pc += 1;
            match op {
                OpCode::Noop => {}
                // arithmetic
                OpCode::Add => self.do_binop(|x, y| {
                    Some(Value::Int(x.into_int()?.overflowing_add(y.into_int()?).0))
                })?,
                OpCode::Sub => self.do_binop(|x, y| {
                    Some(Value::Int(x.into_int()?.overflowing_sub(y.into_int()?).0))
                })?,
                OpCode::Mul => self.do_binop(|x, y| {
                    Some(Value::Int(x.into_int()?.overflowing_mul(y.into_int()?).0))
                })?,
                OpCode::Div => self
                    .do_binop(|x, y| Some(Value::Int(x.into_int()?.checked_div(y.into_int()?)?)))?,
                OpCode::Rem => self
                    .do_binop(|x, y| Some(Value::Int(x.into_int()?.checked_rem(y.into_int()?)?)))?,
                // logic
                OpCode::And => {
                    self.do_binop(|x, y| Some(Value::Int(x.into_int()? & y.into_int()?)))?
                }
                OpCode::Or => {
                    self.do_binop(|x, y| Some(Value::Int(x.into_int()? | y.into_int()?)))?
                }
                OpCode::Xor => {
                    self.do_binop(|x, y| Some(Value::Int(x.into_int()? ^ y.into_int()?)))?
                }
                OpCode::Not => self.do_monop(|x| Some(Value::Int(!x.into_int()?)))?,
                OpCode::Eql => self.do_binop(|x, y| match (x, y) {
                    (Value::Int(x), Value::Int(y)) => {
                        if x == y {
                            Some(Value::Int(1u32.into()))
                        } else {
                            Some(Value::Int(0u32.into()))
                        }
                    }
                    _ => None,
                })?,
                OpCode::Lt => self.do_binop(|x, y| {
                    let x = x.into_int()?;
                    let y = y.into_int()?;
                    if x < y {
                        Some(Value::Int(1u32.into()))
                    } else {
                        Some(Value::Int(0u32.into()))
                    }
                })?,
                OpCode::Gt => self.do_binop(|x, y| {
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
                    let to_hash = to_hash.into_bytes()?;
                    if to_hash.len() > n as usize {
                        return None;
                    }
                    let hash = tmelcrypt::hash_single(&to_hash.iter().cloned().collect::<Vec<_>>());
                    Some(Value::from_bytes(&hash.0))
                })?,
                OpCode::SigEOk(n) => self.do_triop(|message, public_key, signature| {
                    //println!("SIGEOK({:?}, {:?}, {:?})", message, public_key, signature);
                    let pk = public_key.into_bytes()?;
                    if pk.len() > 32 {
                        return Some(Value::from_bool(false));
                    }
                    let pk_b: Vec<u8> = pk.iter().cloned().collect();
                    let public_key = tmelcrypt::Ed25519PK::from_bytes(&pk_b)?;
                    let message = message.into_bytes()?;
                    if message.len() > n as usize {
                        return None;
                    }
                    let message: Vec<u8> = message.iter().cloned().collect();
                    let signature = signature.into_bytes()?;
                    if signature.len() > 64 {
                        return Some(Value::from_bool(false));
                    }
                    let signature: Vec<u8> = signature.iter().cloned().collect();
                    Some(Value::from_bool(public_key.verify(&message, &signature)))
                })?,
                // storage access
                OpCode::Store => {
                    let addr = self.stack.pop()?.into_u16()?;
                    let val = self.stack.pop()?;
                    self.heap.insert(addr, val);
                }
                OpCode::Load => {
                    let addr = self.stack.pop()?.into_u16()?;
                    let res = self.heap.get(&addr)?.clone();
                    self.stack.push(res)
                }
                OpCode::StoreImm(idx) => {
                    let val = self.stack.pop()?;
                    self.heap.insert(idx, val);
                }
                OpCode::LoadImm(idx) => {
                    let res = self.heap.get(&idx)?.clone();
                    self.stack.push(res)
                }
                // vector operations
                OpCode::VRef => self.do_binop(|vec, idx| {
                    let idx = idx.into_u16()? as usize;
                    Some(vec.into_vector()?.get(idx)?.clone())
                })?,
                OpCode::VSet => self.do_triop(|vec, idx, value| {
                    let idx = idx.into_u16()? as usize;
                    let mut vec = vec.into_vector()?;
                    if idx < vec.len() {
                        vec.set(idx, value);
                        Some(Value::Vector(vec))
                    } else {
                        None
                    }
                })?,
                OpCode::VAppend => self.do_binop(|v1, v2| {
                    let mut v1 = v1.into_vector()?;
                    let v2 = v2.into_vector()?;
                    v1.append(v2);
                    Some(Value::Vector(v1))
                })?,
                OpCode::VSlice => self.do_triop(|vec, i, j| {
                    let i = i.into_u16()? as usize;
                    let j = j.into_u16()? as usize;
                    match vec {
                        Value::Vector(mut vec) => {
                            if j >= vec.len() || j <= i {
                                Some(Value::Vector(imbl::Vector::new()))
                            } else {
                                Some(Value::Vector(vec.slice(i..j)))
                            }
                        }
                        _ => None,
                    }
                })?,
                OpCode::VLength => self.do_monop(|vec| match vec {
                    Value::Vector(vec) => Some(Value::Int(U256::from(vec.len() as u64))),
                    _ => None,
                })?,
                OpCode::VEmpty => self.stack.push(Value::Vector(imbl::Vector::new())),
                OpCode::VPush => self.do_binop(|vec, item| {
                    let mut vec = vec.into_vector()?;
                    vec.push_back(item);
                    Some(Value::Vector(vec))
                })?,
                OpCode::VCons => self.do_binop(|item, vec| {
                    let mut vec = vec.into_vector()?;
                    vec.push_front(item);
                    Some(Value::Vector(vec))
                })?,
                // bit stuff
                OpCode::BEmpty => self.stack.push(Value::Bytes(imbl::Vector::new())),
                OpCode::BPush => self.do_binop(|vec, val| {
                    let mut vec = vec.into_bytes()?;
                    let val = val.into_int()?;
                    vec.push_back(*val.low() as u8);
                    Some(Value::Bytes(vec))
                })?,
                OpCode::BCons => self.do_binop(|item, vec| {
                    let mut vec = vec.into_bytes()?;
                    vec.push_front(item.into_truncated_u8()?);
                    Some(Value::Bytes(vec))
                })?,
                OpCode::BRef => self.do_binop(|vec, idx| {
                    let idx = idx.into_u16()? as usize;
                    Some(Value::Int(vec.into_bytes()?.get(idx).copied()?.into()))
                })?,
                OpCode::BSet => self.do_triop(|vec, idx, value| {
                    let idx = idx.into_u16()? as usize;
                    let mut vec = vec.into_bytes()?;
                    if idx < vec.len() {
                        vec.set(idx, value.into_truncated_u8()?);
                        Some(Value::Bytes(vec))
                    } else {
                        None
                    }
                })?,
                OpCode::BAppend => self.do_binop(|v1, v2| {
                    let mut v1 = v1.into_bytes()?;
                    let v2 = v2.into_bytes()?;
                    v1.append(v2);
                    Some(Value::Bytes(v1))
                })?,
                OpCode::BSlice => self.do_triop(|vec, i, j| {
                    let i = i.into_u16()? as usize;
                    let j = j.into_u16()? as usize;
                    match vec {
                        Value::Bytes(mut vec) => {
                            if j >= vec.len() || j <= i {
                                Some(Value::Bytes(imbl::Vector::new()))
                            } else {
                                Some(Value::Bytes(vec.slice(i..j)))
                            }
                        }
                        _ => None,
                    }
                })?,
                OpCode::BLength => self.do_monop(|vec| match vec {
                    Value::Bytes(vec) => Some(Value::Int(U256::from(vec.len() as u64))),
                    _ => None,
                })?,
                // control flow
                OpCode::Bez(jgap) => {
                    let top = self.stack.pop()?;
                    if top == Value::Int(0u32.into()) {
                        self.pc += jgap as usize;
                        return Some(());
                    }
                }
                OpCode::Bnz(jgap) => {
                    let top = self.stack.pop()?;
                    if top != Value::Int(0u32.into()) {
                        self.pc += jgap as usize;
                        return Some(());
                    }
                }
                OpCode::Jmp(jgap) => {
                    self.pc += jgap as usize;
                    return Some(());
                }
                OpCode::Loop(iterations, op_count) => {
                    if iterations > 0 && op_count > 0 {
                        self.loop_state.push(LoopState {
                            // start after loop instruction
                            begin: self.pc,
                            // final op is inclusive
                            end: self.pc + op_count as usize - 1,
                            // dec happens after an iteration so -1 for first loop
                            iterations_left: iterations - 1,
                        });
                    } else {
                        return None;
                    }
                }
                // Conversions
                OpCode::BtoI => self.do_monop(|x| {
                    let bytes = x.into_bytes()?;
                    let bytes: [u8; 32] = bytes.into_iter().collect::<Vec<_>>().try_into().ok()?;

                    Some(Value::Int(U256::from_be_bytes(bytes)))
                })?,
                OpCode::ItoB => self.do_monop(|x| {
                    let n = x.into_int()?;
                    Some(Value::Bytes(n.to_be_bytes().iter().copied().collect()))
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
        let res = inner();
        self.update_pc_state();
        res
    }
}

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum Value {
    Int(U256),
    Bytes(imbl::Vector<u8>),
    Vector(imbl::Vector<Value>),
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
        let mut new = imbl::Vector::new();
        for b in bts {
            new.push_back(*b);
        }
        Value::Bytes(new)
    }
    fn from_bool(b: bool) -> Self {
        if b {
            Value::Int(1u32.into())
        } else {
            Value::Int(0u32.into())
        }
    }

    fn into_bytes(self) -> Option<imbl::Vector<u8>> {
        match self {
            Value::Bytes(bts) => Some(bts),
            _ => None,
        }
    }

    fn into_vector(self) -> Option<imbl::Vector<Value>> {
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
        Value::Vector(imbl::vector![
            cd.covhash.0.into(),
            cd.value.0.into(),
            cd.denom.into(),
            cd.additional_data.into()
        ])
    }
}

impl From<Header> for Value {
    fn from(cd: Header) -> Self {
        Value::Vector(imbl::vector![
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
            cd.stakes_hash.into()
        ])
    }
}

impl From<CoinDataHeight> for Value {
    fn from(cd: CoinDataHeight) -> Self {
        Value::Vector(imbl::vector![cd.coin_data.into(), cd.height.0.into()])
    }
}

impl From<CoinID> for Value {
    fn from(c: CoinID) -> Self {
        Value::Vector(imbl::vector![
            c.txhash.0.into(),
            Value::Int(U256::from(c.index))
        ])
    }
}

impl From<Covenant> for Value {
    fn from(c: Covenant) -> Self {
        Value::Bytes(c.0.into())
    }
}

impl From<[u8; 32]> for Value {
    fn from(v: [u8; 32]) -> Self {
        Value::Bytes(v.iter().cloned().collect::<imbl::Vector<u8>>())
    }
}

impl From<HashVal> for Value {
    fn from(v: HashVal) -> Self {
        Value::Bytes(v.iter().cloned().collect::<imbl::Vector<u8>>())
    }
}

impl From<Denom> for Value {
    fn from(v: Denom) -> Self {
        Value::Bytes(v.to_bytes().into_iter().collect::<imbl::Vector<u8>>())
    }
}

impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bytes(v.into_iter().collect::<imbl::Vector<u8>>())
    }
}

impl From<HexBytes> for Value {
    fn from(v: HexBytes) -> Self {
        Value::Bytes(v.0.into_iter().collect::<imbl::Vector<u8>>())
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Vector(
            v.into_iter()
                .map(|x| x.into())
                .collect::<imbl::Vector<Value>>(),
        )
    }
}

impl From<Transaction> for Value {
    fn from(tx: Transaction) -> Self {
        Value::Vector(imbl::vector![
            Value::Int(U256::from(tx.kind as u8)),
            tx.inputs.into(),
            tx.outputs.into(),
            tx.fee.0.into(),
            tx.scripts.into(),
            tx.data.into(),
            tx.sigs.into()
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::*;
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
        for _ in 0..100000 {
            data.push(0xb0)
        }
        dontcrash(&data.to_vec())
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
