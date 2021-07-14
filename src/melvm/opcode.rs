use std::io::Write;

use super::consts::{
    OPCODE_ADD, OPCODE_AND, OPCODE_BAPPEND, OPCODE_BCONS, OPCODE_BEMPTY, OPCODE_BEZ,
    OPCODE_BLENGTH, OPCODE_BNZ, OPCODE_BPUSH, OPCODE_BREF, OPCODE_BSET, OPCODE_BSLICE, OPCODE_BTOI,
    OPCODE_DIV, OPCODE_DUP, OPCODE_EQL, OPCODE_GT, OPCODE_HASH, OPCODE_ITOB, OPCODE_JMP,
    OPCODE_LOAD, OPCODE_LOADIMM, OPCODE_LOOP, OPCODE_LT, OPCODE_MUL, OPCODE_NOOP, OPCODE_NOT,
    OPCODE_OR, OPCODE_PUSHB, OPCODE_PUSHI, OPCODE_REM, OPCODE_SHL, OPCODE_SHR, OPCODE_SIGEOK,
    OPCODE_STORE, OPCODE_STOREIMM, OPCODE_SUB, OPCODE_TYPEQ, OPCODE_VAPPEND, OPCODE_VCONS,
    OPCODE_VEMPTY, OPCODE_VLENGTH, OPCODE_VPUSH, OPCODE_VREF, OPCODE_VSET, OPCODE_VSLICE,
    OPCODE_XOR,
};
use ethnum::U256;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum OpCode {
    Noop,
    // arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // logic
    And,
    Or,
    Xor,
    Not,
    Eql,
    Lt,
    Gt,
    Shl,
    Shr,
    // cryptographyy
    Hash(u16),
    //SIGE,
    //SIGQ,
    SigEOk(u16),
    //SIGQOK,
    // "heap" access
    Store,
    Load,
    StoreImm(u16),
    LoadImm(u16),
    // vector operations
    VRef,
    VAppend,
    VEmpty,
    VLength,
    VSlice,
    VSet,
    VPush,
    VCons,
    // bytes operations
    BRef,
    BAppend,
    BEmpty,
    BLength,
    BSlice,
    BSet,
    BPush,
    BCons,

    // control flow
    Bez(u16),
    Bnz(u16),
    Jmp(u16),
    // Loop(iterations, instructions)
    Loop(u16, u16),

    // type conversions
    ItoB,
    BtoI,
    TypeQ,
    // SERIAL(u16),

    // literals
    PushB(Vec<u8>),
    PushI(U256),

    // duplication
    Dup,
}

/// Opcode encoding error
#[derive(Error, Debug)]
pub enum EncodeError {
    #[error("PushB has too many bytes")]
    TooManyBytes,
}

/// Opcode decoding error
#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("I/O error: {0:?}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid opcode {0}")]
    InvalidOpcode(u8),
}

fn read_byte<T: std::io::Read>(input: &mut T) -> std::io::Result<u8> {
    let mut z = [0; 1];
    input.read_exact(&mut z)?;
    Ok(z[0])
}

impl OpCode {
    /// Encodes an opcode.
    pub fn encode(&self) -> Result<Vec<u8>, EncodeError> {
        let mut output = Vec::new();
        match self {
            OpCode::Noop => output.write_all(&[OPCODE_NOOP]).unwrap(),

            OpCode::Add => output.write_all(&[OPCODE_ADD]).unwrap(),
            OpCode::Sub => output.write_all(&[OPCODE_SUB]).unwrap(),
            OpCode::Mul => output.write_all(&[OPCODE_MUL]).unwrap(),
            OpCode::Div => output.write_all(&[OPCODE_DIV]).unwrap(),
            OpCode::Rem => output.write_all(&[OPCODE_REM]).unwrap(),

            OpCode::And => output.write_all(&[OPCODE_AND]).unwrap(),
            OpCode::Or => output.write_all(&[OPCODE_OR]).unwrap(),
            OpCode::Xor => output.write_all(&[OPCODE_XOR]).unwrap(),
            OpCode::Not => output.write_all(&[OPCODE_NOT]).unwrap(),
            OpCode::Eql => output.write_all(&[OPCODE_EQL]).unwrap(),

            OpCode::Lt => output.write_all(&[OPCODE_LT]).unwrap(),
            OpCode::Gt => output.write_all(&[OPCODE_GT]).unwrap(),
            OpCode::Shl => output.write_all(&[OPCODE_SHL]).unwrap(),
            OpCode::Shr => output.write_all(&[OPCODE_SHR]).unwrap(),

            OpCode::Hash(i) => {
                output.write_all(&[OPCODE_HASH]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap()
            }
            OpCode::SigEOk(i) => {
                output.write_all(&[OPCODE_SIGEOK]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap()
            }

            OpCode::Load => output.write_all(&[OPCODE_LOAD]).unwrap(),
            OpCode::Store => output.write_all(&[OPCODE_STORE]).unwrap(),
            OpCode::LoadImm(i) => {
                output.write_all(&[OPCODE_LOADIMM]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap()
            }
            OpCode::StoreImm(i) => {
                output.write_all(&[OPCODE_STOREIMM]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap()
            }

            OpCode::VRef => output.write_all(&[OPCODE_VREF]).unwrap(),
            OpCode::VAppend => output.write_all(&[OPCODE_VAPPEND]).unwrap(),
            OpCode::VEmpty => output.write_all(&[OPCODE_VEMPTY]).unwrap(),
            OpCode::VLength => output.write_all(&[OPCODE_VLENGTH]).unwrap(),
            OpCode::VSlice => output.write_all(&[OPCODE_VSLICE]).unwrap(),
            OpCode::VSet => output.write_all(&[OPCODE_VSET]).unwrap(),
            OpCode::VPush => output.write_all(&[OPCODE_VPUSH]).unwrap(),
            OpCode::VCons => output.write_all(&[OPCODE_VCONS]).unwrap(),

            OpCode::BRef => output.write_all(&[OPCODE_BREF]).unwrap(),
            OpCode::BAppend => output.write_all(&[OPCODE_BAPPEND]).unwrap(),
            OpCode::BEmpty => output.write_all(&[OPCODE_BEMPTY]).unwrap(),
            OpCode::BLength => output.write_all(&[OPCODE_BLENGTH]).unwrap(),
            OpCode::BSlice => output.write_all(&[OPCODE_BSLICE]).unwrap(),
            OpCode::BSet => output.write_all(&[OPCODE_BSET]).unwrap(),
            OpCode::BPush => output.write_all(&[OPCODE_BPUSH]).unwrap(),
            OpCode::BCons => output.write_all(&[OPCODE_BCONS]).unwrap(),

            OpCode::Jmp(i) => {
                output.write_all(&[OPCODE_JMP]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap()
            }
            OpCode::Bez(i) => {
                output.write_all(&[OPCODE_BEZ]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap()
            }
            OpCode::Bnz(i) => {
                output.write_all(&[OPCODE_BNZ]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap()
            }
            OpCode::Loop(iter, count) => {
                output.write_all(&[OPCODE_LOOP]).unwrap();
                output.write_all(&iter.to_be_bytes()).unwrap();
                output.write_all(&count.to_be_bytes()).unwrap()
            }

            OpCode::BtoI => output.write_all(&[OPCODE_BTOI]).unwrap(),
            OpCode::ItoB => output.write_all(&[OPCODE_ITOB]).unwrap(),
            OpCode::TypeQ => output.write_all(&[OPCODE_TYPEQ]).unwrap(),

            OpCode::PushB(bts) => {
                if bts.len() > 255 {
                    return Err(EncodeError::TooManyBytes);
                }
                output.write_all(&[OPCODE_PUSHB]).unwrap();
                output.write_all(&[bts.len() as u8]).unwrap();
                output.write_all(&bts).unwrap();
            }
            OpCode::PushI(i) => {
                output.write_all(&[OPCODE_PUSHI]).unwrap();
                output.write_all(&i.to_be_bytes()).unwrap();
            }
            OpCode::Dup => output.write_all(&[OPCODE_DUP]).unwrap(),
        };
        Ok(output)
    }

    /// Decodes an opcode from an input.
    pub fn decode<T: std::io::Read>(input: &mut T) -> Result<Self, DecodeError> {
        let u16arg = |input: &mut T| {
            let mut z = [0; 2];
            input.read_exact(&mut z)?;
            Ok::<_, DecodeError>(u16::from_be_bytes(z))
        };
        match read_byte(input)? {
            OPCODE_NOOP => Ok(OpCode::Noop),
            // arithmetic
            OPCODE_ADD => Ok(OpCode::Add),
            OPCODE_SUB => Ok(OpCode::Sub),
            OPCODE_MUL => Ok(OpCode::Mul),
            OPCODE_DIV => Ok(OpCode::Div),
            OPCODE_REM => Ok(OpCode::Rem),
            // logic
            OPCODE_AND => Ok(OpCode::And),
            OPCODE_OR => Ok(OpCode::Or),
            OPCODE_XOR => Ok(OpCode::Xor),
            OPCODE_NOT => Ok(OpCode::Not),
            OPCODE_EQL => Ok(OpCode::Eql),
            OPCODE_LT => Ok(OpCode::Lt),
            OPCODE_GT => Ok(OpCode::Gt),
            OPCODE_SHL => Ok(OpCode::Shl),
            OPCODE_SHR => Ok(OpCode::Shr),
            // cryptography
            OPCODE_HASH => Ok(OpCode::Hash(u16arg(input)?)),
            //0x31 => Ok(OpCode::SIGE),
            OPCODE_SIGEOK => Ok(OpCode::SigEOk(u16arg(input)?)),
            // storage
            OPCODE_LOAD => Ok(OpCode::Load),
            OPCODE_STORE => Ok(OpCode::Store),
            OPCODE_LOADIMM => Ok(OpCode::LoadImm(u16arg(input)?)),
            OPCODE_STOREIMM => Ok(OpCode::StoreImm(u16arg(input)?)),
            // vectors
            OPCODE_VREF => Ok(OpCode::VRef),
            OPCODE_VAPPEND => Ok(OpCode::VAppend),
            OPCODE_VEMPTY => Ok(OpCode::VEmpty),
            OPCODE_VLENGTH => Ok(OpCode::VLength),
            OPCODE_VSLICE => Ok(OpCode::VSlice),
            OPCODE_VSET => Ok(OpCode::VSet),
            OPCODE_VPUSH => Ok(OpCode::VPush),
            OPCODE_VCONS => Ok(OpCode::VCons),
            // bytes
            OPCODE_BREF => Ok(OpCode::BRef),
            OPCODE_BAPPEND => Ok(OpCode::BAppend),
            OPCODE_BEMPTY => Ok(OpCode::BEmpty),
            OPCODE_BLENGTH => Ok(OpCode::BLength),
            OPCODE_BSLICE => Ok(OpCode::BSlice),
            OPCODE_BSET => Ok(OpCode::BSet),
            OPCODE_BPUSH => Ok(OpCode::BPush),
            OPCODE_BCONS => Ok(OpCode::BCons),
            // control flow
            OPCODE_JMP => Ok(OpCode::Jmp(u16arg(input)?)),
            OPCODE_BEZ => Ok(OpCode::Bez(u16arg(input)?)),
            OPCODE_BNZ => Ok(OpCode::Bnz(u16arg(input)?)),
            OPCODE_LOOP => {
                let iterations = u16arg(input)?;
                let count = u16arg(input)?;
                Ok(OpCode::Loop(iterations, count))
            }
            OPCODE_ITOB => Ok(OpCode::ItoB),
            OPCODE_BTOI => Ok(OpCode::BtoI),
            // literals
            OPCODE_PUSHB => {
                let strlen = read_byte(input)?;
                let mut blit = vec![0u8; strlen as usize];
                input.read_exact(&mut blit)?;
                Ok(OpCode::PushB(blit))
            }
            OPCODE_PUSHI => {
                let mut buf = [0; 32];
                input.read_exact(&mut buf)?;
                Ok(OpCode::PushI(U256::from_be_bytes(buf)))
            }
            b => Err(DecodeError::InvalidOpcode(b)),
        }
    }
}

/// Computes the weight of a bunch of opcodes.
pub fn opcodes_weight(opcodes: &[OpCode]) -> u128 {
    let (mut sum, mut rest) = opcodes_car_weight(opcodes);
    while !rest.is_empty() {
        let (delta_sum, new_rest) = opcodes_car_weight(rest);
        rest = new_rest;
        sum = sum.saturating_add(delta_sum);
    }
    sum
}

/// Compute the weight of the first bit of opcodes, returning a weight and what remains.
fn opcodes_car_weight(opcodes: &[OpCode]) -> (u128, &[OpCode]) {
    if opcodes.is_empty() {
        return (0, opcodes);
    }
    let (first, rest) = opcodes.split_first().unwrap();
    match first {
        OpCode::Noop => (1, rest),
        // handle loops specially
        OpCode::Loop(iters, body_len) => {
            let mut sum = 0u128;
            let mut rest = rest;
            for _ in 0..*body_len {
                let (weight, rem) = opcodes_car_weight(rest);
                sum = sum.saturating_add(weight);
                rest = rem;
            }
            (sum.saturating_mul(*iters as u128).saturating_add(1), rest)
        }
        OpCode::Add => (4, rest),
        OpCode::Sub => (4, rest),
        OpCode::Mul => (6, rest),
        OpCode::Div => (6, rest),
        OpCode::Rem => (6, rest),

        OpCode::And => (4, rest),
        OpCode::Or => (4, rest),
        OpCode::Xor => (4, rest),
        OpCode::Not => (4, rest),
        OpCode::Eql => (4, rest),
        OpCode::Lt => (4, rest),
        OpCode::Gt => (4, rest),
        OpCode::Shl => (4, rest),
        OpCode::Shr => (4, rest),

        OpCode::Hash(n) => (50u128.saturating_add(*n as u128), rest),
        OpCode::SigEOk(n) => (100u128.saturating_add(*n as u128), rest),

        OpCode::Store => (10, rest),
        OpCode::Load => (10, rest),
        OpCode::StoreImm(_) => (4, rest),
        OpCode::LoadImm(_) => (4, rest),

        OpCode::VRef => (10, rest),
        OpCode::VSet => (20, rest),
        OpCode::VAppend => (50, rest),
        OpCode::VSlice => (50, rest),
        OpCode::VLength => (4, rest),
        OpCode::VEmpty => (4, rest),
        OpCode::BEmpty => (4, rest),
        OpCode::BPush => (10, rest),
        OpCode::VPush => (10, rest),
        OpCode::VCons => (10, rest),
        OpCode::BRef => (10, rest),
        OpCode::BAppend => (10, rest),
        OpCode::BLength => (4, rest),
        OpCode::BSlice => (50, rest),
        OpCode::BSet => (20, rest),
        OpCode::BCons => (10, rest),

        OpCode::TypeQ => (4, rest),

        OpCode::ItoB => (50, rest),
        OpCode::BtoI => (50, rest),

        OpCode::Bez(_) => (1, rest),
        OpCode::Bnz(_) => (1, rest),
        OpCode::Jmp(_) => (1, rest),

        OpCode::PushB(_) => (1, rest),
        OpCode::PushI(_) => (1, rest),

        OpCode::Dup => (4, rest),
    }
}
