use crate::melvm::Covenant;
use crate::melvm::OpCode;
use crate::melvm::{Executor, Value};
use ethnum::U256;
use std::collections::HashMap;
use std::error::Error;

pub fn exec_from_args(args: &[Value]) -> Executor {
    let mut hm = HashMap::new();
    for (i, v) in args.iter().enumerate() {
        hm.insert(i as u16, v.clone());
    }
    Executor::new(hm)
}

fn do_math_ops(ops: &[OpCode; 3]) -> Option<Value> {
    let mut ex = exec_from_args(&[]);
    println!("Trying {:?}", ops);

    for op in ops.iter() {
        ex.do_op(&op);
    }

    println!("result: {:?}\n---", ex.stack);
    ex.stack.pop()
}

fn do_math(op: OpCode, args: &[U256]) -> Value {
    do_math_ops(&[OpCode::PushI(args[0]), OpCode::PushI(args[1]), op]).unwrap()
}

fn do_math_int(op: OpCode, args: &[u128]) -> Value {
    let i_args: Vec<U256> = args.into_iter().map(|x| U256::from(x.clone())).collect();
    do_math(op, &i_args[..])
}

fn test_math_int(op: OpCode, args: &[u128]) -> bool {
    let val = do_math_int(op, &args[..2]);
    val == Value::from(args[2])
}

#[test]
fn test_noop() {
    let cov = Covenant::from_ops(&[OpCode::Noop]).unwrap();
    assert_eq!(cov.check_raw(&[]), false)
}

#[test]
fn test_add() {
    assert!(test_math_int(OpCode::Add, &[1, 2, 3]));
    assert!(!test_math_int(OpCode::Add, &[1, 2, 4]));
}
#[test]
fn test_sub() {
    let overflow: U256 = U256::MAX;
    assert!(test_math_int(OpCode::Sub, &[1, 1, 0]));
    assert!(test_math_int(OpCode::Sub, &[1, 2, 1]));
    assert!(!test_math_int(OpCode::Sub, &[1, 2, 4]));
    assert!(do_math_int(OpCode::Sub, &[1, 0]) == Value::Int(overflow));
}

#[test]
fn test_mul() {
    assert!(test_math_int(OpCode::Mul, &[1, 1, 1]));
    assert!(test_math_int(OpCode::Mul, &[4, 2, 8]));
    assert!(!test_math_int(OpCode::Mul, &[1, 2, 4]));
}

#[test]
fn test_div() {
    assert!(test_math_int(OpCode::Div, &[2, 2, 1]));
    assert!(test_math_int(OpCode::Div, &[2, 4, 2]));
    // assert!(!test_math_int(OpCode::Div, &[0, 0, 4]));
}
