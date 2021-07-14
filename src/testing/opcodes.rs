use crate::melvm::Covenant;
use crate::melvm::OpCode;
use crate::melvm::{Executor, Value};
use ethnum::U256;
use std::collections::HashMap;
use std::fmt;
use std::error::Error;

#[derive(Debug)]
struct TestError {
    details: String
}

impl TestError {
    fn new(msg: &str) -> TestError {
        TestError{details: msg.to_string()}
    }
}

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,"{}", self.details)
    }
}

impl Error for TestError {
    fn description(&self) -> &str {
        &self.details
    }
}

pub fn exec_from_args(args: &[Value]) -> Executor {
    let mut hm = HashMap::new();
    for (i, v) in args.iter().enumerate() {
        hm.insert(i as u16, v.clone());
    }
    Executor::new(hm)
}

fn do_math_ops(ops: &[&OpCode; 3]) -> Option<Value> {
    let mut ex = exec_from_args(&[]);
    println!("Trying {:?}", ops);

    for op in ops.iter() {
        ex.do_op(&op);
    }

    println!("result: {:?}\n---", ex.stack);
    ex.stack.pop()
}

fn do_math(op: &OpCode, args: &[U256]) -> Option<Value> {
    do_math_ops(&[&OpCode::PushI(args[0]), &OpCode::PushI(args[1]), op])
}

fn do_math_int(op: &OpCode, args: &[u128]) -> Option<Value> {
    let i_args: Vec<U256> = args.into_iter().map(|x| U256::from(x.clone())).collect();
    do_math(op, &i_args[..])
}

fn test_math_int(op: &OpCode, args: &[u128]) -> Result<bool, TestError> {
    let val = do_math_int(op, &args[..2]);
    match val {
      Some(p) => Ok(p == args[2].into()),
      None => Err(TestError::new(&*format!("Opcode {:?} with args {:?} crashed melvm!!", op, args))),
    } 
}

#[test]
fn test_noop() {
    let cov = Covenant::from_ops(&[OpCode::Noop]).expect("Noop did something!!");
    assert_eq!(cov.check_raw(&[]), false)
}

#[test]
fn test_add() -> Result<(), TestError>{
    assert!(test_math_int(&OpCode::Add, &[1, 2, 3])?);
    assert!(!test_math_int(&OpCode::Add, &[1, 2, 4])?);
    Ok(())
}
#[test]
fn test_sub() -> Result<(), TestError>{
    assert!(test_math_int(&OpCode::Sub, &[1, 1, 0])?);
    assert!(test_math_int(&OpCode::Sub, &[1, 2, 1])?);
    assert!(!test_math_int(&OpCode::Sub, &[1, 2, 4])?);

    let res = do_math_int(&OpCode::Sub, &[1, 0]);

    match res {
      Some(p) => {
        assert!(p == Value::Int(U256::MAX));
        Ok(())
      },
      None => Err(TestError::new("Subtracting doesn't overflow properly!!")),
    }
}

#[test]
fn test_mul()-> Result<(), TestError> {
    assert!(test_math_int(&OpCode::Mul, &[1, 1, 1])?);
    assert!(test_math_int(&OpCode::Mul, &[4, 2, 8])?);
    assert!(!test_math_int(&OpCode::Mul, &[1, 2, 4])?);
    Ok(())
}   

#[test]
fn test_div()-> Result<(), TestError> {
    assert!(test_math_int(&OpCode::Div, &[2, 2, 1])?);
    assert!(test_math_int(&OpCode::Div, &[2, 4, 2])?);

    // division by 0 should fail; if Some was returned something is wrong!
    assert!({
        if let Some(_) = do_math_int(&OpCode::Div, &[0, 0]){
            false
        }
        else{true}
    });
    Ok(())
}

#[test]
fn test_rem()-> Result<(), TestError> {
    assert!(test_math_int(&OpCode::Rem, &[1, 1, 0])?);
    assert!(test_math_int(&OpCode::Rem, &[2, 4, 0])?);
    assert!(!test_math_int(&OpCode::Rem, &[2, 1, 2])?);

    assert!({
        if let Some(_) = do_math_int(&OpCode::Rem, &[0, 0]){
            false
        }
        else{true}
    });
    Ok(())
}  
