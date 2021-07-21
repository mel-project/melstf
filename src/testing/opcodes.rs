use crate::melvm::Covenant;
use crate::melvm::{Executor, Value};
use crate::melvm::opcode::OpCode;
use ethnum::U256;
use std::collections::HashMap;



pub fn exec_from_args(ops: &[OpCode], args: &[Value]) -> Executor {
    let mut hm = HashMap::new();
    for (i, v) in args.iter().enumerate() {
        hm.insert(i as u16, v.clone());
    }
    Executor::new(ops.into(),hm)
    
}

fn run_ops(ops: &[OpCode]) -> Option<Value> {
    let mut ex = exec_from_args(ops, &[]);
    println!("Trying {:?}", ops);
    while ex.pc() < ops.len() {
        if ex.step().is_none() {
            return None;
        }
    }

    println!("result: {:?}\n---", ex.stack);
    ex.stack.pop()
}

fn do_op_with_args(op: OpCode, args: &[U256]) -> Option<Value> {
    let mut i_args: Vec<OpCode> = args.into_iter().map(|x| OpCode::PushI(x.clone())).collect();
    i_args.push(op);
    run_ops(&i_args)
}

fn do_op_with_args_int(op: OpCode, args: &[u128]) -> Option<Value> {
    let i_args: Vec<U256> = args.into_iter().map(|x| U256::from(x.clone())).collect();
    do_op_with_args(op, &i_args[..])
}

fn test_ops_int(op: OpCode, args: &[u128]) -> bool {
    let val = do_op_with_args_int(op, &args[..args.len()-1]);
    match val {
      Some(p) => p == args[args.len()-1].into(),
      None => false,
    } 
}

macro_rules! write_tests {
    ($function_name: ident, $opcode: path, $($statements: tt $(=> $comparator: expr)?),*) => {
        #[test]
        fn $function_name() {
            
            $(assert!(write_tests!(@gen $opcode, $statements) $(== $comparator)?);)*
        }
    };
    (@gen $opcode: path, [$i1: literal, $i2: literal, $i3: literal]) => {
        {
            let val = do_op_with_args($opcode, &[U256::from($i1 as u128),U256::from($i2 as u128)]);
            match val{
                Some(p) => p == Value::from($i3 as u128),
                None => false,
            }
        }
    };
    (@gen $opcode: path, [$i1: literal, $i2: literal, $i3: expr]) => {
        {
            let val = do_op_with_args($opcode, &[U256::from($i1 as u128),U256::from($i2 as u128)]);
            match val{
                Some(p) => p == $i3,
                None => false,
            }
        }
    }

}


// macro_rules! test_thing {
//     ($i: expr) => {
//         #[test]
//         fn test() {
//             println!("{:?}", $i);
//             panic!("")
//         }
        
//     }
// }   

// test_thing!(Value::Int(U256::MAX));

#[test]
fn test_noop() {
    let cov = Covenant::from_ops(&[OpCode::Noop]).expect("Noop did something!!");
    assert_eq!(cov.check_raw(&[]), false)
}

write_tests!(test_add, OpCode::Add, 
    [1,2,3] => true,
    [3,2,2] => false
);

write_tests!(test_sub, OpCode::Sub,
    [1,0, Value::Int(U256::MAX)]
);


#[test]
fn test_mul() {
    assert!(test_ops_int(OpCode::Mul, &[1, 1, 1]));
    assert!(test_ops_int(OpCode::Mul, &[4, 2, 8]));
    assert!(!test_ops_int(OpCode::Mul, &[1, 2, 4]));
}   

#[test]
fn test_div() {
    assert!(test_ops_int(OpCode::Div, &[2, 2, 1]));
    assert!(test_ops_int(OpCode::Div, &[2, 4, 2]));

    // division by 0 should fail; if Some was returned something is wrong!
    assert!({
        if let Some(_) = do_op_with_args_int(OpCode::Div, &[0, 0]){
            false
        }
        else{true}
    });
}

#[test]
fn test_rem() {
    assert!(test_ops_int(OpCode::Rem, &[1, 1, 0]));
    assert!(test_ops_int(OpCode::Rem, &[2, 4, 0]));
    assert!(!test_ops_int(OpCode::Rem, &[2, 1, 2]));

    assert!({
        if let Some(_) = do_op_with_args_int(OpCode::Rem, &[0, 0]){
            false
        }
        else{true}
    });
}  

// Logic tests

#[test]
fn test_and(){
    assert!(test_ops_int(OpCode::And, &[1018,5, 0]));
    assert!(!test_ops_int(OpCode::And, &[2,2,2 & 1]));
}

#[test]
fn test_or(){
    assert!(test_ops_int(OpCode::Or, &[1,2, 3 | 0]));
    assert!(test_ops_int(OpCode::Or, &[2234,23642,2234 | 23642]));
    assert!(test_ops_int(OpCode::Or, &[1018,5, 1023]));
}

#[test]
fn test_xor(){
    assert!(test_ops_int(OpCode::Xor, &[2,2, 0]));
    assert!(!test_ops_int(OpCode::Xor, &[2,2, 1]));
}

#[test]
fn test_eql(){
    assert!(test_ops_int(OpCode::Eql, &[2,2,1]));
    assert!(!test_ops_int(OpCode::Eql, &[2,2,0]));
}

#[test]
fn test_not(){

    {
        let res = do_op_with_args_int(OpCode::Not, &[0])
        .expect("Can't caclulate the bitwise inverse of 0!");
        assert!(res == Value::Int(U256::MAX));
    }
    assert!(!test_ops_int(OpCode::Not, &[1,0]));
}

// comparators


#[test]
fn test_lt(){
    // & args[1] < args[0]
    assert!(test_ops_int(OpCode::Lt, &[1,0,1]));
    assert!(test_ops_int(OpCode::Lt, &[0,1,0]));
    assert!(!test_ops_int(OpCode::Lt, &[1,125,1]));
    assert!(!test_ops_int(OpCode::Lt, &[654654,2121,0]));
}
#[test]
fn test_gt(){
    assert!(test_ops_int(OpCode::Gt, &[1,0,0]));
    assert!(test_ops_int(OpCode::Gt, &[0,1,1]));
    assert!(!test_ops_int(OpCode::Gt, &[1,125,0]));
    assert!(!test_ops_int(OpCode::Gt, &[654654,2121,1]));
}

// bitshifts

fn test_shr(){
    {
        let x: U256 = 10u128.into();
    }
}

fn test_shl(){

}