use crate::melvm::Covenant;
use crate::melvm::{Executor, Value};
use crate::melvm::opcode::OpCode;
use ethnum::U256;
use std::collections::HashMap;



pub fn exec_with_heap(ops: &[OpCode], heap: &[Value]) -> Executor {
    let mut hm = HashMap::new();
    for (i, v) in heap.iter().enumerate() {
        hm.insert(i as u16, v.clone());
    }
    Executor::new(ops.into(),hm)
    
}

fn run_ops(ex: &mut Executor, ops: &[OpCode]){
    println!("Trying {:?}", ops);
    while ex.pc() < ops.len() {
        if ex.step().is_none() {
            break
        }
    }
    println!("result: {:?}\n---", ex.stack);
}



macro_rules! write_tests {
    ({[$($stack_item: tt),*][$($heap_item: tt),*]; $($opcodes: expr);+;}) => {
        {
            
            let heap: &[Value]  = &[$(write_tests!($heap_item)),*];
            let opcodes: &[OpCode] = &[$(write_tests!(@push $stack_item)),*,$($opcodes),*];
            let mut exec: Executor = exec_with_heap(&opcodes, &heap);
            run_ops(&mut exec, &opcodes);
            exec.stack.pop().unwrap()
        }
    };
    (@push $token: expr) => {
        OpCode::PushI(U256::from($token as u128))
    };
    // ($program: block, $pc: tt) => {
    //     write_tests!($pc)
    // };
    ($item: literal) => {
        Value::Int(U256::from($item as u128))
    };
}


fn thingy() -> Value{
    write_tests!({
        [1][{
            [2,2][];
            OpCode::Add;
        }];
        OpCode::LoadImm(0u16);
        OpCode::Add;
    })
}

#[test]
fn test_add(){
    assert!(thingy() == Value::Int(5u128.into()))
}



// write_tests!(test_add, OpCode::Add, 
//     [1,2 == 3];
//     [3,2 /= 2];
// );

// write_tests!(test_sub, OpCode::Sub,
//     [1,2 == 1];
//     [1,0 == (Value::Int(U256::MAX))];
// );

// write_tests!(test_mul, OpCode::Mul,
//     [1, 2 /= 4];
//     [4, 2 == 8];
//     [1, 1 == 1];
// );
// write_tests!(test_div, OpCode::Div,
//     [2,2 == 1];
//     [2,4 == 2];
//     [2,4 /= 3];
//     [0,0 /= ()];
// );

// write_tests!(test_rem, OpCode::Rem, 
//     [1,1 == 0];
//     [2,4 == 0];
//     [2,1 == 1];
//     [0,0 /= ()]; // this one
// );

// // Logic tests

// write_tests!(test_and, OpCode::And,
//     [1018,5 == 0];
//     [2,2 == 2];
// );

// write_tests!(test_or, OpCode::Or,
//     [1,2 == 3];
//     [1018,5 == 1023];
// );

// write_tests!(test_xor, OpCode::Xor, 
//     [2,2 == 0];
//     [2,2 /= 1];
// );

// write_tests!(test_eql, OpCode::Eql, 
//     [2,2 == 1];
//     [2,2 /= 0];
// );

// write_tests!(test_not, OpCode::Not,
//     [0 == (Value::Int(U256::MAX))];
//     [1 == (Value::Int(U256::MAX - 1))];
// );

// // comparators

// write_tests!(test_lt, OpCode::Lt, 
//     [1,0 ==1];
//     [0,1 == 0];
//     [1,125 /= 1];
//     [654654,2121 /= 0];
// );

// write_tests!(test_gt,OpCode::Gt,
//     [1,0 == 0];
//     [0,1 == 1];
//     [1,125 /= 0];
//     [654654,2121 /= 1];
// );

// // bitshifts

// // doesn't wrap right overflows to the left
// write_tests!(test_shr, OpCode::Shr,
//     [2,4 == 1];
//     [7,1 == 0];
// );

// write_tests!(test_shl, OpCode::Shl,
//     [2,4 == 16];
//     [7,4 == 512];
//     [257,1 == 2];
// );

// // cryptography

// // storage access

// write_tests!(test_store, OpCode::Store,
//     [2 == 2];
// );









// best for last 
#[test]
fn test_noop() {
    let cov = Covenant::from_ops(&[OpCode::Noop]).expect("Noop did something!!");
    assert_eq!(cov.check_raw(&[]), false)
}