use crate::melvm::Covenant;
use crate::melvm::{Executor, Value};
use crate::melvm::opcode::OpCode;
use ethnum::U256;
use std::collections::HashMap;
use im;



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

fn pop_assert_none(ex: &mut Executor){
    match ex.stack.pop(){
        Some(_) => assert!(false),
        None => assert!(true)
    }
}

macro_rules! melvm_exec {
    ({[$($stack_item: tt),*][$($heap_item: tt),*]; $($opcodes: expr);+;}) => {
        {
            
            let heap: &[Value]  = &[$(melvm_exec!($heap_item)),*];
            let opcodes: &[OpCode] = &[$($opcodes),*];
            let mut exec: Executor = exec_with_heap(&opcodes, &heap);
            $(melvm_exec!(@push exec, $stack_item);)*
            run_ops(&mut exec, &opcodes);
            exec

        }
    };
    (@push $exec: ident, [$($token: expr),*]) => {
         $exec.stack.push(Value::Vector(im::vector![$(melvm_exec!($token)),*]))
    }; 
    
    (@push $exec: ident, $token: literal) => {
        $exec.stack.push(Value::Int(U256::from($token as u128)))
    };
    
    // ($program: block, $pc: tt) => {
    //     melvm_exec!($pc)
    // };
    ($item: literal) => {
        Value::Int(U256::from($item as u128))
    };
}


macro_rules! write_tests {
    ($function_name: ident, $opcode: path, $($statements: tt);*;) => {
        #[test]
        fn $function_name() {
            $(write_tests!(@enter $opcode, $statements);)*
        }
    };
    (@enter $opcode: path, [$($values: literal),* $(== $match: tt)?]) => {
        {
            let val = melvm_exec!({
                [$($values),*][];
                $opcode;
            }).stack.pop();
            assert!(write_tests!(@mat val $($match)?));
        }
    };
    (@enter $opcode: path, [$($values: literal),* $(/= $match: tt)?]) => {
        {
            let val = melvm_exec!({
                [$($values),*][];
                $opcode;
            }).stack.pop();
            assert!(!write_tests!(@mat val $($match)?));
        }
    };
    // basic case: [1,2,3,... == ()]
    // push list onto stack, call opcode, return true if Some
    (@mat $val: ident ()) => {
        match $val{
            Some(_) => true,
            None => false,
        }
    };

    //general case: [1,2,3,... == (p => func(p))]
    //push values before `==` onto stack
    //call opcode and return Some(p) => func(p) or None => false
    (@mat $val: ident ($left: ident => $right: expr)) => {
        match $val{
            Some($left) => $right,
            None => false,
        }
    };

    //specific case 1: [1,2,... == 1]
    // cast literal following `==`
    (@mat $val: ident $right: literal) => {
        write_tests!(@mat $val (p => p == ($right as u128).into()))
    };
    //specific case 2: [1,2,... == (expr)]
    //directly compare expression `expr` to `p` in Some(p) => p == expr
    //be sure to wrap your expression in (), {}, or [], to catch all possible expressions
    (@mat $val: ident $right: expr) => {
        write_tests!(@mat $val (p => p == $right))
    };

}

// fn thingy() -> Value{
//     melvm_exec!({
//         [1][{
//             [2,2][];
//             OpCode::Add;
//         }];
//         OpCode::LoadImm(0u16);
//         OpCode::Add;
//     })
// }

// #[test]
// fn test_add(){
//     assert!(thingy() == Value::Int(5u128.into()))
// }



write_tests!(test_add, OpCode::Add, 
    [1,2 == 3];
    [3,2 /= 2];
);

write_tests!(test_sub, OpCode::Sub,
    [1,2 == 1];
    [1,0 == (Value::Int(U256::MAX))];
);

write_tests!(test_mul, OpCode::Mul,
    [1, 2 /= 4];
    [4, 2 == 8];
    [1, 1 == 1];
);
write_tests!(test_div, OpCode::Div,
    [2,2 == 1];
    [2,4 == 2];
    [2,4 /= 3];
    [0,0 /= ()];
);

write_tests!(test_rem, OpCode::Rem, 
    [1,1 == 0];
    [2,4 == 0];
    [2,1 == 1];
    [0,0 /= ()]; // this one
);

// // Logic tests

write_tests!(test_and, OpCode::And,
    [1018,5 == 0];
    [2,2 == 2];
);

write_tests!(test_or, OpCode::Or,
    [1,2 == 3];
    [1018,5 == 1023];
);

write_tests!(test_xor, OpCode::Xor, 
    [2,2 == 0];
    [2,2 /= 1];
);

write_tests!(test_eql, OpCode::Eql, 
    [2,2 == 1];
    [2,2 /= 0];
);

write_tests!(test_not, OpCode::Not,
    [0 == (Value::Int(U256::MAX))];
    [1 == (Value::Int(U256::MAX - 1))];
);

// comparators

write_tests!(test_lt, OpCode::Lt, 
    [1,0 ==1];
    [0,1 == 0];
    [1,125 /= 1];
    [654654,2121 /= 0];
);

write_tests!(test_gt,OpCode::Gt,
    [1,0 == 0];
    [0,1 == 1];
    [1,125 /= 0];
    [654654,2121 /= 1];
);

// bitshifts

// doesn't wrap right overflows to the left
write_tests!(test_shr, OpCode::Shr,
    [2,4 == 1];
    [7,1 == 0];
);

write_tests!(test_shl, OpCode::Shl,
    [2,4 == 16];
    [7,4 == 512];
    [257,1 == 2];
);

// // cryptography

// storage access

#[test]
fn test_store_load(){
    let mut exec =  melvm_exec!({
        [0][1];
        OpCode::Load;
    });
    pop_assert(&mut exec, Value::Int(1u128.into()));
    let mut exec = melvm_exec!({
        [0,1,0][];
        OpCode::Store;
        OpCode::Load;
    });
    pop_assert(&mut exec, Value::Int(1u128.into()));
}

fn test_store() {
}
#[test]
fn test_store_load_imm(){
    // heap address 1,3,  onto the stack
    let mut exec = melvm_exec!({
        [100][99,3];
        OpCode::LoadImm(1);
    });
    // pop a value from the Executor stack 
    // assert_eq to second param
    pop_assert(&mut exec, Value::Int(3u128.into()));

    // store 100 to 0th hash address
    // this overwrites 0
    let mut exec = melvm_exec!({
       [100][99,3];
       OpCode::StoreImm(0);
    });
    pop_assert_none(&mut exec);

    //values aren't deleted from heap when loaded
    //the final stack is: [100, 3, 3,3,3] and the heap is [3,3]
    let mut exec = melvm_exec!({
       [100][99,3];
       OpCode::LoadImm(1);
       OpCode::StoreImm(0); // overwrite 99 
       OpCode::LoadImm(0);
       OpCode::LoadImm(0);
       OpCode::LoadImm(0);
       OpCode::LoadImm(1);
    });
    pop_assert(&mut exec, Value::Int(3u128.into()));
}

// vector ops
#[test]
fn test_vref() {
    let mut exec = melvm_exec!({
        [0, [1,2,3]][];
        OpCode::VRef;
    });
    assert_eq!(exec, Value::Int(1u128.into()));
}

#[test]
fn test_vref_failure(){
    let mut exec = melvm_exec!({
        [1, [1,2,3]][];
        OpCode::VRef;
    });
    pop_assert(&mut exec, Value::Int(3u128.into()));
}











// best for last 
#[test]
fn test_noop() {
    let cov = Covenant::from_ops(&[OpCode::Noop]).expect("Noop did something!!");
    assert_eq!(cov.check_raw(&[]), false)
}
