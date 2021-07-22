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
    ($function_name: ident, $opcode: path, $($statements: tt);*;) => {
        #[test]
        fn $function_name() {
            $(write_tests!(@enter $opcode, $statements);)*
        }
    };
    (@enter $opcode: path, [$($values: literal),* $(== $match: tt)?]) => {
        {
            let val = do_op_with_args($opcode, &[$(U256::from($values as u128)),*]);
            assert!(write_tests!(@mat val $($match)?));
        }
    };
    (@enter $opcode: path, [$($values: literal),* $(/= $match: tt)?]) => {
        {
            let val = do_op_with_args($opcode, &[$(U256::from($values as u128)),*]);
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


// macro_rules! a_thing {
//     ($($i: tt),*) => {
//         #[test]
//         fn a() {
//             $(assert!(!$i);)*
//         }
        
//     }
// }   

// a_thing!(!false,!true);

#[test]
fn test_noop() {
    let cov = Covenant::from_ops(&[OpCode::Noop]).expect("Noop did something!!");
    assert_eq!(cov.check_raw(&[]), false)
}

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

// Logic tests

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