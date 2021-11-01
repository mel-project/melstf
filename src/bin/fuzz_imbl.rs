use arbitrary::{Arbitrary, Unstructured};

#[cfg(fuzzing)]
use honggfuzz::fuzz;

#[cfg(fuzzing)]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(fuzzing)]
fn main() {
    loop {
        fuzz!(|data: &[u8]| { test_once(&data) });
    }
}

#[derive(Debug, Arbitrary, Clone)]
enum Op {
    Literal(Vec<u8>),
    Append,
}

fn eval(ops: &[Op]) -> Option<imbl::Vector<u8>> {
    let mut stack: Vec<imbl::Vector<u8>> = Vec::new();
    for op in ops {
        match op {
            Op::Literal(v) => stack.push(v.into()),
            Op::Append => {
                let mut x = stack.pop()?;
                let y = stack.pop()?;
                x.append(y);
                stack.push(x);
            }
        }
    }
    stack.pop()
}

fn test_once(data: &[u8]) {
    let data = Vec::<Op>::arbitrary(&mut Unstructured::new(data));
    if let Ok(data) = data {
        eval(&data);
    }
}

#[cfg(not(fuzzing))]
fn main() {}
