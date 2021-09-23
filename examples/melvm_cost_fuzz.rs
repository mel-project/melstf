use std::time::Instant;

use arbitrary::{Arbitrary, Unstructured};
use ordered_float::OrderedFloat;
use themelio_stf::melvm::Covenant;
type OF64 = OrderedFloat<f64>;

/// Computes the fitness of the input
fn eval_fitness(input: &[u8]) -> OF64 {
    if let Ok(val) = Covenant::arbitrary(&mut Unstructured::new(input)) {
        if val.to_ops().is_err() {
            return 0.0.into();
        }
        if !val.check_raw(&[]) {
            return 0.0.into();
        }
        // dbg!(val.to_ops());
        let weight = val.weight().unwrap() as f64;
        let start = Instant::now();
        // if val.scripts.is_empty() {
        //     return 0.0.into();
        // }
        start.elapsed().as_secs_f64().into()
    } else {
        0.0.into()
    }
}

/// Maybe mutate the input
fn mutate(input: &mut Vec<u8>) {
    for i in 0..input.len() {
        if i >= input.len() {
            break;
        }
        if rand::random::<f64>() < 0.1 {
            // either mutate or indel
            if rand::random() {
                input[i] = rand::random();
            } else {
                input.insert(i, rand::random());
            }
        }
    }
}

fn main() {
    let mut population: Vec<(Vec<u8>, OrderedFloat<f64>)> = (0..10000)
        .map(|_| {
            let r: u128 = rand::random();
            let v = r.to_le_bytes().to_vec();
            let fitness = eval_fitness(&v);
            (v, fitness)
        })
        .collect();
    for generation in 0.. {
        population.sort_unstable_by_key(|f| f.1);
        // if generation  == 0 {
        eprintln!(
            "GENERATION {}; MAX FITNESS {} LEN {}",
            generation,
            population.last().unwrap().1,
            population.last().unwrap().0.len()
        );
        // }
        for i in 0..population.len() / 2 {
            population[i] = population[population.len() / 2 + i].clone();
            mutate(&mut population[i].0);
            population[i].1 = eval_fitness(&population[i].0)
        }
    }
}
