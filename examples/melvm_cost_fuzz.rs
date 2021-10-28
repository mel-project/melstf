use std::time::Instant;

use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use quanta::Clock;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use themelio_stf::melvm::{opcode::OpCode, Covenant};
type OF64 = OrderedFloat<f64>;

static CLOCK: Lazy<Clock> = Lazy::new(Clock::new);

/// Computes the fitness of the input
fn eval_fitness(input: &[u8]) -> OF64 {
    let val = Covenant(input.to_vec());
    if val.to_ops().is_err() {
        return 0.0.into();
    }
    let mut runtime = f64::MAX;
    for _ in 0..10 {
        let start = CLOCK.start();
        val.check_raw(&[]);
        runtime = runtime.min((CLOCK.end() - start) as f64);
    }
    if val
        .to_ops()
        .unwrap_or_default()
        .iter()
        .any(|f| matches!(f, OpCode::Noop))
    {
        return 0.0.into();
    };
    let weight = val.weight().unwrap() as f64;
    let ilen = input.len() as f64;
    if ilen == 0.0 {
        return 0.0.into();
    }
    (runtime / ilen).into()
}

/// Maybe mutate the input
fn mutate(input: &mut Vec<u8>) {
    while rand::random::<f64>() < 0.9 && !input.is_empty() {
        let i = rand::random::<usize>() % input.len();
        // either mutate or indel
        if rand::random() {
            input.insert(i, rand::random());
        } else {
            input.remove(i);
        }
    }
}

fn main() {
    let mut population: Vec<(Vec<u8>, OrderedFloat<f64>)> = (0..2048)
        .map(|_| loop {
            let r: u64 = rand::random();
            let v = r.to_le_bytes().to_vec();
            let fitness = eval_fitness(&v);
            if fitness > 0.0.into() {
                return (v, fitness);
            }
        })
        .collect();
    println!("generation,fitness");
    for generation in 0.. {
        population.sort_unstable_by_key(|f| f.1);
        if generation % 100 == 0 {
            eprintln!(
                "GENERATION {}; MAX FITNESS {} LEN {}",
                generation,
                population.last().unwrap().1,
                population.last().unwrap().0.len()
            );
            if population.last().unwrap().1 > 0.0.into() {
                let ops = Covenant(population.last().unwrap().0.clone())
                    .to_ops()
                    .unwrap();
                eprintln!("{:?}", ops);
            }
        }
        if generation % 10 == 0 {
            println!("{},{}", generation, population.last().unwrap().1);
        }
        for i in 0..population.len() / 2 {
            population[i] = population[population.len() / 2 + i].clone();
        }
        let halflen = population.len() / 2;
        population[..halflen].par_iter_mut().for_each(|elem| {
            mutate(&mut elem.0);
            // elem.1 = eval_fitness(&elem.0)
        });
        population
            .iter_mut()
            .for_each(|elem| elem.1 = eval_fitness(&elem.0));
    }
}
