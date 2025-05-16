use std::env;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

fn simple_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 16) & 0x7FFF) as f64 / 32768.0
}

fn func(x: f64) -> f64 {
    4.0 / (1.0 + x * x)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} N", args[0]);
        process::exit(1);
    }
    let n: u64 = args[1].parse().expect("Invalid number");
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let start = SystemTime::now();
    let mut seed = start.duration_since(UNIX_EPOCH).unwrap().as_secs();
    for _ in 0..n {
        let x = simple_rand(&mut seed);
        let fx = func(x);
        sum += fx;
        sum2 += fx * fx;
    }
    let mean = sum / n as f64;
    let mean2 = sum2 / n as f64;
    let variance = mean2 - mean * mean;
    println!("Integral = {}, variance = {}", mean, variance);
}