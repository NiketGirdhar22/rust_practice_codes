use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

fn simple_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 16) & 0x7FFF) as f64 / 32768.0
}

fn montecarlo(num_particles: usize, max_time: usize, decay_prob: f64, seed: &mut u64) -> Vec<usize> {
    let mut result = vec![0; max_time + 1];
    let mut remaining = num_particles;
    result[0] = remaining;
    for t in 1..max_time {
        let mut decayed = 0;
        for _ in 0..remaining {
            if simple_rand(seed) <= decay_prob {
                decayed += 1;
            }
        }
        remaining = remaining.saturating_sub(decayed);
        result[t] = remaining;
        if remaining == 0 {
            break;
        }
    }
    result
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "Usage: {} outfilename num_particles_init max_time num_cycles decay_prob",
            args[0]
        );
        process::exit(1);
    }
    let filename = &args[1];
    let num_particles: usize = args[2].parse().expect("Invalid num_particles");
    let max_time: usize = args[3].parse().expect("Invalid max_time");
    let num_cycles: usize = args[4].parse().expect("Invalid num_cycles");
    let decay_prob: f64 = args[5].parse().expect("Invalid decay_prob");
    let file = File::create(filename).expect("Could not create file");
    let mut writer = BufWriter::new(file);
    let mut total = vec![0usize; max_time + 1];
    let base_seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    for cycle in 0..num_cycles {
        let mut seed = base_seed + cycle as u64;
        let result = montecarlo(num_particles, max_time, decay_prob, &mut seed);
        for (i, val) in result.iter().enumerate() {
            total[i] += val;
        }
    }
    for val in total {
        writeln!(writer, "{:E}", val as f64 / num_cycles as f64).unwrap();
    }
    println!("Simulation complete. Results written to {}", filename);
}