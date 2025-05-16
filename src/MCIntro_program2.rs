use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

fn simple_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 16) & 0x7FFF) as f64 / 32768.0
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} filename N", args[0]);
        process::exit(1);
    }
    let filename = &args[1];
    let n: u64 = args[2].parse().expect("Invalid number");
    let file = File::create(filename).expect("Could not create file");
    let mut writer = BufWriter::new(file);
    let mut seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let mut nl = n;
    for t in 0..10 * n {
        let r = (simple_rand(&mut seed) * n as f64).floor() as u64;
        if r < nl {
            nl -= 1;
        } else {
            nl += 1;
        }
        writeln!(writer, "{} {}", t, nl).expect("Write failed");
    }
    println!("Simulation complete. Data written to {}", filename);
}