use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <infile> <outfile>", args[0]);
        std::process::exit(1);
    }
    let infile_name = &args[1];
    let outfile_name = &args[2];
    let infile = File::open(infile_name).unwrap_or_else(|_| {
        eprintln!("Oops! Could not read {}", infile_name);
        std::process::exit(1);
    });
    let reader = BufReader::new(infile);
    let mut outfile = File::create(outfile_name).unwrap_or_else(|_| {
        eprintln!("Oops! Could not open {} for writing", outfile_name);
        std::process::exit(1);
    });
    for line in reader.lines() {
        match line {
            Ok(content) => {
                writeln!(outfile, "{}", content).unwrap();
            }
            Err(e) => {
                eprintln!("Error reading line: {}", e);
            }
        }
    }
    println!("Copied contents from {} to {}", infile_name, outfile_name);
}
