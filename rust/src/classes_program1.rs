use std::fs::File;
use std::io::{self, Write, BufRead};
use std::f64::consts::E;

use std::f64;

fn main() {
    let stdin = io::stdin();

    println!("Initial stepsize:");
    let initial_step = read_input(&stdin).trim().parse::<f64>().unwrap();

    println!("Evaluate at point x:");
    let x = read_input(&stdin).trim().parse::<f64>().unwrap();

    println!("Number of steps (stepsize will be halved each iteration):");
    let number_of_steps = read_input(&stdin).trim().parse::<usize>().unwrap();

    let (h_steps, computed_derivatives) = compute_second_derivative(number_of_steps, x, initial_step);

    write_output(&h_steps, &computed_derivatives, x).expect("Failed to write to file");
}

fn read_input(stdin: &io::Stdin) -> String {
    let mut line = String::new();
    stdin.lock().read_line(&mut line).unwrap();
    line
}

fn compute_second_derivative(n: usize, x: f64, mut h: f64) -> (Vec<f64>, Vec<f64>) {
    let mut h_steps = Vec::with_capacity(n);
    let mut computed = Vec::with_capacity(n);
    for _ in 0..n {
        h_steps.push(h);
        let deriv = (f64::exp(x + h) - 2.0 * f64::exp(x) + f64::exp(x - h)) / (h * h);
        computed.push(deriv);
        h /= 2.0;
    }
    (h_steps, computed)
}

fn write_output(h_steps: &Vec<f64>, computed: &Vec<f64>, x: f64) -> io::Result<()> {
    let mut file = File::create("out.dat")?;
    for (h, approx) in h_steps.iter().zip(computed.iter()) {
        let rel_error = (approx - f64::exp(x)).abs() / f64::exp(x);
        writeln!(file, "{:.6} {:12.5e}", h.log10(), rel_error.log10())?;
    }
    Ok(())
}