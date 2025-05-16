use std::env;

fn integrand(x: &[f64; 6]) -> f64 {
    let a = 1.0;
    let b = 0.5;
    let x2 = x.iter().map(|&xi| xi * xi).sum::<f64>();
    let xy = (x[0] - x[3]).powi(2) + (x[1] - x[4]).powi(2) + (x[2] - x[5]).powi(2);
    (-a * x2 - b * xy).exp()
}

fn simple_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 16) & 0x7FFF) as f64 / 32768.0
}

fn montecarlo_integration(samples: u64, l: f64, jacobi: f64) -> (f64, f64) {
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut seed = 12345;
    for _ in 0..samples {
        let x: [f64; 6] = [
            -l + 2.0 * l * simple_rand(&mut seed),
            -l + 2.0 * l * simple_rand(&mut seed),
            -l + 2.0 * l * simple_rand(&mut seed),
            -l + 2.0 * l * simple_rand(&mut seed),
            -l + 2.0 * l * simple_rand(&mut seed),
            -l + 2.0 * l * simple_rand(&mut seed),
        ];
        let fx = integrand(&x);
        sum += fx;
        sum2 += fx * fx;
    }
    sum /= samples as f64;
    sum2 /= samples as f64;
    let integral = jacobi * sum;
    let sigma = jacobi * ((sum2 - sum * sum) / samples as f64).sqrt();
    (integral, sigma)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} number_of_samples", args[0]);
        std::process::exit(1);
    }
    let n: u64 = args[1].parse().expect("Invalid number of samples");
    let l = 5.0;
    let jacobi = (2.0 * l as f64).powi(6);
    println!("Running with N = {}...", n);
    let (integral, sigma) = montecarlo_integration(n, l, jacobi);
    println!("Monte Carlo result = {:10.8E}", integral);
    println!("Sigma             = {:10.8E}", sigma);
    Ok(())
}