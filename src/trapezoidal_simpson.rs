use std::io;
use meval::{Expr, Context};

type Function = Box<dyn Fn(f64) -> f64>;

struct Integrator {
    a: f64,
    b: f64,
    n: usize,
    f: Function,
}

impl Integrator {
    fn trapezoidal(&self) -> f64 {
        let h = (self.b - self.a) / self.n as f64;
        let mut sum = (self.f)(self.a) + (self.f)(self.b);

        for i in 1..self.n {
            let x = self.a + i as f64 * h;
            sum += 2.0 * (self.f)(x);
        }

        (h / 2.0) * sum
    }
    fn simpson(&self) -> Option<f64> {
        if self.n % 2 != 0 {
            return None;
        }

        let h = (self.b - self.a) / self.n as f64;
        let mut sum = (self.f)(self.a) + (self.f)(self.b);

        for i in 1..self.n {
            let x = self.a + i as f64 * h;
            sum += if i % 2 == 0 {
                2.0 * (self.f)(x)
            } else {
                4.0 * (self.f)(x)
            };
        }

        Some((h / 3.0) * sum)
    }
}

fn main() {
    println!("Enter the function f(x): ");
    let mut func_str = String::new();
    io::stdin().read_line(&mut func_str).unwrap();
    let func_str = func_str.trim();=

    let expr: Expr = match func_str.parse() {
        Ok(e) => e,
        Err(_) => {
            println!("Invalid expression.");
            return;
        }
    };

    let f: Function = Box::new(move |x: f64| {
        let mut ctx = Context::new();
        ctx.var("x", x);
        expr.eval_with_context(ctx.clone()).unwrap()
    });

    let a = read_f64("Enter lower limit a:");
    let b = read_f64("Enter upper limit b:");
    let n = read_usize("Enter number of intervals n:");

    let integrator = Integrator { a, b, n, f };

    println!("\nResult using Trapezoidal Rule: {}", integrator.trapezoidal());

    match integrator.simpson() {
        Some(simpson_result) => {
            println!("Result using Simpson's Rule: {}", simpson_result);
        }
        None => {
            println!("Simpson's Rule requires even n. Skipping Simpson’s Rule.");
        }
    }
}


fn read_f64(prompt: &str) -> f64 {
    println!("{}", prompt);
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().expect("Invalid number")
}

fn read_usize(prompt: &str) -> usize {
    println!("{}", prompt);
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().expect("Invalid number")
}