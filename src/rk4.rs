use std::io;
use meval::{Expr, Context};

fn main() {
    println!("Enter the differential equation f(x, y):");
    let mut func_str = String::new();
    io::stdin().read_line(&mut func_str).unwrap();
    let func_str = func_str.trim();

    let expr: Expr = match func_str.parse() {
        Ok(e) => e,
        Err(_) => {
            println!("Invalid expression.");
            return;
        }
    };

    let f = |x: f64, y: f64| -> f64 {
        let mut ctx = Context::new();
        ctx.var("x", x);
        ctx.var("y", y);
        expr.eval_with_context(ctx).unwrap()
    };

    let x0 = read_f64("Enter initial x (x0): ");
    let y0 = read_f64("Enter initial y (y0): ");
    let h = read_f64("Enter step size (h): ");

    let y1 = rk4(f, x0, y0, h);
    
    println!("Estimated y1 = {}", y1);
}

fn rk4<F>(f: F, x0: f64, y0: f64, h: f64) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let k1 = h * f(x0, y0);
    let k2 = h * f(x0 + h / 2.0, y0 + k1 / 2.0);
    let k3 = h * f(x0 + h / 2.0, y0 + k2 / 2.0);
    let k4 = h * f(x0 + h, y0 + k3);

    println!("k1 = {}", k1);
    println!("k2 = {}", k2);
    println!("k3 = {}", k3);
    println!("k4 = {}", k4);

    y0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
}

fn read_f64(prompt: &str) -> f64 {
    println!("{}", prompt);
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().expect("Invalid float input")
}