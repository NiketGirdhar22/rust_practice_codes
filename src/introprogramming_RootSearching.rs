fn func(x: f64) -> f64 {
    x.powi(3) - x - 2.0
}

fn func_derivative(x: f64) -> f64 {
    3.0 * x.powi(2) - 1.0
}

fn bisection(f: fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64, nmax: usize) -> Option<f64> {
    for n in 1..=nmax {
        let c = (a + b) / 2.0;
        println!("n={}\ta={:.6}\tb={:.6}\tc={:.6}\tf(c)={:.6}", n, a, b, c, f(c));
        if f(c).abs() < tol || (b - a).abs() / 2.0 < tol {
            return Some(c);
        }

        if f(c) * f(a) > 0.0 {
            a = c;
        } else {
            b = c;
        }
    }
    None
}

fn secant(f: fn(f64) -> f64, mut x0: f64, mut x1: f64, tol: f64, nmax: usize) -> Option<f64> {
    for n in 1..=nmax {
        let f_x0 = f(x0);
        let f_x1 = f(x1);
        if (f_x1 - f_x0).abs() < tol {
            return None;
        }

        let x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0);
        println!("n={}\tx0={:.6}\tx1={:.6}\tx2={:.6}\tf(x2)={:.6}", n, x0, x1, x2, f(x2));

        if (x2 - x1).abs() < tol {
            return Some(x2);
        }

        x0 = x1;
        x1 = x2;
    }
    None
}

fn newton_raphson(f: fn(f64) -> f64, f_prime: fn(f64) -> f64, mut x0: f64, tol: f64, nmax: usize) -> Option<f64> {
    for n in 1..=nmax {
        let fx = f(x0);
        let fpx = f_prime(x0);

        if fpx.abs() < tol {
            return None;
        }

        let x1 = x0 - fx / fpx;
        println!("n={}\tx0={:.6}\tx1={:.6}\tf(x1)={:.6}", n, x0, x1, f(x1));

        if (x1 - x0).abs() < tol {
            return Some(x1);
        }

        x0 = x1;
    }
    None
}

fn main() {
    let tol = 0.001;
    let nmax = 100;

    println!("Bisection Method");
    if let Some(root) = bisection(func, 1.0, 2.0, tol, nmax) {
        println!("Root found by Bisection: {:.6}", root);
    } else {
        println!("Bisection failed to converge");
    }

    println!("Secant Method");
    if let Some(root) = secant(func, 1.0, 2.0, tol, nmax) {
        println!("Root found by Secant: {:.6}", root);
    } else {
        println!("Secant failed to converge");
    }

    println!("Newton-Raphson Method");
    if let Some(root) = newton_raphson(func, func_derivative, 1.0, tol, nmax) {
        println!("Root found by Newton-Raphson: {:.6}", root);
    } else {
        println!("Newton-Raphson failed to converge");
    }
}