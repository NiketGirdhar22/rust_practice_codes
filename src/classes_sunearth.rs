use std::fs::File;
use std::io::{self, Write};
use std::f64::consts::PI;

fn solver(m: f64, dt: f64, t0: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let num_intervals = (m / dt).round() as usize;
    println!("Time steps: {}", num_intervals);

    let mut t: Vec<f64> = Vec::with_capacity(num_intervals + 1);
    let mut x: Vec<f64> = Vec::with_capacity(num_intervals + 1);
    let mut y: Vec<f64> = Vec::with_capacity(num_intervals + 1);
    let mut vx: Vec<f64> = Vec::with_capacity(num_intervals + 1);
    let mut vy: Vec<f64> = Vec::with_capacity(num_intervals + 1);
    let mut r: Vec<f64> = Vec::with_capacity(num_intervals + 1);
    let mut v: Vec<f64> = Vec::with_capacity(num_intervals + 1);

    let pi4 = 4.0 * PI * PI;

    t.push(t0);
    x.push(1.0);
    y.push(0.0);
    vx.push(2.0 * PI);
    vy.push(0.0);
    r.push((x[0] * x[0] + y[0] * y[0]).sqrt());
    v.push((vx[0] * vx[0] + vy[0] * vy[0]).sqrt());

    for n in 0..num_intervals {
        let tn = t0 + (n as f64 + 1.0) * dt;
        t.push(tn);

        let xn = x[n] + dt * vx[n];
        let yn = y[n] + dt * vy[n];
        x.push(xn);
        y.push(yn);

        let r3 = (xn * xn + yn * yn).powf(1.5);
        let vxn = vx[n] - dt * pi4 * xn / r3;
        let vyn = vy[n] - dt * pi4 * yn / r3;
        vx.push(vxn);
        vy.push(vyn);

        v.push((vxn * vxn + vyn * vyn).sqrt());
        r.push((xn * xn + yn * yn).sqrt());
    }

    (r, v, t)
}

fn save_to_csv(filename: &str, t: &[f64], r: &[f64], v: &[f64]) -> io::Result<()> {
    let mut file = File::create(filename)?;
    writeln!(file, "time,radius,speed")?;
    for i in 0..t.len() {
        writeln!(file, "{:.5},{:.5},{:.5}", t[i], r[i], v[i])?;
    }
    Ok(())
}

fn main() {
    let m = 20.0;
    let dt = 0.01;
    let t0 = 0.0;

    let (r, v, t) = solver(m, dt, t0);

    if let Err(e) = save_to_csv("orbit.csv", &t, &r, &v) {
        eprintln!("Failed to save data: {}", e);
    } else {
        println!("Simulation complete. Output written to orbit.csv");
    }
}