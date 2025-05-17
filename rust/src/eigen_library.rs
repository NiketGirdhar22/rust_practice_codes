use ndarray::{Array2};
use ndarray_linalg::Eig;
use num_complex::Complex;
use std::io;

fn print_real_if_possible(c: &Complex<f64>) -> f64 {
    if c.im.abs() < 1e-10 {
        c.re
    } else {
        panic!("Complex number with non-zero imaginary part: {:?}", c);
    }
}

fn main() {
    println!("Enter the size of the square matrix (n):");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let n: usize = input.trim().parse().unwrap();

    let mut data = Vec::new();
    println!("Enter the matrix elements row by row, space-separated:");

    for i in 0..n {
        input.clear();
        println!("Row {}:", i + 1);
        std::io::stdin().read_line(&mut input).unwrap();
        let row: Vec<f64> = input
            .trim()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        if row.len() != n {
            panic!("Each row must have exactly {} elements", n);
        }
        data.extend(row);
    }

    let matrix = Array2::from_shape_vec((n, n), data).unwrap();

    println!("\nInput matrix:\n{:?}", matrix);

    match matrix.eig() {
        Ok((eigenvalues, eigenvectors)) => {
            println!("\nEigenvalues:");
            for val in eigenvalues.iter() {
                println!("{}", print_real_if_possible(val));
            }

            println!("\nEigenvectors (each column is an eigenvector):");
            for row in eigenvectors.rows() {
                for val in row.iter() {
                    print!("{:>12.6} ", print_real_if_possible(val));
                }
                println!();
            }
        }
        Err(e) => {
            eprintln!("Error computing eigenvalues/vectors: {}", e);
        }
    }
}