use std::io;

fn main() {
    println!("Enter matrix size n:");

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let n: usize = input.trim().parse().unwrap();

    let mut matrix = vec![vec![0.0; n]; n];
    println!("Enter the matrix elements row by row:");

    for i in 0..n {
        input.clear();
        println!("Row {}:", i + 1);
        io::stdin().read_line(&mut input).unwrap();
        let row: Vec<f64> = input
            .trim()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        matrix[i] = row;
    }

    println!("\nMatrix:");
    for row in &matrix {
        println!("{:?}", row);
    }

    let eigenvalues = qr_algorithm(matrix, 100);
    println!("\nApproximated Eigenvalues:");
    for (i, lambda) in eigenvalues.iter().enumerate() {
        println!("lambda{} ≈ {:.6}", i + 1, lambda);
    }
}

fn transpose(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[j][i] = a[i][j];
        }
    }
    result
}

fn mat_mul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn qr_decomposition(a: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];
    let a_t = transpose(&a);

    for i in 0..n {
        let mut v = a_t[i].clone();

        for j in 0..i {
            let dot = dot_product(&a_t[i], &q[j]);
            r[j][i] = dot;
            for k in 0..n {
                v[k] -= dot * q[j][k];
            }
        }

        let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        for k in 0..n {
            q[i][k] = v[k] / norm;
        }
        r[i][i] = norm;
    }

    (transpose(&q), r)
}

fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn qr_algorithm(mut a: Vec<Vec<f64>>, iterations: usize) -> Vec<f64> {
    for _ in 0..iterations {
        let (q, r) = qr_decomposition(a.clone());
        a = mat_mul(&r, &q);
    }
    (0..a.len()).map(|i| a[i][i]).collect()
}