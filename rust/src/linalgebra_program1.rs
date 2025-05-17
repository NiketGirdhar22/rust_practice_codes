use std::io;

fn lu_decompose(a: &mut Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n]; // Lower triangular matrix
    let mut u = a.clone(); // Upper triangular matrix

    for i in 0..n {
        l[i][i] = 1.0; // Set the diagonal of L to 1
    }

    // Perform LU Decomposition without pivoting
    for i in 0..n {
        for j in i..n {
            // Calculate the upper triangular matrix U
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[i][k] * u[k][j];
            }
            u[i][j] -= sum;
        }

        for j in i + 1..n {
            // Calculate the lower triangular matrix L
            let mut sum = 0.0;
            for k in 0..i {
                sum += l[j][k] * u[k][i];
            }
            l[j][i] = (u[j][i] - sum) / u[i][i];
        }
    }

    (l, u)
}

fn lu_back_substitution(l: &Vec<Vec<f64>>, u: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Forward substitution (Ly = b)
    for i in 0..n {
        y[i] = b[i];
        for j in 0..i {
            y[i] -= l[i][j] * y[j];
        }
    }

    // Backward substitution (Ux = y)
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in i + 1..n {
            x[i] -= u[i][j] * x[j];
        }
        x[i] /= u[i][i];
    }

    x
}

fn round_small_values(matrix: &mut Vec<Vec<f64>>, tolerance: f64) {
    for row in matrix.iter_mut() {
        for value in row.iter_mut() {
            if value.abs() < tolerance {
                *value = 0.0;
            }
        }
    }
}

fn main() {
    let mut input = String::new();
    println!("Enter the size of the matrix (n x n):");
    io::stdin().read_line(&mut input).unwrap();
    let size: usize = input.trim().parse().unwrap();

    let mut matrix: Vec<Vec<f64>> = Vec::new();
    println!("Enter the elements of the matrix (row by row):");

    for _ in 0..size {
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let row: Vec<f64> = input
            .trim()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        matrix.push(row);
    }

    let mut matrix_copy = matrix.clone();

    // LU Decomposition
    let (l, u) = lu_decompose(&mut matrix_copy);

    // Display LU Decomposed matrices
    println!("LU Decomposed Matrix:");
    println!("L: {:?}", l);
    println!("U: {:?}", u);

    // Inverse calculation by solving Ax = I for each column
    let mut inverse = vec![vec![0.0; size]; size];
    for i in 0..size {
        let mut b = vec![0.0; size];
        b[i] = 1.0;
        let column = lu_back_substitution(&l, &u, &b);
        for j in 0..size {
            inverse[j][i] = column[j];
        }
    }

    // Round small values in the inverse matrix using a tolerance of 1e-6
    round_small_values(&mut inverse, 1e-6);

    // Display inverse matrix
    println!("Inverse Matrix:");
    for row in &inverse {
        println!("{:?}", row);
    }

    // Verify if A * Ainv = Identity Matrix
    let mut result = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                result[i][j] += matrix[i][k] * inverse[k][j];
            }
        }
    }

    // Round small values in the result matrix
    round_small_values(&mut result, 1e-6);

    // Display Matrix * Inverse result
    println!("Matrix * Inverse = Identity Matrix:");
    for row in &result {
        println!("{:?}", row);
    }
}