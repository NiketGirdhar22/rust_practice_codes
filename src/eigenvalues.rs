// use std::io;

// fn main() {
//     println!("Enter matrix size (2 or 3):");

//     let mut input = String::new();
//     io::stdin().read_line(&mut input).unwrap();
//     let n: usize = input.trim().parse().unwrap();

//     let mut matrix = vec![vec![0.0; n]; n];
//     println!("Enter the matrix elements row by row:");

//     for i in 0..n {
//         input.clear();
//         println!("Row {}:", i + 1);
//         io::stdin().read_line(&mut input).unwrap();
//         let row: Vec<f64> = input
//             .trim()
//             .split_whitespace()
//             .map(|x| x.parse().unwrap())
//             .collect();

//         matrix[i] = row;
//     }

//     println!("\nMatrix:");
//     for row in &matrix {
//         println!("{:?}", row);
//     }

//     println!("\nEigenvalues:");

//     match n {
//         2 => {
//             let a = matrix[0][0];
//             let b = matrix[0][1];
//             let c = matrix[1][0];
//             let d = matrix[1][1];

//             let trace = a + d;
//             let det = a * d - b * c;
//             let discriminant = trace * trace - 4.0 * det;

//             if discriminant < 0.0 {
//                 println!("Complex eigenvalues (not supported).");
//             } else {
//                 let sqrt_disc = discriminant.sqrt();
//                 let lambda1 = 0.5 * (trace + sqrt_disc);
//                 let lambda2 = 0.5 * (trace - sqrt_disc);
//                 println!("lambda1 = {:.6}", lambda1);
//                 println!("lambda2 = {:.6}", lambda2);
//             }
//         }
//         3 => {
//             let a = matrix;

//             let trace = a[0][0] + a[1][1] + a[2][2];
//             let p1 = a[0][1] * a[1][0] + a[0][2] * a[2][0] + a[1][2] * a[2][1];
//             let det = determinant3x3(&a);

//             println!("Characteristic polynomial coefficients:");
//             println!("lambda3 - ({:.3})lambda2 + ({:.3})lambda - ({:.3}) = 0", trace, p1, det);
//             println!("Solving cubic equations is not implemented in this simple version.");
//         }
//         _ => unreachable!(),
//     }
// }

// fn determinant3x3(a: &Vec<Vec<f64>>) -> f64 {
//     a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
//     - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
//     + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
// }



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
    let mut a_t = transpose(&a);

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