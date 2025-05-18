# RUST / Python code implementation

1. 
<details>
<summary>Classes Program 1 (2nd derivative of e^x)</summary>

1. Input parameters : h(initial step size), x(point of eval), n (number of steps, halves of h)
2. 2 lists created : for step sizes and for corresponding values of second derivatives
3. For n iterations:
    - Compute derivative by cntral difference formula
    - Append  h and approximation to the corresponding lists
    - Half h for next iteration
4. Calculate error by computing true value and for each approximation compute relative error
5. Write step size and relative error to file

-   <details>
    <summary>RUST implementation</summary>
    <pre><code class="language-rust">
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
    </code></pre>

    ### Output

    ![Code run](outputs/rust/classes_program1_1.png)
    ![Output file](outputs/rust/classes_program1_2.png)

    </details>

-   <details>
    <summary>Python implementation</summary>
    <pre><code class="language-python">
    def factorial(n):
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def exp_approx(x, terms=20):
        total = 0.0
        for k in range(terms):
            total += (x ** k) / factorial(k)
        return total

    def compute_second_derivative(n, x, h):
        h_steps = []
        computed = []
        for _ in range(n):
            h_steps.append(h)
            approx = (exp_approx(x + h) - 2 * exp_approx(x) + exp_approx(x - h)) / (h * h)
            computed.append(approx)
            h /= 2
        return h_steps, computed

    def write_output(h_steps, computed, x):
        with open("out.dat", "w") as f:
            true_val = exp_approx(x)
            for h, approx in zip(h_steps, computed):
                rel_error = abs(approx - true_val) / true_val
                f.write(f"{h:.6e} {rel_error:.5e}\n")

    initial_step = float(input("Initial stepsize: "))
    x = float(input("Evaluate at point x: "))
    number_of_steps = int(input("Number of steps: "))

    h_steps, computed = compute_second_derivative(number_of_steps, x, initial_step)
    write_output(h_steps, computed, x)
    print("Results written to out.dat")
    </code></pre>

    ### Output

    ![Code run](outputs/python/classes_program1_1.png)
    ![Output file](outputs/python/classes_program1_2.png)
    
    </details>


</details>