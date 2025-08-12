
# HPC Optimization Benchmark — Eliminating Inefficient Algorithm Implementations

This project demonstrates how to benchmark and compare two implementations of a frequency counting algorithm:  
1. **Naïve** — uses `list.count()` inside a loop (**O(n²)**).  
2. **Optimized** — uses a dictionary/hash table for single-pass counting (**O(n)**).


## 📂 Project Structure


1. benchmark.py                # Main benchmark script
2. optimization_technique.py   # The prototype for the code
3. runtime_vs_n.png            # Runtime vs input size plot (optional)
4. speedup_vs_n.png            # Speedup vs input size plot (optional)
5. README.md                   # This file
6. optimization paper.docx     # detailed report of the system


## Requirements

- Python **3.8+**
- (Optional, for plots) `matplotlib`


##  Running Inside a Virtual Environment (venv)

1. **Create and activate a virtual environment** (if you haven't already):

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
    ```

2. **Install optional dependencies** (only needed for plots):

   ```bash
   pip install matplotlib
   ```

3. **Run the benchmark**:

   ```bash
   python benchmark_freq.py
   ```

   This will:

   * Generate synthetic datasets of different sizes.
   * Benchmark both implementations with repeated runs.
   * Optionally create plots (`runtime_vs_n.png`, `speedup_vs_n.png`).

## Notes

* To deactivate the virtual environment after running:

  ```bash
  deactivate
  ```
* If you skip installing `matplotlib`, the script will still run and save the CSV but won’t generate plots.

