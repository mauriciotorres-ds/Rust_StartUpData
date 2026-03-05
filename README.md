# Rust_StartUpData
Using rust and a start up data set to build a simple linear regression to determine the best factors for startups. 


A command line machine learning pipeline built in Rust that trains a multiple linear regression model on startup company data to predict profit based on spending patterns.

This was my own personal learning project focused on getting hands on with Rust as a systems language, understanding how ML pipelines are structured in a statically typed compiled environment, and interpreting real model outputs rather than just running someone else's notebook.

## What It Does

Given a dataset of 50 startup companies with four features/columns — R&D spend, Administration spend, Marketing spend, and State — the program:

- Loads and parses the CSV data
- Encodes categorical variables (State) into numeric form
- Trains a multiple linear regression model using the `linfa` crate
- Evaluates the model with R², MAE, and RMSE
- Displays the learned equation with per-feature coefficients
- Prints a full actual vs predicted table with error per row
- Predicts profit for a new, hypothetical startup

## Project Structure

```
Rust_StartUpData/
data/
- Startups.csv        # 50 startup records, pre-cleaned
src/
- main.rs             # Orchestrator -- calls modules in order, no business logic
- data.rs             # CSV loading and categorical encoding
- model.rs            # Model training, evaluation, and prediction
- eport.rs           # All terminal output and formatting
Cargo.toml
README.md
```

The project is deliberately split into four modules following a single responsibility pattern — each file owns exactly one concern.


## Crates Used

| Crate | Version | Purpose |
| `csv` | 1.3 | Reads and parses the CSV file row by row |
| `serde` | 1.0 | Deserializes each CSV row directly into a typed Rust struct |
| `ndarray` | 0.15 | N-dimensional array type — Rust's NumPy equivalent, required by linfa |
| `linfa` | 0.7 | Core ML framework — defines the standard Fit/Predict traits |
| `linfa-linear` | 0.7 | Linear regression algorithm implementation |

`linfa` is the Rust equivalent of scikit-learn. Like sklearn, it separates the core framework from individual algorithms — you pull in only what you need.

---

## Running It

```bash
cargo run
```

First run will take 30-60 seconds while Cargo downloads and compiles dependencies. Subsequent runs compile in under a second.

## Results

**Model performance on the full dataset:**

| Metric | Value |
| R-squared | 0.9507 |
| Mean Absolute Error | $6,468 |
| RMSE | $8,855 |

An R² of 0.9507 means the model explains 95% of the variance in profit across 50 companies. On profits ranging from $14k to $192k, an average error of $6,468 is a strong result for a dataset this size.

**Learned equation:**

```
Profit = 50142.51
       + (0.8058  x R&D Spend)
       + (-0.0268 x Administration)
       + (0.0272  x Marketing)
       + (-22.32  x State)
```

**Key finding:** R&D spend is the only feature that meaningfully drives profit. Its coefficient of 0.8058 means every $1 invested in R&D returns $0.81 in profit. Administration and Marketing coefficients are near-zero, meaning once you control for R&D they have almost no independent effect. State is similarly negligible.

This is a real and non-obvious result. In raw correlations, all three spending categories appear related to profit — but multiple linear regression isolates each feature's independent contribution and reveals that R&D is doing almost all the work.

**Sample prediction:**

A California startup spending $120k on R&D, $100k on Administration, and $300k on Marketing:
- Predicted profit: **$152,319**
- Likely range: **$145,851 – $158,787**

---

## What I Learned

**Rust fundamentals applied in this project:**
- Struct definition with explicit types and `#[derive]` macros
- The `Result` type and `?` operator for error propagation without panics
- Ownership and borrowing — passing `&[T]` references vs owned `Vec<T>`
- Iterators: `.map()`, `.filter()`, `.zip()`, `.fold()`, `.collect()`
- The module system: declaring modules with `mod`, exposing items with `pub`, importing with `use crate::`
- Pattern matching with `match` including guard conditions
- `HashMap` for grouping and aggregating data

**ML concepts applied:**
- Multiple linear regression with Ordinary Least Squares
- Label encoding for categorical variables
- Model evaluation with R², MAE, and RMSE — and what each one actually means
- Reading and interpreting feature coefficients
- Understanding the difference between correlation and independent contribution

**Architecture:**
- Separating concerns into modules so each file has one job
- Designing a `ModelMetrics` struct to carry all results cleanly between layers
- Writing an orchestrator (`main.rs`) that reads like a plain-English description of the pipeline

---

## Limitations and Next Steps

The model is currently evaluated on the same data it was trained on, which gives an optimistic accuracy estimate. The natural next step is an 80/20 train/test split to get a more honest performance number.

Other extensions worth exploring:

- Serialize the trained model to disk so it doesn't retrain on every run
- Add a CLI interface so users can input their own values interactively
- Experiment with feature normalization to see if it improves low-end predictions
- Row 50 (actual: $14,681, predicted: $48,234) is the model's largest error — investigating outliers like this would be a good exercise in understanding where linear models break down