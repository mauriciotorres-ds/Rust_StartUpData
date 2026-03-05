// report.rs -- responsible for ONLY one thing: printing output.
// No data loading, no model logic -- just formatting and display.
// If you later swap terminal output for HTML or JSON, you only change this file.

use crate::data::{Startup, StartupEncoded};
use crate::model::ModelMetrics;

// ── Utility helpers ────────────────────────────────────────────────────────────

pub fn separator(width: usize) {
    println!("{}", "-".repeat(width));
}

// ASCII progress bar. `#` = filled portion, `.` = empty portion.
// Example: bar(75000.0, 100000.0, 16) => "############...."
fn bar(value: f64, max: f64, width: usize) -> String {
    let filled = ((value / max) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}", "#".repeat(filled), ".".repeat(empty))
}

// ── Report sections ───────────────────────────────────────────────────────────

// Prints the header banner shown at program startup.
pub fn print_header(count: usize) {
    println!();
    println!("  =========================================");
    println!("      STARTUP PROFIT PREDICTION SYSTEM     ");
    println!("  =========================================");
    println!("  Records loaded: {}", count);
    println!();
}

// Prints high-level descriptive statistics about the raw dataset.
// This runs before the model so the user understands what they're working with.
pub fn print_overview(startups: &[Startup]) {
    let profits: Vec<f64>  = startups.iter().map(|s| s.profit).collect();
    let rds: Vec<f64>      = startups.iter().map(|s| s.rd).collect();
    let admins: Vec<f64>   = startups.iter().map(|s| s.administration).collect();
    let markets: Vec<f64>  = startups.iter().map(|s| s.marketing).collect();

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max  = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min  = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);

    separator(50);
    println!("  DATASET OVERVIEW");
    separator(50);
    println!("  {:22}  {:>12}", "Metric", "Mean ($)");
    separator(50);
    println!("  {:22}  {:>12.0}", "Avg R&D Spend",      mean(&rds));
    println!("  {:22}  {:>12.0}", "Avg Administration",  mean(&admins));
    println!("  {:22}  {:>12.0}", "Avg Marketing",       mean(&markets));
    println!("  {:22}  {:>12.0}", "Avg Profit",          mean(&profits));
    println!("  {:22}  {:>12.0}", "Max Profit",          max(&profits));
    println!("  {:22}  {:>12.0}", "Min Profit",          min(&profits));
    separator(50);
    println!();
}

// Prints the model evaluation metrics after training.
// This tells you how accurate the model is before you trust its predictions.
pub fn print_metrics(metrics: &ModelMetrics) {
    separator(50);
    println!("  MODEL PERFORMANCE");
    separator(50);
    // R-squared: 1.0 is perfect. Anything above 0.9 is very strong for real-world data.
    println!("  {:22}  {:>12.4}", "R-squared (R2)",    metrics.r_squared);
    // MAE: average dollar amount the model is off by per prediction.
    println!("  {:22}  ${:>11.0}", "Mean Abs Error",   metrics.mae);
    // RMSE: similar to MAE but larger errors are weighted more heavily.
    println!("  {:22}  ${:>11.0}", "RMSE",             metrics.rmse);
    separator(50);

    // Give the user a plain-English interpretation of R-squared.
    let quality = match metrics.r_squared {
        r if r >= 0.95 => "Excellent fit -- model explains almost all profit variance.",
        r if r >= 0.85 => "Strong fit -- model is reliable for predictions.",
        r if r >= 0.70 => "Moderate fit -- predictions are directionally useful.",
        _              => "Weak fit -- consider adding more features or data.",
    };
    println!("  {}", quality);
    println!();
}

// Prints a table comparing actual vs predicted profit for every startup.
// The error column shows how far off each prediction was in dollars.
pub fn print_predictions(data: &[StartupEncoded], metrics: &ModelMetrics) {
    separator(60);
    println!("  ACTUAL vs PREDICTED PROFIT");
    separator(60);
    println!(
        "  {:<5}  {:<12}  {:>14}  {:>14}  {:>10}",
        "Row", "State", "Actual ($)", "Predicted ($)", "Error ($)"
    );
    separator(60);

    let state_label = |code: f64| match code as i32 {
        0 => "California",
        1 => "Florida",
        2 => "New York",
        _ => "Unknown",
    };

    for (i, (startup, pred)) in data.iter().zip(metrics.predictions.iter()).enumerate() {
        let error = pred - startup.profit; // positive = over-predicted, negative = under-predicted
        println!(
            "  {:<5}  {:<12}  {:>14.0}  {:>14.0}  {:>+10.0}",
            i + 1,
            state_label(startup.state_encoded),
            startup.profit,
            pred,
            error
        );
    }
    separator(60);
    println!();
}

// Prints the result of predicting profit for a brand new, unseen startup.
// This is the "product" moment -- the model doing what it was built to do.
pub fn print_single_prediction(
    rd: f64,
    administration: f64,
    marketing: f64,
    state: &str,
    predicted_profit: f64,
    mae: f64,
) {
    println!();
    separator(50);
    println!("  PREDICTION: NEW STARTUP");
    separator(50);
    println!("  {:22}  ${:>12.0}", "R&D Spend",       rd);
    println!("  {:22}  ${:>12.0}", "Administration",   administration);
    println!("  {:22}  ${:>12.0}", "Marketing",        marketing);
    println!("  {:22}  {:>13}", "State",               state);
    separator(50);
    println!("  {:22}  ${:>12.0}", "Predicted Profit", predicted_profit);
    // Show a realistic confidence range using MAE as margin of error.
    // This is a simplified interval -- in production you'd use prediction intervals.
    println!(
        "  {:22}  ${:.0} -- ${:.0}",
        "Likely Range",
        predicted_profit - mae,
        predicted_profit + mae
    );
    separator(50);
    println!();
}

// Prints state-by-state profit averages with ASCII bar chart.
pub fn print_state_breakdown(startups: &[Startup]) {
    use std::collections::HashMap;

    let mut state_profits: HashMap<&str, Vec<f64>> = HashMap::new();
    for s in startups {
        state_profits.entry(s.state.trim()).or_default().push(s.profit);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    let mut avgs: Vec<(&str, f64)> = state_profits
        .iter()
        .map(|(state, vals)| (*state, mean(vals)))
        .collect();
    avgs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let max_avg = avgs.iter().map(|x| x.1).fold(0.0_f64, f64::max);

    separator(50);
    println!("  AVERAGE PROFIT BY STATE");
    separator(50);
    for (state, avg) in &avgs {
        println!(
            "  {:<14}  ${:>9.0}  [{}]",
            state, avg, bar(*avg, max_avg, 16)
        );
    }
    separator(50);
    println!();
}

// Replace your existing print_model_equation in report.rs with this version.

pub fn print_model_equation(coefficients: &[f64], intercept: f64) {
    let feature_names = ["R&D Spend", "Administration", "Marketing", "State (encoded)"];

    separator(58);
    println!("  THE LEARNED MODEL EQUATION");
    separator(58);
    // Each coefficient = how much profit changes per $1 increase in that feature,
    // holding all other features constant. This is the core of linear regression.
    println!("  {:22}  {:>12}  {}", "Feature", "Coefficient", "Meaning");
    separator(58);

    for (i, name) in feature_names.iter().enumerate() {
        // Guard against unexpected coefficient count -- never panic on index
        let coef = match coefficients.get(i) {
            Some(c) => *c,
            None => {
                println!("  {:22}  {:>12}  {}", name, "N/A", "not returned by model");
                continue;
            }
        };

        let meaning = if coef > 0.5 {
            format!("${:.2} added to profit per $1 increase", coef)
        } else if coef < -0.5 {
            format!("${:.2} removed from profit per $1 increase", coef.abs())
        } else {
            // Near-zero means this feature doesn't independently move profit much.
            // It doesn't mean it's useless -- it may be correlated with R&D.
            "near-zero independent effect".to_string()
        };

        println!("  {:22}  {:>12.4}  {}", name, coef, meaning);
    }

    separator(58);
    println!("  {:22}  {:>12.2}  baseline constant", "Intercept", intercept);
    separator(58);
    println!();

    // Only print the full equation if we have all 4 coefficients
    if coefficients.len() == 4 {
        println!("  Full equation:");
        println!(
            "  Profit = {:.2}",
            intercept
        );
        println!("         + ({:.4} x R&D)", coefficients[0]);
        println!("         + ({:.4} x Administration)", coefficients[1]);
        println!("         + ({:.4} x Marketing)", coefficients[2]);
        println!("         + ({:.4} x State)", coefficients[3]);
        separator(58);
        println!();
    }
}