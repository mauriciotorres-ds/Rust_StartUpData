// main.rs -- the orchestrator. This file does NOT contain business logic.
// It only calls the right functions in the right order.
// A senior dev reading this file should understand the entire program flow
// in under 30 seconds without reading any other file.

// `mod` tells Rust "there is a file called X.rs in this same folder, include it."
// This is how Rust's module system works -- you explicitly declare every file.
// Without these declarations, Rust won't compile or even look at those files.
mod data;
mod model;
mod report;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {

    // ------------------------------------------------------------------
    // Stage 1: Load raw data
    // ------------------------------------------------------------------
    // data::load_csv lives in data.rs. The `data::` prefix is the module path.
    // If this fails (file not found, bad format), the error prints and we exit.
    let raw = data::load_csv("data/Startups.csv")?;

    report::print_header(raw.len());
    report::print_overview(&raw);
    report::print_state_breakdown(&raw);

    // ------------------------------------------------------------------
    // Stage 2: Preprocess -- encode categorical variables (State -> number)
    // ------------------------------------------------------------------
    // ML models require all inputs to be numeric.
    // encode() converts "California" -> 0.0, "Florida" -> 1.0, etc.
    let encoded = data::encode(&raw);

    // ------------------------------------------------------------------
    // Stage 3: Train the model and get evaluation metrics
    // ------------------------------------------------------------------
    // train_and_evaluate fits a LinearRegression model on the full dataset
    // and returns metrics (R2, MAE, RMSE) plus all predictions.
    let (trained_model, metrics) = model::train_and_evaluate(&encoded)?;

    report::print_metrics(&metrics);
    report::print_model_equation(&metrics.coefficients, metrics.intercept);
    report::print_predictions(&encoded, &metrics);

    // ------------------------------------------------------------------
    // Stage 4: Predict profit for a brand new, hypothetical startup
    // ------------------------------------------------------------------
    // This is the actual product use case -- given a new startup's spending,
    // what profit should they expect?
    // Change these numbers to test different scenarios.
    let new_rd        = 120_000.0; // underscores in numbers are just visual separators in Rust
    let new_admin     = 100_000.0;
    let new_marketing = 300_000.0;
    let new_state     = "California";

    let predicted = model::predict_single(
        new_rd,
        new_admin,
        new_marketing,
        new_state,
        &trained_model,
    )?;

    report::print_single_prediction(
        new_rd,
        new_admin,
        new_marketing,
        new_state,
        predicted,
        metrics.mae,
    );

    Ok(())
}
