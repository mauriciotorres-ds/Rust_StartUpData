// model.rs -- responsible for ONLY one thing: the ML model.
// Training, evaluating, and predicting all live here.
// No CSV logic, no printing -- just math and model operations.

use ndarray::{Array1, Array2};
use linfa::prelude::*;
use linfa_linear::{FittedLinearRegression, LinearRegression};
use std::error::Error;

use crate::data::StartupEncoded;

pub struct ModelMetrics {
    pub r_squared: f64,
    pub mae: f64,
    pub rmse: f64,
    pub predictions: Vec<f64>,
    pub coefficients: Vec<f64>, // one weight per feature: [rd, admin, marketing, state]
    pub intercept: f64,         // the baseline constant term
}

pub fn train_and_evaluate(data: &[StartupEncoded]) -> Result<(FittedLinearRegression<f64>, ModelMetrics), Box<dyn Error>> {

    let n = data.len();
    let n_features = 4;

    let x_flat: Vec<f64> = data
        .iter()
        .flat_map(|s| vec![s.rd, s.administration, s.marketing, s.state_encoded])
        .collect();

    let x: Array2<f64> = Array2::from_shape_vec((n, n_features), x_flat)?;
    let y: Array1<f64> = Array1::from_vec(data.iter().map(|s| s.profit).collect());

    let dataset = Dataset::new(x.clone(), y.clone());

    let model = LinearRegression::default().fit(&dataset)?;

    // linfa stores the intercept separately from the feature weights.
    // .intercept() returns a plain f64 (not an array) for single-target regression.
    // .params() returns ONLY the feature coefficients, no intercept mixed in.
    // This is different from sklearn which bundles them -- linfa keeps them split.
    let intercept = model.intercept();
    let coefficients: Vec<f64> = model.params().iter().cloned().collect();

    // Debug print so we can verify the count is correct (should say 4)
    println!("  [model] Coefficients extracted: {}", coefficients.len());

    let predictions_array = model.predict(&x);
    let predictions: Vec<f64> = predictions_array.iter().cloned().collect();

    let actuals: Vec<f64> = y.iter().cloned().collect();
    let r_squared = compute_r_squared(&actuals, &predictions);
    let mae       = compute_mae(&actuals, &predictions);
    let rmse      = compute_rmse(&actuals, &predictions);

    println!("  [model] Training complete. R2 = {:.4}", r_squared);

    Ok((model, ModelMetrics { r_squared, mae, rmse, predictions, coefficients, intercept }))
}

pub fn predict_single(
    rd: f64,
    administration: f64,
    marketing: f64,
    state: &str,
    model: &FittedLinearRegression<f64>,
) -> Result<f64, Box<dyn Error>> {

    let state_encoded = match state.trim() {
        "California" => 0.0,
        "Florida"    => 1.0,
        "New York"   => 2.0,
        _            => -1.0,
    };

    let input: Array2<f64> = Array2::from_shape_vec(
        (1, 4),
        vec![rd, administration, marketing, state_encoded],
    )?;

    let result = model.predict(&input);
    Ok(result[0])
}

fn compute_r_squared(actuals: &[f64], predictions: &[f64]) -> f64 {
    let mean = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let ss_tot: f64 = actuals.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f64 = actuals.iter().zip(predictions.iter())
        .map(|(y, p)| (y - p).powi(2))
        .sum();
    1.0 - (ss_res / ss_tot)
}

fn compute_mae(actuals: &[f64], predictions: &[f64]) -> f64 {
    let total: f64 = actuals.iter().zip(predictions.iter())
        .map(|(y, p)| (y - p).abs())
        .sum();
    total / actuals.len() as f64
}

fn compute_rmse(actuals: &[f64], predictions: &[f64]) -> f64 {
    let total: f64 = actuals.iter().zip(predictions.iter())
        .map(|(y, p)| (y - p).powi(2))
        .sum();
    (total / actuals.len() as f64).sqrt()
}