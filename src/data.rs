// data.rs -- responsible for ONLY one thing: loading raw CSV data
// and handing it back in a clean, typed format the rest of the app can use.
// No model logic, no printing -- just data in, clean structs out.
// This separation means if the CSV format ever changes, you edit ONE file.

use csv::Reader;
use serde::Deserialize;
use std::error::Error;

// The raw shape of one row in Startups.csv.
// Deserialize lets serde automatically map CSV columns to these fields.
// Debug lets us print a Startup with {:?} during development/inspection.
// Clone lets us make copies of it when needed (linfa will need this).
#[derive(Debug, Deserialize, Clone)]
pub struct Startup {
    // `pub` means public -- other modules (main.rs, model.rs) can access this field.
    // Without pub, fields are private to this file only.
    #[serde(rename = "RD")]
    pub rd: f64,

    #[serde(rename = "Administration")]
    pub administration: f64,

    #[serde(rename = "Marketing")]
    pub marketing: f64,

    // State is text, not a number -- we'll convert it to a number in preprocessing.
    // ML models can't work with raw strings, only numbers.
    #[serde(rename = "State")]
    pub state: String,

    #[serde(rename = "Profit")]
    pub profit: f64,
}

// A cleaned, model-ready version of a Startup row.
// State has been converted from a string ("California") to a number (0.0, 1.0, 2.0).
// This is called "label encoding" -- a standard preprocessing step.
#[derive(Debug, Clone)]
pub struct StartupEncoded {
    pub rd: f64,
    pub administration: f64,
    pub marketing: f64,
    pub state_encoded: f64, // 0.0 = California, 1.0 = Florida, 2.0 = New York
    pub profit: f64,
}

// Loads the CSV from the given path and returns a Vec of raw Startup structs.
// `-> Result<Vec<Startup>, Box<dyn Error>>` means: return the list on success,
// or return any error on failure. The caller decides what to do with errors.
pub fn load_csv(path: &str) -> Result<Vec<Startup>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(path)?;
    let mut startups = Vec::new();

    for result in rdr.deserialize() {
        let record: Startup = result?;
        startups.push(record);
    }

    println!("  [data] Loaded {} records from '{}'", startups.len(), path);
    Ok(startups)
}

// Converts raw Startup structs into encoded structs the model can use.
// The key step here is state encoding -- turning strings into numbers.
// This is called "label encoding" and is one of the most common ML preprocessing steps.
pub fn encode(startups: &[Startup]) -> Vec<StartupEncoded> {
    startups
        .iter()
        .map(|s| {
            // Match the state string to a numeric code.
            // These numbers are arbitrary but consistent -- the model just needs
            // a stable numeric representation for each category.
            let state_encoded = match s.state.trim() {
                "California" => 0.0,
                "Florida"    => 1.0,
                "New York"   => 2.0,
                // If an unexpected state appears, we assign -1.0 as a signal value.
                // In production you'd handle this more robustly, but this is fine for now.
                other => {
                    eprintln!("  [warn] Unknown state '{}', encoding as -1.0", other);
                    -1.0
                }
            };

            StartupEncoded {
                rd: s.rd,
                administration: s.administration,
                marketing: s.marketing,
                state_encoded,
                profit: s.profit,
            }
        })
        .collect()
}
