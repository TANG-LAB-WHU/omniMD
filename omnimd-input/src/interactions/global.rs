// omnimd-input/src/interactions/global.rs

use toml::value::Table;

use omnimd_core::energy::TorchPotential;
use omnimd_core::System;

use crate::interactions::InteractionsInput;
use crate::{Error, FromToml};

impl InteractionsInput {
    /// Read the "global" section from the configuration.
    pub(crate) fn read_globals(&self, system: &mut System) -> Result<(), Error> {
        let globals = match self.config.get("global") {
            Some(globals) => globals,
            None => return Ok(()),
        };

        let globals = globals.as_array().ok_or(Error::from("'global' section must be an array"))?;

        for global in globals {
            let global =
                global.as_table().ok_or(Error::from("Global potential must be a table"))?;

            let key = global.keys().next().ok_or(Error::from("Empty global potential table"))?;

            match key.as_str() {
                "torch" => {
                    let path = global["torch"]
                        .as_str()
                        .ok_or(Error::from("Torch potential must be a string path"))?;
                    let potential = TorchPotential::new(path)
                        .map_err(|e| Error::from(format!("Failed to load torch model: {}", e)))?;
                    system.add_global_potential(Box::new(potential));
                }
                other => return Err(Error::from(format!("Unknown global potential '{}'", other))),
            }
        }
        Ok(())
    }
}
