// omnimd-input/src/interactions/global.rs

use omnimd_core::energy::TorchPotential;
use omnimd_core::System;

use crate::interactions::InteractionsInput;
use crate::Error;

impl InteractionsInput {
    /// Read the "global" section from the configuration.
    /// This expects [[global]] array format for global potentials.
    /// If [global] is used as a table (for pairs defaults), this is handled by pairs.rs.
    pub(crate) fn read_globals(&self, system: &mut System) -> Result<(), Error> {
        let Some(globals) = self.config.get("global") else {
            return Ok(());
        };

        // If it's a table (used for pairs defaults like cutoff), skip here - pairs.rs handles it
        let Some(globals) = globals.as_array() else {
            return Ok(());
        };

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
                        .map_err(|e| Error::from(format!("Failed to load torch model: {e}")))?;
                    system.add_global_potential(Box::new(potential));
                }
                other => return Err(Error::from(format!("Unknown global potential '{other}'"))),
            }
        }
        Ok(())
    }
}
