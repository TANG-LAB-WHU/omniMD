use omnimd_ml_contract::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::process::Command;

/// CUDA index is relative to the parent's CUDA_VISIBLE_DEVICES list when set.
/// The selected worker sees exactly one device and DeepMD always receives rank 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(u32),
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        enum Legacy {
            #[serde(alias = "cpu")]
            Cpu,
            #[serde(alias = "cuda")]
            Cuda(u32),
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Input {
            Text(String),
            Legacy(Legacy),
        }
        match Input::deserialize(deserializer)? {
            Input::Legacy(Legacy::Cpu) => Ok(Self::Cpu),
            Input::Legacy(Legacy::Cuda(i)) => Ok(Self::Cuda(i)),
            Input::Text(s) => match s.as_str() {
                "Cpu" | "cpu" => Ok(Self::Cpu),
                "Cuda" | "cuda" => Ok(Self::Cuda(0)),
                _ => s
                    .strip_prefix("cuda:")
                    .and_then(|s| s.parse().ok())
                    .map(Self::Cuda)
                    .ok_or_else(|| {
                        serde::de::Error::custom("device must be cpu, cuda, cuda:N or {\"Cuda\":N}")
                    }),
            },
        }
    }
}

fn visible_device(index: u32, inherited: Option<&str>) -> Result<String> {
    match inherited {
        None => Ok(index.to_string()),
        Some(list) => {
            let tokens: Vec<_> = list.split(',').map(str::trim).collect();
            // Reject malformed masks instead of interpreting CUDA's truncation rules.
            let mut unique = std::collections::HashSet::new();
            for token in &tokens {
                let valid = token.parse::<u32>().is_ok()
                    || token.starts_with("GPU-")
                    || token.starts_with("MIG-");
                if !valid || !unique.insert(*token) {
                    return Err(Error("CUDA request cannot use an empty, disabled, duplicated or invalid CUDA_VISIBLE_DEVICES mask; select cpu explicitly for CPU execution".into()));
                }
            }
            tokens
                .get(index as usize)
                .map(|s| (*s).to_owned())
                .ok_or_else(|| {
                    Error(format!(
                        "CUDA index {index} is outside inherited visibility list {list:?}"
                    ))
                })
        }
    }
}

impl Device {
    /// Configure a fresh child, without mutating the parent process environment.
    pub fn configure_worker(self, command: &mut Command) -> Result<()> {
        let inherited = std::env::var("CUDA_VISIBLE_DEVICES")
            .map(Some)
            .or_else(|e| match e {
                std::env::VarError::NotPresent => Ok(None),
                _ => Err(Error("CUDA_VISIBLE_DEVICES is not valid UTF-8".into())),
            })?;
        self.configure(command, inherited.as_deref())
    }

    fn configure(self, command: &mut Command, inherited: Option<&str>) -> Result<()> {
        command
            .env("HIP_VISIBLE_DEVICES", "-1")
            .env("ROCR_VISIBLE_DEVICES", "-1");
        match self {
            Self::Cpu => {
                command
                    .env("CUDA_VISIBLE_DEVICES", "-1")
                    .env_remove("OMNIMD_CUDA_INDEX");
            }
            Self::Cuda(index) => {
                command
                    .env("CUDA_VISIBLE_DEVICES", visible_device(index, inherited)?)
                    .env("OMNIMD_CUDA_INDEX", index.to_string());
            }
        }
        Ok(())
    }

    pub(crate) fn validate_environment(self) -> Result<()> {
        for name in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"] {
            if std::env::var(name).as_deref() != Ok("-1") {
                return Err(Error(format!(
                    "DeepMD requires a fresh worker with {name}=-1; use the isolated CLI"
                )));
            }
        }
        let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
        match self {
            Self::Cpu if visible == "-1" => Ok(()),
            Self::Cuda(index) if !visible.contains(',')
                && visible_device(0, Some(&visible)).is_ok()
                && std::env::var("OMNIMD_CUDA_INDEX").ok() == Some(index.to_string()) => Ok(()),
            _ => Err(Error("device visibility does not match the request; launch a fresh process with Device::configure_worker before loading any accelerator runtime".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_and_explicit_device_formats() {
        for (text, expected) in [
            ("\"Cpu\"", Device::Cpu),
            ("\"cpu\"", Device::Cpu),
            ("\"cuda\"", Device::Cuda(0)),
            ("\"cuda:3\"", Device::Cuda(3)),
            ("{\"Cuda\":2}", Device::Cuda(2)),
            ("{\"cuda\":1}", Device::Cuda(1)),
        ] {
            assert_eq!(serde_json::from_str::<Device>(text).unwrap(), expected);
        }
        for text in ["\"gpu\"", "\"cuda:-1\"", "{\"Cuda\":4294967296}"] {
            assert!(serde_json::from_str::<Device>(text).is_err());
        }
    }
    #[test]
    fn logical_visibility_mapping_and_no_parent_mutation() {
        assert_eq!(visible_device(2, None).unwrap(), "2");
        assert_eq!(visible_device(1, Some("3,GPU-abc")).unwrap(), "GPU-abc");
        for list in ["", "-1", "0,0", "1,,2", "0,-1", "abc"] {
            assert!(visible_device(0, Some(list)).is_err(), "{list}");
        }
        assert!(visible_device(2, Some("0,1")).is_err());
        let mut command = Command::new("unused");
        Device::Cuda(1)
            .configure(&mut command, Some("2,5"))
            .unwrap();
        let env: std::collections::HashMap<_, _> = command.get_envs().collect();
        assert_eq!(
            env[std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES")],
            Some(std::ffi::OsStr::new("5"))
        );
        assert_eq!(
            env[std::ffi::OsStr::new("OMNIMD_CUDA_INDEX")],
            Some(std::ffi::OsStr::new("1"))
        );
        Device::Cpu
            .configure(&mut command, Some("garbage"))
            .unwrap();
        let env: std::collections::HashMap<_, _> = command.get_envs().collect();
        assert_eq!(
            env[std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES")],
            Some(std::ffi::OsStr::new("-1"))
        );
    }
}
