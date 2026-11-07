//! NVRTC compilation with an in-memory module cache and an on-disk PTX cache.
//!
//! The disk cache lives in `$OMNIMD_KERNEL_CACHE` (default: the system temp
//! directory's `omnimd-kernel-cache`; an empty value disables it). Entries are
//! keyed by SHA-256 of the source, target architecture and compile options, so a
//! stale entry is never reused. The driver's own JIT cache then covers PTX→SASS.
use crate::cuda_error;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use omnimd_ml_contract::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Bumped whenever the compile pipeline changes in a way the key cannot see.
const CACHE_FORMAT: &str = "omnimd-kernel-cache-v1 cudarc-0.19 cuda-12080";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub memory_hits: u64,
    pub disk_hits: u64,
    pub compiled: u64,
}

pub struct KernelCache {
    context: Arc<CudaContext>,
    options: Vec<String>,
    dir: Option<PathBuf>,
    modules: Mutex<HashMap<String, Arc<CudaModule>>>,
    stats: Mutex<CacheStats>,
}

fn default_dir() -> Option<PathBuf> {
    match std::env::var_os("OMNIMD_KERNEL_CACHE") {
        Some(value) if value.is_empty() => None,
        Some(value) => Some(PathBuf::from(value)),
        None => Some(std::env::temp_dir().join("omnimd-kernel-cache")),
    }
}

/// Hex SHA-256 over everything that changes the compiled PTX.
pub(crate) fn cache_key(source: &str, options: &[String]) -> String {
    let mut hash = Sha256::new();
    for part in std::iter::once(CACHE_FORMAT)
        .chain(options.iter().map(String::as_str))
        .chain(std::iter::once(source))
    {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part.as_bytes());
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl KernelCache {
    pub(crate) fn new(context: Arc<CudaContext>, (major, minor): (i32, i32)) -> Self {
        Self {
            context,
            options: vec![
                format!("--gpu-architecture=compute_{major}{minor}"),
                "--std=c++17".into(),
            ],
            dir: default_dir(),
            modules: Mutex::new(HashMap::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// A separate cache over the same context, reading and writing `dir`.
    /// Lets tests observe disk hits without touching the process-wide cache.
    pub fn sibling(&self, dir: Option<PathBuf>) -> Self {
        Self {
            context: self.context.clone(),
            options: self.options.clone(),
            dir,
            modules: Mutex::new(HashMap::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    pub fn stats(&self) -> CacheStats {
        *self.stats.lock().unwrap()
    }

    pub fn function(&self, source: &str, name: &str) -> Result<CudaFunction> {
        self.module(source)?
            .load_function(name)
            .map_err(|error| Error(format!("kernel {name} not found in module: {error:?}")))
    }

    fn module(&self, source: &str) -> Result<Arc<CudaModule>> {
        let key = cache_key(source, &self.options);
        if let Some(module) = self.modules.lock().unwrap().get(&key) {
            self.stats.lock().unwrap().memory_hits += 1;
            return Ok(module.clone());
        }
        let path = self.dir.as_ref().map(|dir| dir.join(format!("{key}.ptx")));
        let cached = path
            .as_deref()
            .and_then(|p| std::fs::read_to_string(p).ok());
        let ptx = match cached {
            Some(text) => {
                self.stats.lock().unwrap().disk_hits += 1;
                text
            }
            None => {
                let text = self.compile(source)?;
                self.stats.lock().unwrap().compiled += 1;
                if let Some(path) = &path {
                    // A cache that cannot be written only costs a recompile.
                    let _ = store(path, &text);
                }
                text
            }
        };
        let module = self
            .context
            .load_module(Ptx::from_src(ptx))
            .map_err(cuda_error("module load"))?;
        self.modules.lock().unwrap().insert(key, module.clone());
        Ok(module)
    }

    fn compile(&self, source: &str) -> Result<String> {
        let options = CompileOptions {
            options: self.options.clone(),
            ..Default::default()
        };
        compile_ptx_with_opts(source, options)
            .map(|ptx| ptx.to_src())
            .map_err(|error| Error(format!("NVRTC compilation failed: {error}")))
    }
}

/// Write-then-rename so a concurrent reader never sees a partial file.
fn store(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = path.with_extension(format!("{}.partial", std::process::id()));
    std::fs::write(&partial, text)?;
    std::fs::rename(&partial, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&partial);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_covers_source_and_options() {
        let options = vec!["--gpu-architecture=compute_89".to_string()];
        let base = cache_key("a", &options);
        assert_eq!(base, cache_key("a", &options));
        assert_eq!(base.len(), 64);
        assert_ne!(base, cache_key("b", &options));
        assert_ne!(
            base,
            cache_key("a", &["--gpu-architecture=compute_80".to_string()])
        );
        // Length prefixes keep part boundaries unambiguous.
        assert_ne!(
            cache_key("bc", &["a".to_string()]),
            cache_key("c", &["ab".to_string()])
        );
    }
}
