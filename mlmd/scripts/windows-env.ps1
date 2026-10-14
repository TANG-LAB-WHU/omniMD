# Dot-source with the path to an isolated Rust setup; changes this process only.
# Example: . ./mlmd/scripts/windows-env.ps1 -ToolchainRoot C:/path/to/omnimd-toolchain
param([Parameter(Mandatory=$true)][string]$ToolchainRoot)
$ErrorActionPreference = 'Stop'
$toolchainPath = (Resolve-Path -LiteralPath $ToolchainRoot).Path
$env:RUSTUP_HOME = Join-Path $toolchainPath 'rustup'
$env:CARGO_HOME = Join-Path $toolchainPath 'cargo'
$compilerBin = Join-Path $env:RUSTUP_HOME 'toolchains/stable-x86_64-pc-windows-gnu/lib/rustlib/x86_64-pc-windows-gnu/bin'
if (-not (Test-Path -LiteralPath (Join-Path $env:CARGO_HOME 'bin/cargo.exe'))) {
    throw 'The supplied isolated toolchain does not contain cargo.exe'
}
$env:PATH = (Join-Path $env:CARGO_HOME 'bin') + ';' + (Join-Path $compilerBin 'self-contained') + ';' + $compilerBin + ';' + $env:PATH
