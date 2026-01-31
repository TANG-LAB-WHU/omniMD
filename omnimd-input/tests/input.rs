// OmniMD, an extensible molecular simulation engine
// Copyright (C) OmniMD's contributors — MIT license
#![allow(clippy::needless_return)]

use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::{env, fs, io};

use walkdir::WalkDir;

use libtest_mimic::{Arguments, Failed, Trial};

use omnimd_core::System;
use omnimd_input::{Error, Input, InteractionsInput};

fn main() {
    env_logger::init();
    let _cleanup = TestsCleanup;

    let args = Arguments::from_args();
    let tests = all_tests();

    libtest_mimic::run(&args, tests).exit();
}

fn all_tests() -> Vec<Trial> {
    let mut tests = Vec::new();

    tests.extend(
        generate_tests("simulation/good", |path, content| {
            move || {
                let input = Input::from_str(path.clone(), &content)
                    .map_err(|e| Failed::from(e.to_string()))?;
                input.read().map_err(|e| Failed::from(e.to_string()))?;
                Ok(())
            }
        })
        .expect("Could not generate the tests"),
    );

    tests.extend(
        generate_tests("simulation/bad", |path, content| {
            move || {
                let message = get_error_message(&content);
                let result = Input::from_str(path.clone(), &content).and_then(|input| input.read());

                match result {
                    Err(Error::Config(reason)) => {
                        if reason == message {
                            Ok(())
                        } else {
                            Err(Failed::from(format!(
                                "Expected error message: {}\nGot: {}",
                                message, reason
                            )))
                        }
                    }
                    _ => Err(Failed::from("This test should fail with a Config error")),
                }
            }
        })
        .expect("Could not generate the tests"),
    );

    tests.extend(
        generate_tests("interactions/good", |_, content| {
            move || {
                let mut system = System::new();
                let input = InteractionsInput::from_str(&content)
                    .map_err(|e| Failed::from(e.to_string()))?;
                input.read(&mut system).map_err(|e| Failed::from(e.to_string()))?;
                Ok(())
            }
        })
        .expect("Could not generate the tests"),
    );

    tests.extend(
        generate_tests("interactions/bad", |_, content| {
            move || {
                let message = get_error_message(&content);

                let mut system = System::new();
                let result =
                    InteractionsInput::from_str(&content).and_then(|input| input.read(&mut system));

                match result {
                    Err(Error::Config(reason)) => {
                        if reason == message {
                            Ok(())
                        } else {
                            Err(Failed::from(format!(
                                "Expected error message: {}\nGot: {}",
                                message, reason
                            )))
                        }
                    }
                    _ => Err(Failed::from("This test should fail with a Config error")),
                }
            }
        })
        .expect("Could not generate the tests"),
    );

    return tests;
}

/// Generate the tests by calling `callback` for every TOML files at the given
/// `root`.
fn generate_tests<F, T>(root: &str, callback: F) -> Result<Vec<Trial>, io::Error>
where
    F: Fn(PathBuf, String) -> T,
    T: Fn() -> Result<(), Failed> + Send + 'static,
{
    let mut tests = Vec::new();

    let dir = PathBuf::new().join(env!("CARGO_MANIFEST_DIR")).join("tests").join(root);
    for entry in WalkDir::new(dir) {
        let entry = entry?;
        let file_type = entry.file_type();
        if file_type.is_file() {
            if let Some(extension) = entry.path().extension() {
                if extension == "toml" {
                    let path = entry.path();
                    let name = String::from(root) + "/";
                    let name = name
                        + path
                            .file_name()
                            .expect("Missing file name")
                            .to_str()
                            .expect("File name is invalid UTF-8");

                    let mut content = String::new();
                    File::open(path)
                        .and_then(|mut file| file.read_to_string(&mut content))
                        .expect("Could not read the input file");

                    let count = content.split("+++").count();
                    for (i, test_case) in content.split("+++").enumerate() {
                        let test_name = if count > 1 {
                            format!("{} - {}/{}", name, i + 1, count)
                        } else {
                            name.clone()
                        };
                        let test_fn = callback(path.to_path_buf(), test_case.into());
                        let test = Trial::test(test_name, test_fn);
                        tests.push(test);
                    }
                }
            }
        }
    }

    Ok(tests)
}

/// Cleanup temporary files after the tests
struct TestsCleanup;
impl Drop for TestsCleanup {
    fn drop(&mut self) {
        const REMOVE: &[&str] = &[
            "energy.dat",
            "filename.xyz",
            "cell.dat",
            "properties.dat",
            "file.log",
            "custom.dat",
            "stress.dat",
            "forces.xyz",
        ];

        for file in REMOVE {
            if let Err(err) = fs::remove_file(file) {
                match err.kind() {
                    io::ErrorKind::NotFound => {}
                    _ => panic!("io error in cleanup code: {}", err),
                }
            }
        }
    }
}

fn get_error_message(content: &str) -> String {
    for line in content.lines() {
        let line = line.trim();
        if let Some(message) = line.strip_prefix("#^ ") {
            return message.into();
        }
    }

    panic!("No error message found. Please add one with the '#^ <message>' syntax.");
}
