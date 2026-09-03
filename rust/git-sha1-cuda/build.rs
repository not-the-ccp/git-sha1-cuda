use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {}: {}", description, error));
    if !status.success() {
        panic!("failed to {}: command exited with {}", description, status);
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=GSV_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("../..");
    println!(
        "cargo:rerun-if-changed={}",
        source.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source.join("include/git_sha1_cuda.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source.join("src/git_sha1_cuda.cu").display()
    );

    let link_dir = if let Some(dir) = env::var_os("GSV_LIB_DIR") {
        PathBuf::from(dir)
    } else {
        let build = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("native");
        let mut configure = Command::new("cmake");
        configure
            .arg("-S")
            .arg(&source)
            .arg("-B")
            .arg(&build)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("-DBUILD_TESTING=OFF")
            .arg("-DGSV_BUILD_STATIC=OFF");
        if let Some(architectures) = env::var_os("CMAKE_CUDA_ARCHITECTURES") {
            configure.arg(format!(
                "-DCMAKE_CUDA_ARCHITECTURES={}",
                architectures.to_string_lossy()
            ));
        }
        run(&mut configure, "configure the CUDA library with CMake");
        run(
            Command::new("cmake")
                .arg("--build")
                .arg(&build)
                .arg("--config")
                .arg("Release")
                .arg("--target")
                .arg("git_sha1_cuda")
                .arg("-j"),
            "build the CUDA library",
        );
        build
    };

    println!("cargo:rustc-link-search=native={}", link_dir.display());
    println!("cargo:rustc-link-lib=dylib=git_sha1_cuda");
    if env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some(std::ffi::OsStr::new("linux")) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", link_dir.display());
    }
}
