use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GSV_LIB_DIR");
    if let Some(dir) = env::var_os("GSV_LIB_DIR") {
        println!(
            "cargo:rustc-link-search=native={}",
            PathBuf::from(dir).display()
        );
    } else {
        let fallback = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
            .join("../..")
            .join("build");
        println!("cargo:rustc-link-search=native={}", fallback.display());
    }
    println!("cargo:rustc-link-lib=dylib=git_sha1_cuda");
}
