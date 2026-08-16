use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let profile = match env::var("OPT_LEVEL").as_deref() {
        Ok("0") => "Debug",
        Ok("1" | "2" | "3") => "Release",
        Ok("s" | "z") => "MinSizeRel",
        Ok(level) => panic!("unknown optimization level: {level}"),
        Err(error) => panic!("OPT_LEVEL is required: {error}"),
    };

    let source = project_root().join("native").join("ale");
    let library_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is required"))
        .join("build")
        .join("lib");
    let mut config = cmake::Config::new(&source);
    config
        .define("USE_SDL", "OFF")
        .define("USE_RLGLUE", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_CPP_LIB", "OFF")
        .define("BUILD_CLI", "OFF")
        .define("BUILD_C_LIB", "ON")
        .define(
            format!("CMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}", profile.to_uppercase()),
            &library_dir,
        )
        .profile(profile)
        .build_target("ale-c-lib-static");

    if cfg!(windows) {
        // The vendored ALE CMake file uses GCC's `-O3`, and cmake-rs
        // replaces the usual MSVC Release flags. Supply MSVC's actual
        // optimizer switch explicitly; otherwise ALE is built effectively
        // unoptimized on Windows. Candle's CUDA kernels use the static release
        // MSVC runtime even in non-release Cargo profiles, so use /MT for ALE
        // in every profile as well. In particular, do not select /MTd for a
        // debug build: Rust does not link the debug CRT by default.
        config
            .define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW")
            .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded")
            .cflag("-DWIN32=1")
            .cflag("/O2")
            .cxxflag("-DWIN32=1")
            .cxxflag("/O2");
    }

    let destination = config.build();
    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=ale_c_static");
    println!("cargo:rerun-if-changed=native/ale/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/ale/src");
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}
