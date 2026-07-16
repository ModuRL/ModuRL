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
        config.cflag("-DWIN32=1").cxxflag("-DWIN32=1");
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
