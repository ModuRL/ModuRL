use std::{env, fs, path::PathBuf};

const MUJOCO_VERSION: &str = "3.9.0";

fn main() {
    println!("cargo::rerun-if-env-changed=MUJOCO_DOWNLOAD_DIR");
    println!("cargo::rerun-if-env-changed=MUJOCO_DYNAMIC_LINK_DIR");
    println!("cargo::rerun-if-env-changed=MUJOCO_STATIC_LINK_DIR");

    let installation_root = env::var_os("MUJOCO_DYNAMIC_LINK_DIR")
        .map(PathBuf::from)
        .or_else(|| env::var_os("MUJOCO_STATIC_LINK_DIR").map(PathBuf::from))
        .and_then(|lib| lib.parent().map(PathBuf::from))
        .or_else(|| {
            env::var_os("MUJOCO_DOWNLOAD_DIR")
                .map(|cache| PathBuf::from(cache).join(format!("mujoco-{MUJOCO_VERSION}")))
        });

    let Some(installation_root) = installation_root.filter(|path| path.is_dir()) else {
        // mujoco-rs emits the actionable build-time error when no installation
        // exists, so avoid producing a second, less useful failure here.
        return;
    };

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"));
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };

    let mut files = vec![
        (installation_root.join("LICENSE"), "mujoco-LICENSE.txt"),
        (
            installation_root.join("THIRD_PARTY_NOTICES.txt"),
            "mujoco-THIRD_PARTY_NOTICES.txt",
        ),
    ];
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // Windows searches beside the executable for dependent DLLs.
        files.push((
            installation_root.join("bin").join("mujoco.dll"),
            "mujoco.dll",
        ));
    }

    // Cover normal binaries and Cargo's test/benchmark executables. Keeping
    // MuJoCo's notices beside build products also makes it harder to
    // accidentally redistribute the native library without them.
    for directory in [profile_dir.to_path_buf(), profile_dir.join("deps")] {
        fs::create_dir_all(&directory).expect("failed to create Cargo output directory");
        for (source, destination) in &files {
            if source.is_file() {
                fs::copy(source, directory.join(destination))
                    .expect("failed to place a MuJoCo runtime or notice file beside Cargo output");
            }
        }
    }
}
