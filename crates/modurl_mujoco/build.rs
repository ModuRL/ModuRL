use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

const MUJOCO_VERSION: &str = "3.9.0";

fn copy_if_needed(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.is_file() && fs::read(source)? == fs::read(destination)? {
        return Ok(());
    }
    match fs::copy(source, destination) {
        Ok(_) => Ok(()),
        Err(error)
            if destination.is_file()
                && (error.kind() == io::ErrorKind::PermissionDenied
                    || error.raw_os_error() == Some(32)) =>
        {
            println!(
                "cargo::warning=kept locked MuJoCo runtime at {}; stop the running executable to replace it",
                destination.display()
            );
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn main() -> io::Result<()> {
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
        })
        .filter(|path| path.is_dir())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "MuJoCo installation not found; configure MUJOCO_DOWNLOAD_DIR or MUJOCO_DYNAMIC_LINK_DIR",
            )
        })?;

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"));

    let mut files = vec![
        (installation_root.join("LICENSE"), "mujoco-LICENSE.txt"),
        (
            installation_root.join("THIRD_PARTY_NOTICES.txt"),
            "mujoco-THIRD_PARTY_NOTICES.txt",
        ),
    ];
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        files.push((
            installation_root.join("bin").join("mujoco.dll"),
            "mujoco.dll",
        ));
    }

    for (source, destination) in &files {
        if source.is_file() || *destination == "mujoco.dll" {
            copy_if_needed(source, &out_dir.join(destination)).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("failed to stage {} in OUT_DIR: {error}", source.display()),
                )
            })?;
        }
    }
    println!("cargo::rustc-link-search=native={}", out_dir.display());
    Ok(())
}
