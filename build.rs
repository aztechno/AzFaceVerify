//! AzFaceVerify build script - built by Azis Ichwan.
//!
//! Embeds the Windows application icon and version metadata into AzFaceVerify.exe
//! so Explorer shows your icon instead of the generic one.
//!
//! This step is BEST-EFFORT: if icon.ico or the resource compiler is missing,
//! the build prints a warning and continues instead of failing.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=icon.ico");
    println!("cargo:rerun-if-changed=assets/icon.png");
    println!("cargo:rerun-if-changed=icon.png");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let candidates = ["assets/icon.ico", "icon.ico"];
    let icon = candidates
        .iter()
        .find(|p| std::path::Path::new(p).is_file());

    let Some(icon) = icon else {
        println!(
            "cargo:warning=AzFaceVerify: no icon.ico found (looked in assets/icon.ico and icon.ico). \
             The .exe keeps the default Windows icon, but the window icon still works at runtime."
        );
        return;
    };

    let version = format!("{}.0", env!("CARGO_PKG_VERSION"));

    let mut res = winresource::WindowsResource::new();
    res.set_icon(icon);
    res.set("ProductName", "AzFaceVerify");
    res.set("FileDescription", "AzFaceVerify - offline 1:1 face verification");
    res.set("CompanyName", "Azis Ichwan");
    res.set("LegalCopyright", "Built by Azis Ichwan");
    res.set("OriginalFilename", "AzFaceVerify.exe");
    res.set("InternalName", "AzFaceVerify");
    res.set("FileVersion", &version);
    res.set("ProductVersion", &version);

    match res.compile() {
        Ok(_) => println!("cargo:warning=AzFaceVerify: embedded exe icon from {icon}"),
        Err(e) => println!(
            "cargo:warning=AzFaceVerify: could not embed the exe icon ({e}). \
             The app still builds and the window/taskbar icon is loaded at runtime."
        ),
    }
}