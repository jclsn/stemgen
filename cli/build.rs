fn main() {
    // ort copies ONNX Runtime's shared libraries (e.g. libwebgpu_dawn.so) next to the binary, so
    // let the loader find them there without LD_LIBRARY_PATH.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
    }
}
