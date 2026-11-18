//! Webcam capture via nokhwa (Media Foundation on Windows).

use anyhow::Result;
use image::RgbImage;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
    Camera,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

pub fn list() -> Result<Vec<String>> {
    Ok(nokhwa::query(ApiBackend::Auto)?
        .iter().enumerate()
        .map(|(i, c)| format!("{i}: {}", c.human_name()))
        .collect())
}

pub struct Cam {
    pub latest: Arc<Mutex<Option<RgbImage>>>,
    pub counter: Arc<AtomicU64>,
    pub error: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

pub fn start(index: u32, w: u32, h: u32) -> Result<Cam> {
    let latest = Arc::new(Mutex::new(None));
    let error = Arc::new(Mutex::new(None));
    let counter = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (l2, e2, c2, s2) = (latest.clone(), error.clone(), counter.clone(), stop.clone());

    let handle = std::thread::spawn(move || {
        let req = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
            CameraFormat::new(Resolution::new(w, h), FrameFormat::MJPEG, 30),
        ));

        let mut cam = match Camera::new(CameraIndex::Index(index), req) {
            Ok(c) => c,
            Err(e) => { *e2.lock().unwrap() = Some(e.to_string()); return; }
        };
        if let Err(e) = cam.open_stream() {
            *e2.lock().unwrap() = Some(e.to_string());
            return;
        }

        while !s2.load(Ordering::Relaxed) {
            match cam.frame().and_then(|b| b.decode_image::<RgbFormat>()) {
                Ok(img) => {
                    *l2.lock().unwrap() = Some(img);
                    c2.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => { *e2.lock().unwrap() = Some(e.to_string()); break; }
            }
        }
        let _ = cam.stop_stream();
    });

    Ok(Cam { latest, counter, error, stop, handle: Some(handle) })
}

impl Drop for Cam {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
    }
}