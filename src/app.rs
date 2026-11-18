//! AzFaceVerify GUI (egui / eframe).
//! Built by Azis Ichwan.
//!
//! Layout:
//!   Top    : tab bar + status
//!   Bottom : pinned result panel (Photo tab) - resizable and scrollable
//!   Center : scrollable photo panels / webcam / deep analysis / settings

use crate::analyze::{self, Calibration, Level, Quality};
use crate::cam::Cam;
use crate::embed;
use crate::engine::Provider;
use crate::regions::{self, DeepReport, GEO_LABELS};
use crate::worker::{Analysis, Entry, Job, Reply, Settings, Tag};
use eframe::egui::{
    self, Color32, ColorImage, Pos2, Rect, RichText, Stroke, TextureHandle, TextureOptions,
};
use image::RgbImage;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

const ENROLL_N: usize = 12;

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Photo,
    Webcam,
    Analysis,
    Settings,
}

#[derive(Default)]
struct Slot {
    path: String,
    img: Option<RgbImage>,
    tex: Option<TextureHandle>,
    analysis: Option<Analysis>,
    thumbs: Vec<TextureHandle>,
    chosen: usize,
    seq: u64,
    pending: bool,
}

/// A webcam-tab reference: either a single photo or an averaged enrolment.
#[derive(Clone)]
struct Reference {
    entry: Entry,
    source: String,
    frames: usize,
    thumb: Option<TextureHandle>,
}

enum Action {
    Open,
    Paste,
    Rotate(bool),
    Clear,
}

pub struct App {
    models: PathBuf,
    tx: Sender<Job>,
    rx: Receiver<Reply>,
    cfg: Settings,
    cal: Calibration,
    tab: Tab,

    a: Slot,
    b: Slot,
    live: Slot,
    a_rect: Option<Rect>,
    b_rect: Option<Rect>,

    cam: Option<Cam>,
    cam_list: Vec<String>,
    cam_idx: u32,
    cam_tex: Option<TextureHandle>,
    cam_seen: u64,
    cam_fps: f32,
    fps_t: Instant,
    fps_n: u64,

    live_pending: bool,
    live_hist: Vec<f32>,
    live_match: Option<Tag>,
    enroll: Option<(Tag, Vec<Vec<f32>>)>,
    ref_a: Option<Reference>,
    ref_b: Option<Reference>,
    ref_src_a: String,
    ref_src_b: String,

    deep: Option<DeepReport>,
    deep_tex: Option<(TextureHandle, TextureHandle, TextureHandle)>,
    deep_pending: bool,
    deep_source: String,

    preview_max_h: f32,
    icon_info: String,
    last_report: Option<String>,

    status: String,
    ready: bool,
    seq: u64,
}

fn tex(ctx: &egui::Context, img: &RgbImage, name: &str) -> TextureHandle {
    ctx.load_texture(
        name,
        ColorImage::from_rgb([img.width() as usize, img.height() as usize], img.as_raw()),
        TextureOptions::LINEAR,
    )
}

fn stamp() -> (String, String, i64) {
    let now = chrono::Local::now();
    (
        now.format("%Y-%m-%d %H:%M:%S %:z").to_string(),
        now.format("%Y%m%d-%H%M%S").to_string(),
        now.timestamp(),
    )
}

fn verdict_slug(level: Level) -> &'static str {
    match level {
        Level::StrongSame => "SAME-STRONG",
        Level::Same => "SAME",
        Level::Unsure => "INCONCLUSIVE",
        Level::Diff => "DIFFERENT",
        Level::StrongDiff => "DIFFERENT-STRONG",
    }
}

fn quality_line(e: &Entry) -> String {
    let q = &e.quality;
    format!(
        "size={:.0}px det={:.3} blur={:.1} brightness={:.1} contrast={:.1} yaw={:+.3} roll={:+.1}deg pitch={:.3}",
        q.size, q.det, q.blur, q.bright, q.contrast, q.yaw, q.roll, q.pitch
    )
}

fn delta_color(d: f32) -> Color32 {
    if d >= 0.03 {
        Color32::from_rgb(80, 225, 130)
    } else if d <= -0.03 {
        Color32::from_rgb(240, 110, 90)
    } else {
        Color32::from_rgb(160, 165, 175)
    }
}

pub fn run(models: PathBuf, provider: Provider) -> anyhow::Result<()> {
    let cfg = Settings { provider, ..Default::default() };
    let (tx, rx) = crate::worker::spawn(models.clone(), cfg);
    let (icon, icon_info) = crate::icon::load();

    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1420.0, 940.0])
            .with_min_inner_size([960.0, 620.0])
            .with_title("AzFaceVerify - built by Azis Ichwan")
            .with_icon(icon),
        ..Default::default()
    };

    eframe::run_native(
        "AzFaceVerify",
        opts,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(App {
                models: models.clone(),
                tx,
                rx,
                cfg,
                cal: Calibration::default(),
                tab: Tab::Photo,
                a: Slot::default(),
                b: Slot::default(),
                live: Slot::default(),
                a_rect: None,
                b_rect: None,
                cam: None,
                cam_list: Vec::new(),
                cam_idx: 0,
                cam_tex: None,
                cam_seen: 0,
                cam_fps: 0.0,
                fps_t: Instant::now(),
                fps_n: 0,
                live_pending: false,
                live_hist: Vec::new(),
                live_match: None,
                enroll: None,
                ref_a: None,
                ref_b: None,
                ref_src_a: String::new(),
                ref_src_b: String::new(),
                deep: None,
                deep_tex: None,
                deep_pending: false,
                deep_source: String::new(),
                preview_max_h: 430.0,
                icon_info: icon_info.clone(),
                last_report: None,
                status: "loading models...".into(),
                ready: false,
                seq: 0,
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))
}

impl App {
    fn slot_mut(&mut self, t: Tag) -> &mut Slot {
        match t {
            Tag::A => &mut self.a,
            Tag::B => &mut self.b,
            _ => &mut self.live,
        }
    }

    fn set_image(&mut self, ctx: &egui::Context, t: Tag, img: RgbImage, name: String) {
        self.seq += 1;
        let seq = self.seq;
        let texture = tex(ctx, &img, "src");
        let job_img = img.clone();

        {
            let s = self.slot_mut(t);
            s.tex = Some(texture);
            s.img = Some(img);
            s.analysis = None;
            s.thumbs.clear();
            s.chosen = 0;
            s.seq = seq;
            s.pending = true;
            s.path = name;
        }

        let _ = self.tx.send(Job::Analyze { tag: t, seq, img: job_img, fast: false });
    }

    fn apply(&mut self, ctx: &egui::Context, t: Tag, act: Action) {
        match act {
            Action::Open => {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Images", &["jpg", "jpeg", "png", "bmp", "webp"])
                    .pick_file()
                {
                    match image::open(&p) {
                        Ok(d) => self.set_image(ctx, t, d.to_rgb8(), p.display().to_string()),
                        Err(e) => self.status = format!("failed to open file: {e}"),
                    }
                }
            }

            Action::Paste => match arboard::Clipboard::new().and_then(|mut c| c.get_image()) {
                Ok(d) => {
                    if let Some(rgba) = image::RgbaImage::from_raw(
                        d.width as u32, d.height as u32, d.bytes.into_owned(),
                    ) {
                        let rgb = image::DynamicImage::ImageRgba8(rgba).to_rgb8();
                        self.set_image(ctx, t, rgb, "clipboard".into());
                    } else {
                        self.status = "clipboard image has an unsupported layout".into();
                    }
                }
                Err(_) => self.status = "clipboard does not contain an image".into(),
            },

            Action::Rotate(cw) => {
                let current = self.slot_mut(t).img.clone();
                if let Some(img) = current {
                    let rotated = if cw {
                        image::imageops::rotate90(&img)
                    } else {
                        image::imageops::rotate270(&img)
                    };
                    let name = self.slot_mut(t).path.clone();
                    self.set_image(ctx, t, rotated, name);
                }
            }

            Action::Clear => *self.slot_mut(t) = Slot::default(),
        }
    }

    fn pump(&mut self, ctx: &egui::Context) {
        while let Ok(r) = self.rx.try_recv() {
            match r {
                Reply::Ready(s) => {
                    self.status = s;
                    self.ready = true;
                }

                Reply::Failed(e) => {
                    self.status = format!("ERROR: {e}");
                    self.a.pending = false;
                    self.b.pending = false;
                    self.live_pending = false;
                    self.deep_pending = false;
                }

                Reply::Deep(rep) => {
                    self.deep_tex = Some((
                        tex(ctx, &rep.aligned_a, "deep_a"),
                        tex(ctx, &rep.aligned_b, "deep_b"),
                        tex(ctx, &rep.heatmap, "deep_heat"),
                    ));
                    self.status = format!(
                        "deep analysis done: {} model runs in {:.0} ms",
                        rep.forwards, rep.ms
                    );
                    self.deep = Some(*rep);
                    self.deep_pending = false;
                }

                Reply::Done { tag, seq, analysis } => {
                    if tag == Tag::Live {
                        self.live_pending = false;
                        let mut finished: Option<(Tag, Reference)> = None;

                        if let Some(e) = analysis.entries.first() {
                            if let Some((target, buf)) = self.enroll.as_mut() {
                                if e.face.short_side() >= 70.0 && e.quality.blur >= 25.0 {
                                    buf.push(e.emb.clone());
                                }
                                if buf.len() >= ENROLL_N {
                                    let mut avg = vec![0f32; e.emb.len()];
                                    for v in buf.iter() {
                                        for i in 0..avg.len() {
                                            avg[i] += v[i];
                                        }
                                    }
                                    embed::normalize(&mut avg);
                                    let mut entry = e.clone();
                                    entry.emb = avg;
                                    let thumb = tex(ctx, &entry.aligned, "refthumb");
                                    finished = Some((
                                        *target,
                                        Reference {
                                            entry,
                                            source: format!("webcam enrolment ({ENROLL_N} frames)"),
                                            frames: ENROLL_N,
                                            thumb: Some(thumb),
                                        },
                                    ));
                                }
                            }

                            if let Some(t) = self.live_match {
                                let r = if t == Tag::A { self.ref_a.as_ref() } else { self.ref_b.as_ref() };
                                if let Some(rv) = r {
                                    self.live_hist.push(embed::cosine(&rv.entry.emb, &e.emb));
                                    if self.live_hist.len() > 15 {
                                        self.live_hist.remove(0);
                                    }
                                }
                            }
                        } else {
                            self.live_hist.clear();
                        }

                        if let Some((t, r)) = finished {
                            if t == Tag::A { self.ref_a = Some(r); } else { self.ref_b = Some(r); }
                            self.enroll = None;
                            self.live_hist.clear();
                            self.status = "webcam enrolment finished".into();
                        }

                        self.live.thumbs = analysis
                            .entries
                            .first()
                            .map(|e| vec![tex(ctx, &e.aligned, "livecrop")])
                            .unwrap_or_default();
                        self.live.analysis = Some(analysis);
                    } else if tag == Tag::RefA || tag == Tag::RefB {
                        if let Some(e) = analysis.entries.first() {
                            let thumb = tex(ctx, &e.aligned, "refthumb");
                            let source = if tag == Tag::RefA {
                                self.ref_src_a.clone()
                            } else {
                                self.ref_src_b.clone()
                            };
                            let r = Reference {
                                entry: e.clone(),
                                source,
                                frames: 1,
                                thumb: Some(thumb),
                            };
                            if tag == Tag::RefA { self.ref_a = Some(r); } else { self.ref_b = Some(r); }
                            self.live_hist.clear();
                            self.status = format!(
                                "reference {} loaded from photo",
                                if tag == Tag::RefA { "A" } else { "B" }
                            );
                        } else {
                            self.status = "no face found in that reference photo".into();
                        }
                    } else {
                        let slot = self.slot_mut(tag);
                        if slot.seq == seq {
                            slot.thumbs = analysis
                                .entries
                                .iter()
                                .enumerate()
                                .map(|(i, e)| tex(ctx, &e.aligned, &format!("th{tag:?}{i}")))
                                .collect();
                            slot.analysis = Some(analysis);
                            slot.pending = false;
                        }
                    }
                }
            }
        }
    }

    fn pump_camera(&mut self, ctx: &egui::Context) {
        let Some(cam) = &self.cam else { return };

        if let Some(e) = cam.error.lock().unwrap().clone() {
            self.status = format!("camera error: {e}");
        }

        let c = cam.counter.load(std::sync::atomic::Ordering::Relaxed);
        if c != self.cam_seen {
            self.cam_seen = c;
            self.fps_n += 1;
            if self.fps_t.elapsed() >= Duration::from_millis(500) {
                self.cam_fps = self.fps_n as f32 / self.fps_t.elapsed().as_secs_f32();
                self.fps_t = Instant::now();
                self.fps_n = 0;
            }

            let frame = cam.latest.lock().unwrap().clone();
            if let Some(img) = frame {
                self.cam_tex = Some(tex(ctx, &img, "cam"));
                let wanted = self.enroll.is_some() || self.live_match.is_some();
                if wanted && !self.live_pending && self.ready {
                    self.live_pending = true;
                    let _ = self.tx.send(Job::Analyze { tag: Tag::Live, seq: 0, img, fast: true });
                }
            }
        }
        ctx.request_repaint_after(Duration::from_millis(16));
    }

    fn handle_drops(&mut self, ctx: &egui::Context) {
        if self.tab != Tab::Photo {
            return;
        }
        let files = ctx.input(|i| i.raw.dropped_files.clone());
        if files.is_empty() {
            return;
        }

        let pos = ctx.input(|i| i.pointer.hover_pos());
        let mut target = Tag::A;
        if let Some(p) = pos {
            if self.b_rect.map_or(false, |r| r.contains(p)) {
                target = Tag::B;
            }
        }

        for (k, f) in files.iter().enumerate() {
            if let Some(p) = &f.path {
                if let Ok(d) = image::open(p) {
                    let t = if k == 0 { target } else if target == Tag::A { Tag::B } else { Tag::A };
                    self.set_image(ctx, t, d.to_rgb8(), p.display().to_string());
                } else {
                    self.status = format!("unsupported file: {}", p.display());
                }
            }
        }
    }

    fn slot_ui(&mut self, ui: &mut egui::Ui, tag: Tag) -> Option<Action> {
        let Self { a, b, a_rect, b_rect, preview_max_h, .. } = self;
        let slot = if tag == Tag::A { &mut *a } else { &mut *b };
        let max_h = *preview_max_h;

        let mut act = None;
        let mut new_chosen = None;

        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.heading(if tag == Tag::A { "Photo A" } else { "Photo B" });
                if ui.button("Open").clicked() { act = Some(Action::Open); }
                if ui.button("Paste").clicked() { act = Some(Action::Paste); }
                if ui.button("Rotate L").clicked() { act = Some(Action::Rotate(false)); }
                if ui.button("Rotate R").clicked() { act = Some(Action::Rotate(true)); }
                if ui.button("Clear").clicked() { act = Some(Action::Clear); }
            });
            ui.label(RichText::new(&slot.path).small().weak());

            if let (Some(img), Some(t)) = (&slot.img, &slot.tex) {
                let max_w = ui.available_width().min(620.0);
                let scale = (max_w / img.width() as f32).min(max_h / img.height() as f32);
                let size = egui::vec2(img.width() as f32 * scale, img.height() as f32 * scale);

                ui.label(
                    RichText::new(format!(
                        "{} x {} px  (preview {:.0}%)",
                        img.width(), img.height(), scale * 100.0
                    ))
                    .small()
                    .weak(),
                );

                let resp = ui.add(
                    egui::Image::new(t).fit_to_exact_size(size).sense(egui::Sense::click()),
                );
                let rect = resp.rect;
                if tag == Tag::A { *a_rect = Some(rect); } else { *b_rect = Some(rect); }

                if let Some(an) = &slot.analysis {
                    let p = ui.painter();
                    for (i, e) in an.entries.iter().enumerate() {
                        let f = &e.face;
                        let r = Rect::from_min_max(
                            Pos2::new(rect.min.x + f.bbox[0] * scale, rect.min.y + f.bbox[1] * scale),
                            Pos2::new(rect.min.x + f.bbox[2] * scale, rect.min.y + f.bbox[3] * scale),
                        );
                        let col = if i == slot.chosen {
                            Color32::from_rgb(80, 230, 130)
                        } else {
                            Color32::from_rgb(130, 170, 255)
                        };
                        p.rect_stroke(r, 4.0_f32, Stroke::new(2.0_f32, col));
                        for k in f.kps {
                            p.circle_filled(
                                Pos2::new(rect.min.x + k[0] * scale, rect.min.y + k[1] * scale),
                                2.5,
                                Color32::YELLOW,
                            );
                        }
                        p.text(
                            r.left_top(),
                            egui::Align2::LEFT_BOTTOM,
                            format!("#{} {:.2}", i + 1, f.score),
                            egui::FontId::proportional(13.0),
                            col,
                        );
                    }

                    if resp.clicked() {
                        if let Some(pt) = resp.interact_pointer_pos() {
                            let ix = (pt.x - rect.min.x) / scale;
                            let iy = (pt.y - rect.min.y) / scale;
                            new_chosen = an
                                .entries
                                .iter()
                                .enumerate()
                                .filter(|(_, e)| {
                                    let b = e.face.bbox;
                                    b[0] <= ix && ix <= b[2] && b[1] <= iy && iy <= b[3]
                                })
                                .min_by_key(|(_, e)| e.face.area() as i32)
                                .map(|(i, _)| i);
                        }
                    }
                }
            } else {
                ui.add_sized(
                    [ui.available_width(), 240.0],
                    egui::Label::new(
                        RichText::new("Drag & drop an image here,\nor use Open / Paste")
                            .weak()
                            .size(16.0),
                    ),
                );
            }

            if slot.pending {
                ui.label(RichText::new("analysing...").italics());
            }

            if let Some(an) = &slot.analysis {
                if an.entries.is_empty() {
                    ui.colored_label(
                        Color32::YELLOW,
                        "No face detected. Try a larger / brighter photo, rotate it, or lower the detection threshold.",
                    );
                } else {
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "{} face(s) - {:.0} ms total ({:.0} ms detection)",
                            an.entries.len(), an.ms, an.det_ms
                        ));
                        if an.entries.len() > 1 {
                            egui::ComboBox::new(format!("pick{tag:?}"), "face")
                                .selected_text(format!("#{}", slot.chosen + 1))
                                .show_ui(ui, |ui| {
                                    for i in 0..an.entries.len() {
                                        if ui.selectable_label(i == slot.chosen, format!("#{}", i + 1)).clicked() {
                                            new_chosen = Some(i);
                                        }
                                    }
                                });
                        }
                    });

                    if let Some(e) = an.entries.get(slot.chosen) {
                        ui.horizontal_top(|ui| {
                            if let Some(t) = slot.thumbs.get(slot.chosen) {
                                ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(88.0, 88.0)));
                            }
                            ui.vertical(|ui| quality_ui(ui, &e.quality));
                        });
                    }
                }
            }
        });

        if let Some(i) = new_chosen {
            slot.chosen = i;
        }
        act
    }

    fn photo_tab(&mut self, ui: &mut egui::Ui) {
        let mut acts = Vec::new();

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(2, |cols| {
                if let Some(a) = self.slot_ui(&mut cols[0], Tag::A) { acts.push((Tag::A, a)); }
                if let Some(a) = self.slot_ui(&mut cols[1], Tag::B) { acts.push((Tag::B, a)); }
            });
        });

        let ctx = ui.ctx().clone();
        for (t, a) in acts {
            self.apply(&ctx, t, a);
        }
    }

    fn photo_entry(&self, t: Tag) -> Option<Entry> {
        let slot = if t == Tag::A { &self.a } else { &self.b };
        slot.analysis.as_ref().and_then(|x| x.entries.get(slot.chosen)).cloned()
    }

    fn result_panel(&mut self, ui: &mut egui::Ui) {
        let ea = self.a.analysis.as_ref().and_then(|x| x.entries.get(self.a.chosen));
        let eb = self.b.analysis.as_ref().and_then(|x| x.entries.get(self.b.chosen));

        let mut want_txt = false;
        let mut want_json = false;
        let mut want_deep = false;
        let mut pair: Option<(Entry, Entry)> = None;

        match (ea, eb) {
            (Some(x), Some(y)) => {
                self.verdict_ui(ui, &x.emb, &y.emb, "Photo A", "Photo B");

                ui.collapsing("Technical details & export", |ui| {
                    ui.label(format!(
                        "embedding: 512-d, L2-normalised - flip-TTA {}",
                        if self.cfg.flip_tta { "ON" } else { "OFF" }
                    ));
                    ui.label(format!("A: {}", quality_line(x)));
                    ui.label(format!("B: {}", quality_line(y)));

                    ui.horizontal(|ui| {
                        want_txt = ui.button("Save report (.txt)").clicked();
                        want_json = ui.button("Save report (.json)").clicked();
                        want_deep = ui
                            .button(RichText::new("Deep region analysis ->").strong())
                            .on_hover_text("Which facial area supports / contradicts the match")
                            .clicked();
                    });

                    if let Some(last) = &self.last_report {
                        ui.label(RichText::new(format!("last export: {last}")).small().weak());
                    }
                });

                if want_txt || want_json || want_deep {
                    pair = Some((x.clone(), y.clone()));
                }
            }
            _ => {
                let missing = match (self.a.img.is_some(), self.b.img.is_some()) {
                    (false, false) => "Load Photo A and Photo B to see the analysis.",
                    (true, false) => "Photo A loaded. Now load Photo B.",
                    (false, true) => "Photo B loaded. Now load Photo A.",
                    _ => "Waiting for face detection on both photos...",
                };
                ui.label(missing);
                ui.label(
                    RichText::new("This result panel is pinned at the bottom - drag its top edge to resize it.")
                        .small()
                        .weak(),
                );
            }
        }

        if let Some((a, b)) = pair {
            if want_txt { self.export_txt(&a, &b); }
            if want_json { self.export_json(&a, &b); }
            if want_deep {
                self.deep_pending = true;
                self.deep = None;
                self.deep_tex = None;
                self.deep_source = "Photo A <-> Photo B".into();
                self.tab = Tab::Analysis;
                let _ = self.tx.send(Job::DeepAnalyze { a, b });
            }
        }
    }

    fn verdict_ui(&self, ui: &mut egui::Ui, a: &[f32], b: &[f32], na: &str, nb: &str) {
        let v = analyze::verdict(a, b, &self.cal);
        let col = match v.level {
            Level::StrongSame => Color32::from_rgb(60, 220, 120),
            Level::Same => Color32::from_rgb(150, 220, 100),
            Level::Unsure => Color32::from_rgb(240, 200, 60),
            Level::Diff => Color32::from_rgb(240, 150, 70),
            Level::StrongDiff => Color32::from_rgb(240, 90, 90),
        };

        ui.group(|ui| {
            ui.label(RichText::new(format!("{na}  <->  {nb}")).weak());
            ui.heading(RichText::new(v.label).color(col).size(22.0));

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(format!("{:.1}%", v.prob * 100.0))
                            .size(46.0).strong().color(col),
                    );
                    ui.label(RichText::new("calibrated probability").small().weak());
                });
                ui.vertical(|ui| {
                    ui.label(format!("Cosine similarity  : {:.4}", v.cos));
                    ui.label(format!("L2 distance        : {:.4}", v.l2));
                    ui.label(format!("Decision threshold : {:.3}", self.cal.threshold));
                    ui.add(egui::ProgressBar::new(v.prob).show_percentage());
                });
            });

            ui.collapsing("How to read these numbers (important)", |ui| {
                ui.label("- Cosine similarity is the raw model score (-1..1). It is NOT a percentage of likeness.");
                ui.label("- The percentage above is sigmoid(slope * (cos - mid)), i.e. a calibrated probability.");
                ui.label("- Without calibration on YOUR own data, that percentage is only a heuristic.");
                ui.label("- Run: AzFaceVerify calibrate pairs.csv --out calibration.json");
                ui.label("- An explicit INCONCLUSIVE band exists on purpose.");
                ui.label("- Glasses, moustache, beard, hat/head-covering and makeup are generally handled well.");
                ui.label("- Heavy occlusion (mask, dark sunglasses) and identical twins are the main failure modes.");
                ui.label("- Use 'Deep region analysis' to see WHICH facial area drives the score.");
            });
        });
    }

    fn webcam_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            if ui.button("Scan cameras").clicked() {
                self.cam_list = crate::cam::list().unwrap_or_default();
            }
            egui::ComboBox::new("camidx", "camera")
                .selected_text(format!("{}", self.cam_idx))
                .show_ui(ui, |ui| {
                    for (i, n) in self.cam_list.iter().enumerate() {
                        if ui.selectable_label(self.cam_idx == i as u32, n).clicked() {
                            self.cam_idx = i as u32;
                        }
                    }
                });
            if self.cam.is_none() {
                if ui.button("Start").clicked() {
                    match crate::cam::start(self.cam_idx, 1280, 720) {
                        Ok(c) => self.cam = Some(c),
                        Err(e) => self.status = format!("camera error: {e}"),
                    }
                }
            } else if ui.button("Stop").clicked() {
                self.cam = None;
                self.cam_tex = None;
            }
            ui.separator();
            ui.label(format!("camera FPS ~{:.1}", self.cam_fps));
        });
        ui.separator();

        let max_h = self.preview_max_h;
        let mut load_ref: Option<Tag> = None;
        let mut want_deep = false;

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.columns(2, |cols| {
                // ---- left: live preview ----
                let ui = &mut cols[0];
                if let Some(t) = &self.cam_tex {
                    let tw = t.size()[0] as f32;
                    let th = t.size()[1] as f32;
                    let scale = (ui.available_width().min(680.0) / tw).min(max_h / th);
                    ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(tw * scale, th * scale)));
                } else {
                    ui.label("Camera is not running.");
                }

                if let Some(an) = &self.live.analysis {
                    if let Some(e) = an.entries.first() {
                        ui.label(format!("inference {:.0} ms - {} face(s)", an.ms, an.entries.len()));
                        ui.horizontal(|ui| {
                            if let Some(t) = self.live.thumbs.first() {
                                ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(80.0, 80.0)));
                            }
                            ui.vertical(|ui| quality_ui(ui, &e.quality));
                        });
                    } else {
                        ui.colored_label(Color32::YELLOW, "no face in frame");
                    }
                }

                // ---- right: references & matching ----
                let ui = &mut cols[1];
                ui.heading("Reference A / B");
                ui.label(
                    RichText::new("Load a photo as the reference, then let the webcam recognise against it.")
                        .weak(),
                );

                ui.horizontal(|ui| {
                    if ui.button("Load photo -> A").clicked() { load_ref = Some(Tag::RefA); }
                    if ui.button("Load photo -> B").clicked() { load_ref = Some(Tag::RefB); }
                });

                let busy = self.enroll.is_some() || self.cam.is_none();
                ui.horizontal(|ui| {
                    if ui.add_enabled(!busy, egui::Button::new("Record A from webcam")).clicked() {
                        self.enroll = Some((Tag::A, Vec::new()));
                    }
                    if ui.add_enabled(!busy, egui::Button::new("Record B from webcam")).clicked() {
                        self.enroll = Some((Tag::B, Vec::new()));
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Clear A").clicked() { self.ref_a = None; self.live_hist.clear(); }
                    if ui.button("Clear B").clicked() { self.ref_b = None; self.live_hist.clear(); }
                });

                if let Some((t, buf)) = &self.enroll {
                    ui.label(format!(
                        "Recording {t:?}: {}/{} - move your head slightly, vary the expression",
                        buf.len(), ENROLL_N
                    ));
                    ui.add(egui::ProgressBar::new(buf.len() as f32 / ENROLL_N as f32));
                }

                for (label, r) in [("A", &self.ref_a), ("B", &self.ref_b)] {
                    ui.separator();
                    match r {
                        Some(r) => {
                            ui.horizontal(|ui| {
                                if let Some(t) = &r.thumb {
                                    ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(64.0, 64.0)));
                                }
                                ui.vertical(|ui| {
                                    ui.label(RichText::new(format!("Ref {label}")).strong());
                                    ui.label(RichText::new(&r.source).small().weak());
                                    ui.label(
                                        RichText::new(format!(
                                            "{} frame(s) - face {:.0}px - blur {:.0}",
                                            r.frames, r.entry.quality.size, r.entry.quality.blur
                                        ))
                                        .small(),
                                    );
                                });
                            });
                        }
                        None => {
                            ui.label(RichText::new(format!("Ref {label}: not set")).weak());
                        }
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Live match against:");
                    ui.selectable_value(&mut self.live_match, None, "off");
                    ui.selectable_value(&mut self.live_match, Some(Tag::A), "A");
                    ui.selectable_value(&mut self.live_match, Some(Tag::B), "B");
                });

                if !self.live_hist.is_empty() {
                    let mean = self.live_hist.iter().sum::<f32>() / self.live_hist.len() as f32;
                    let sd = (self.live_hist.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                        / self.live_hist.len() as f32).sqrt();
                    ui.label(format!(
                        "rolling cosine over {} frames: {mean:.4}  +/-{sd:.4}",
                        self.live_hist.len()
                    ));
                    let p = analyze::probability(mean, &self.cal);
                    ui.add(egui::ProgressBar::new(p).text(format!("{:.1}% (rolling)", p * 100.0)));
                    let v = analyze::verdict(
                        &vec![mean], &vec![1.0],
                        &Calibration { slope: 1.0, mid: 0.0, threshold: self.cal.threshold },
                    );
                    let _ = v;
                    ui.label(
                        RichText::new(
                            if mean >= self.cal.threshold { "above threshold -> SAME" }
                            else if mean >= self.cal.threshold - 0.08 { "INCONCLUSIVE band" }
                            else { "below threshold -> DIFFERENT" },
                        )
                        .strong(),
                    );
                }

                ui.separator();
                if let (Some(a), Some(b)) = (&self.ref_a, &self.ref_b) {
                    self.verdict_ui(ui, &a.entry.emb, &b.entry.emb, "Ref A", "Ref B");
                    want_deep = ui.button("Deep region analysis of Ref A <-> Ref B").clicked();
                }

                ui.colored_label(
                    Color32::from_rgb(255, 170, 80),
                    "No liveness detection: a photo or video held up to the camera will pass.",
                );
            });
        });

        if let Some(t) = load_ref {
            if let Some(p) = rfd::FileDialog::new()
                .add_filter("Images", &["jpg", "jpeg", "png", "bmp", "webp"])
                .pick_file()
            {
                match image::open(&p) {
                    Ok(d) => {
                        let src = p.display().to_string();
                        if t == Tag::RefA { self.ref_src_a = src; } else { self.ref_src_b = src; }
                        let _ = self.tx.send(Job::Analyze {
                            tag: t, seq: 0, img: d.to_rgb8(), fast: false,
                        });
                        self.status = "analysing reference photo...".into();
                    }
                    Err(e) => self.status = format!("failed to open: {e}"),
                }
            }
        }

        if want_deep {
            if let (Some(a), Some(b)) = (&self.ref_a, &self.ref_b) {
                let ea = a.entry.clone();
                let eb = b.entry.clone();
                self.deep_pending = true;
                self.deep = None;
                self.deep_tex = None;
                self.deep_source = "Webcam Ref A <-> Ref B".into();
                self.tab = Tab::Analysis;
                let _ = self.tx.send(Job::DeepAnalyze { a: ea, b: eb });
            }
        }

        let _ = ctx;
    }

    fn analysis_tab(&mut self, ui: &mut egui::Ui) {
        let mut job: Option<(Entry, Entry, String)> = None;

        ui.horizontal(|ui| {
            let can_photo = self.a.analysis.is_some() && self.b.analysis.is_some();
            let can_ref = self.ref_a.is_some() && self.ref_b.is_some();

            if ui
                .add_enabled(can_photo && !self.deep_pending, egui::Button::new("Analyse Photo A <-> Photo B"))
                .clicked()
            {
                if let (Some(x), Some(y)) = (self.photo_entry(Tag::A), self.photo_entry(Tag::B)) {
                    job = Some((x, y, "Photo A <-> Photo B".into()));
                }
            }
            if ui
                .add_enabled(can_ref && !self.deep_pending, egui::Button::new("Analyse Webcam Ref A <-> Ref B"))
                .clicked()
            {
                if let (Some(a), Some(b)) = (&self.ref_a, &self.ref_b) {
                    job = Some((a.entry.clone(), b.entry.clone(), "Webcam Ref A <-> Ref B".into()));
                }
            }
            if self.deep_pending {
                ui.spinner();
                ui.label("running occlusion analysis (22 model runs)...");
            }
        });

        if let Some((a, b, label)) = job {
            self.deep_pending = true;
            self.deep = None;
            self.deep_tex = None;
            self.deep_source = label;
            let _ = self.tx.send(Job::DeepAnalyze { a, b });
        }

        ui.separator();

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            egui::CollapsingHeader::new("READ THIS FIRST - what this analysis can and cannot prove")
                .default_open(true)
                .show(ui, |ui| {
                    ui.colored_label(
                        Color32::from_rgb(255, 200, 90),
                        "ArcFace outputs ONE global 512-d vector. There is no built-in 'eye similarity' output.",
                    );
                    ui.label("Everything below is DERIVED by re-running the model on modified crops:");
                    ui.label("1) OCCLUSION DELTA (strongest method): the same region is hidden in BOTH faces with neutral grey, then the model runs again.");
                    ui.label("     delta = baseline_cosine - cosine_with_region_hidden");
                    ui.label("     delta > 0  ->  that region was EVIDENCE FOR 'same person' (hiding it lowers the score).");
                    ui.label("     delta < 0  ->  that region was PUSHING TOWARD 'different' (hiding it raises the score).");
                    ui.label("     A negative mouth delta usually means expression / moustache / beard, NOT a different identity.");
                    ui.label("2) REGION-ONLY cosine (weak): only that region stays visible. The model was trained on FULL faces, so these are out-of-distribution.");
                    ui.label("     Use it to RANK regions. NEVER compare it to the full-face threshold.");
                    ui.label("3) GEOMETRY (weak for identity): 5-landmark ratios. Pose and expression dominate these numbers.");
                    ui.label("4) HEATMAP (visual only): pixel difference is dominated by lighting, pose and expression. It is NOT identity evidence.");
                    ui.label("Limitations: masking itself shifts the input distribution, regions overlap, and deltas do not sum to the total score.");
                });

            if let (Some(rep), Some(texs)) = (&self.deep, &self.deep_tex) {
                ui.separator();
                ui.label(RichText::new(format!("Source: {}", self.deep_source)).weak());
                ui.label(format!(
            "headline cosine {:.4}   |   no-TTA baseline {:.4}   |   {} model runs in {:.0} ms",
            rep.cos_headline, rep.cos_base, rep.forwards, rep.ms
        ));
                ui.label(
                    RichText::new("All deltas are measured against the no-TTA baseline so the comparison is apples-to-apples.")
                        .small()
                        .weak(),
                );

                ui.separator();
                ui.heading("Summary");
                if let Some(r) = rep.strongest_for() {
                    ui.colored_label(
                        Color32::from_rgb(80, 225, 130),
                        format!("Strongest evidence FOR same person: {}  (delta {:+.4})", r.name, r.delta),
                    );
                } else {
                    ui.label("No region produced positive evidence for 'same person'.");
                }
                if let Some(r) = rep.strongest_against() {
                    ui.colored_label(
                        Color32::from_rgb(240, 110, 90),
                        format!("Strongest push TOWARD different: {}  (delta {:+.4})", r.name, r.delta),
                    );
                } else {
                    ui.label("No region pushed the score toward 'different'.");
                }

                ui.separator();
                ui.heading("Per-region occlusion analysis");
                egui::Grid::new("az_region_grid").num_columns(5).striped(true).show(ui, |ui| {
                    ui.strong("Facial region");
                    ui.strong("cos if hidden");
                    ui.strong("delta");
                    ui.strong("cos region only");
                    ui.strong("reading");
                    ui.end_row();

                    for r in &rep.regions {
                        ui.label(r.name).on_hover_text(r.note);
                        ui.label(format!("{:.4}", r.cos_occluded));
                        ui.colored_label(delta_color(r.delta), format!("{:+.4}", r.delta));
                        ui.label(
                            RichText::new(format!("{:.4}", r.cos_region_only)).weak(),
                        );
                        ui.label(regions::interpret(r.delta));
                        ui.end_row();
                    }
                });
                ui.label(
                    RichText::new("Hover a region name for what usually affects it. 'cos region only' is out-of-distribution - ranking only.")
                        .small()
                        .weak(),
                );

                ui.separator();
                ui.heading("Aligned crops and difference heatmap");
                ui.horizontal(|ui| {
                    for (t, cap) in [
                        (&texs.0, "A aligned 112x112"),
                        (&texs.1, "B aligned 112x112"),
                        (&texs.2, "abs difference (visual only)"),
                    ] {
                        ui.vertical(|ui| {
                            ui.add(egui::Image::new(t).fit_to_exact_size(egui::vec2(150.0, 150.0)));
                            ui.label(RichText::new(cap).small().weak());
                        });
                    }
                });
                ui.label(
                    RichText::new("Blue = similar pixels, red = very different pixels. Lighting and pose dominate this map - do NOT read identity from it.")
                        .small()
                        .weak(),
                );

                ui.separator();
                ui.heading("Landmark geometry (coarse, pose/expression sensitive)");
                egui::Grid::new("az_geo_grid").num_columns(4).striped(true).show(ui, |ui| {
                    ui.strong("metric");
                    ui.strong("A");
                    ui.strong("B");
                    ui.strong("difference");
                    ui.end_row();

                    for i in 0..GEO_LABELS.len() {
                        let (name, note) = GEO_LABELS[i];
                        let (x, y) = (rep.geo_a[i], rep.geo_b[i]);
                        ui.label(name).on_hover_text(note);
                        ui.label(format!("{:.3}", x));
                        ui.label(format!("{:.3}", y));
                        if i == 0 {
                            ui.label(format!("{:+.1} px", y - x));
                        } else if i == 7 {
                            ui.label(format!("{:+.1} deg", y - x));
                        } else if x.abs() > 1e-4 {
                            ui.label(format!("{:+.1} %", (y - x) / x.abs() * 100.0));
                        } else {
                            ui.label(format!("{:+.3}", y - x));
                        }
                        ui.end_row();
                    }
                });
                ui.label(
                    RichText::new("Only 5 landmarks are available, so this is NOT a morphometric measurement. Large differences here usually mean pose/expression, not a different person.")
                        .small()
                        .weak(),
                );
            } else if !self.deep_pending {
                ui.label("Run one of the analyses above to see the per-region breakdown.");
            }
        });
    }

    fn settings_tab(&mut self, ui: &mut egui::Ui) {
        let before = self.cfg;

        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.heading("Inference engine");
            egui::ComboBox::new("ep", "Execution provider")
                .selected_text(self.cfg.provider.label())
                .show_ui(ui, |ui| {
                    for p in [Provider::Cuda, Provider::DirectMl, Provider::Cpu] {
                        ui.selectable_value(&mut self.cfg.provider, p, p.label());
                    }
                });
            ui.label(
                RichText::new("This is the REQUESTED provider. If CUDA/cuDNN is missing, ONNX Runtime silently falls back to CPU - verify with Task Manager -> GPU.")
                    .small()
                    .weak(),
            );

            egui::ComboBox::new("ds", "Detector input size")
                .selected_text(format!("{}", self.cfg.det_size))
                .show_ui(ui, |ui| {
                    for s in [320usize, 480, 640] {
                        ui.selectable_value(&mut self.cfg.det_size, s, format!("{s}"));
                    }
                });

            ui.add(egui::Slider::new(&mut self.cfg.threads, 1..=16).text("CPU threads"));
            ui.add(egui::Slider::new(&mut self.cfg.max_faces, 1..=8).text("max faces processed"));
            ui.checkbox(&mut self.cfg.flip_tta, "flip-TTA (2x slower, more accurate)");
            ui.add(egui::Slider::new(&mut self.cfg.det_thresh, 0.1..=0.9).text("detection threshold"));
            ui.add(egui::Slider::new(&mut self.cfg.nms, 0.2..=0.7).text("NMS IoU"));

            ui.separator();
            ui.heading("Layout");
            ui.add(egui::Slider::new(&mut self.preview_max_h, 200.0..=900.0).text("preview max height (px)"));

            ui.separator();
            ui.heading("Decision & calibration");
            ui.add(egui::Slider::new(&mut self.cal.threshold, 0.10..=0.80).text("cosine threshold"));
            ui.add(egui::Slider::new(&mut self.cal.slope, 4.0..=30.0).text("slope (a)"));
            ui.add(egui::Slider::new(&mut self.cal.mid, 0.10..=0.70).text("midpoint (b)"));

            ui.horizontal(|ui| {
                if ui.button("Reset defaults").clicked() { self.cal = Calibration::default(); }
                if ui.button("Load calibration.json").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
                        let loaded = std::fs::read_to_string(&p)
                            .map_err(anyhow::Error::from)
                            .and_then(|s| serde_json::from_str::<Calibration>(&s).map_err(anyhow::Error::from));
                        match loaded {
                            Ok(c) => { self.cal = c; self.status = "calibration loaded".into(); }
                            Err(e) => self.status = format!("calibration error: {e:#}"),
                        }
                    }
                }
            });

            ui.label(RichText::new("AzFaceVerify.exe calibrate pairs.csv --out calibration.json").monospace());
            ui.label(RichText::new("pairs.csv header: path_a,path_b,label   (1 = same person, 0 = different)").small().weak());

            ui.separator();
            ui.heading("Models & environment");
            ui.label(format!("Model folder: {}", self.models.display()));
            ui.label("Required files: det_10g.onnx and w600k_r50.onnx (InsightFace buffalo_l).");
            ui.label("CUDA needs CUDA 12.x + cuDNN 9.x on PATH. If that fails, use DirectML.");
            ui.label("InsightFace pretrained weights are licensed for non-commercial research.");

            ui.separator();
            ui.heading("Privacy & legal");
            ui.label("All processing is 100% local - the application makes no network calls.");
            ui.label("Face data is biometric / special-category personal data.");
            ui.label("Obtain consent, define a retention policy, never store embeddings unencrypted.");
            ui.label("Do not use a single score as the sole basis for a high-stakes decision.");

            ui.separator();
            ui.heading("About");
            ui.label(format!("AzFaceVerify v{} - built by Azis Ichwan", crate::APP_VERSION));
            ui.label("Rust + egui/eframe + ONNX Runtime (CUDA / DirectML / CPU).");
            ui.label("SCRFD detection -> 5-point alignment -> ArcFace R50 -> calibrated verdict.");
            ui.label("Deep analysis = occlusion attribution over 5 facial regions.");
            ui.label(format!("Window icon: {}", self.icon_info));
            if let Some(last) = &self.last_report {
                ui.label(format!("Last report exported: {last}"));
            }
        });

        if self.cfg != before {
            self.ready = false;
            self.status = "reloading models...".into();
            let _ = self.tx.send(Job::Configure(self.cfg));
        }
    }

    fn export_txt(&mut self, a: &Entry, b: &Entry) {
        let (human, file_stamp, unix) = stamp();
        let v = analyze::verdict(&a.emb, &b.emb, &self.cal);
        let slug = verdict_slug(v.level);
        let path_a = self.a.path.clone();
        let path_b = self.b.path.clone();
        let default_name = format!("AzFaceVerify_{file_stamp}_{slug}_cos{:.3}.txt", v.cos);

        let mut region_text = String::from("(not run - use Deep region analysis)\n");
        if let Some(rep) = &self.deep {
            region_text.clear();
            region_text.push_str(&format!("baseline (no TTA): {:.6}\n", rep.cos_base));
            for r in &rep.regions {
                region_text.push_str(&format!(
                    "{:<22} hidden={:.6}  delta={:+.6}  region_only={:.6}  {}\n",
                    r.name, r.cos_occluded, r.delta, r.cos_region_only, regions::interpret(r.delta)
                ));
            }
        }

        let body = format!(
            "=========================================================\n\
             AzFaceVerify - face verification report\n\
             Built by Azis Ichwan\n\
             =========================================================\n\
             Report time (local) : {human}\n\
             Report time (unix)  : {unix}\n\
             Report id           : AZ-{file_stamp}\n\
             App version         : {version}\n\
             Execution provider  : {provider} (requested)\n\
             Models              : det_10g.onnx (SCRFD) + w600k_r50.onnx (ArcFace R50)\n\
             Detector input size : {det_size}\n\
             flip-TTA            : {tta}\n\
             \n\
             ------------------------- INPUT -------------------------\n\
             Photo A : {path_a}\n\
             Photo B : {path_b}\n\
             \n\
             ------------------------- RESULT ------------------------\n\
             Verdict            : {label}\n\
             Cosine similarity  : {cos:.6}\n\
             L2 distance        : {l2:.6}\n\
             Calibrated prob.   : {prob:.4}  ({prob_pct:.1}%)\n\
             Decision threshold : {thr:.4}\n\
             Calibration        : slope={slope:.3} mid={mid:.4}\n\
             \n\
             --------------------- REGION ANALYSIS -------------------\n\
             {region_text}\
             \n\
             ------------------------- FACE A ------------------------\n\
             bbox   : [{ax1:.1}, {ay1:.1}, {ax2:.1}, {ay2:.1}]\n\
             quality: {qa}\n\
             warnings: {wa:?}\n\
             \n\
             ------------------------- FACE B ------------------------\n\
             bbox   : [{bx1:.1}, {by1:.1}, {bx2:.1}, {by2:.1}]\n\
             quality: {qb}\n\
             warnings: {wb:?}\n\
             \n\
             ------------------------- NOTES -------------------------\n\
             - Cosine similarity is the raw model score, NOT a probability.\n\
             - Probability comes from a logistic calibration; calibrate on your own data.\n\
             - Region deltas are occlusion attribution, not a per-organ similarity readout.\n\
             - Region-only cosines are out-of-distribution; use for ranking only.\n\
             - Heavy occlusion and identical twins are known failure modes.\n\
             - No liveness detection was performed.\n\
             - All processing was local; no image left this machine.\n",
            version = crate::APP_VERSION,
            provider = self.cfg.provider.label(),
            det_size = self.cfg.det_size,
            tta = if self.cfg.flip_tta { "ON" } else { "OFF" },
            label = v.label, cos = v.cos, l2 = v.l2, prob = v.prob, prob_pct = v.prob * 100.0,
            thr = self.cal.threshold, slope = self.cal.slope, mid = self.cal.mid,
            ax1 = a.face.bbox[0], ay1 = a.face.bbox[1], ax2 = a.face.bbox[2], ay2 = a.face.bbox[3],
            bx1 = b.face.bbox[0], by1 = b.face.bbox[1], bx2 = b.face.bbox[2], by2 = b.face.bbox[3],
            qa = quality_line(a), qb = quality_line(b),
            wa = a.quality.warnings, wb = b.quality.warnings,
        );

        if let Some(p) = rfd::FileDialog::new()
            .set_file_name(&default_name)
            .add_filter("Text report", &["txt"])
            .save_file()
        {
            match std::fs::write(&p, body) {
                Ok(_) => {
                    self.last_report = Some(format!("{human} (txt)"));
                    self.status = format!("report saved: {}", p.display());
                }
                Err(e) => self.status = format!("failed to save report: {e}"),
            }
        }
    }

    fn export_json(&mut self, a: &Entry, b: &Entry) {
        let (human, file_stamp, unix) = stamp();
        let v = analyze::verdict(&a.emb, &b.emb, &self.cal);
        let slug = verdict_slug(v.level);
        let default_name = format!("AzFaceVerify_{file_stamp}_{slug}_cos{:.3}.json", v.cos);

        let face_json = |e: &Entry, path: &String| {
            serde_json::json!({
                "source": path,
                "bbox": e.face.bbox,
                "landmarks": e.face.kps,
                "det_score": e.face.score,
                "quality": {
                    "size_px": e.quality.size, "det": e.quality.det, "blur": e.quality.blur,
                    "brightness": e.quality.bright, "contrast": e.quality.contrast,
                    "yaw": e.quality.yaw, "roll_deg": e.quality.roll, "pitch": e.quality.pitch,
                    "warnings": e.quality.warnings,
                },
                "embedding_512": e.emb,
            })
        };

        let region_json = self.deep.as_ref().map(|rep| {
            serde_json::json!({
                "method": "occlusion attribution (region hidden in both faces, neutral grey fill)",
                "baseline_cosine_no_tta": rep.cos_base,
                "model_runs": rep.forwards,
                "regions": rep.regions.iter().map(|r| serde_json::json!({
                    "name": r.name,
                    "cos_if_hidden": r.cos_occluded,
                    "delta": r.delta,
                    "cos_region_only_out_of_distribution": r.cos_region_only,
                    "reading": regions::interpret(r.delta),
                })).collect::<Vec<_>>(),
                "geometry_a": rep.geo_a,
                "geometry_b": rep.geo_b,
                "geometry_labels": GEO_LABELS.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            })
        });

        let doc = serde_json::json!({
            "schema": "azfaceverify.report/2",
            "app": { "name": crate::APP_NAME, "version": crate::APP_VERSION, "author": crate::APP_AUTHOR },
            "timestamp": { "local": human, "unix": unix, "report_id": format!("AZ-{file_stamp}") },
            "engine": {
                "provider_requested": self.cfg.provider.label(),
                "detector": "det_10g.onnx",
                "recognizer": "w600k_r50.onnx",
                "detector_input_size": self.cfg.det_size,
                "flip_tta": self.cfg.flip_tta,
            },
            "result": {
                "verdict": v.label, "slug": slug, "cosine": v.cos, "l2": v.l2,
                "probability": v.prob, "threshold": self.cal.threshold,
                "calibration": { "slope": self.cal.slope, "mid": self.cal.mid },
            },
            "region_analysis": region_json,
            "face_a": face_json(a, &self.a.path),
            "face_b": face_json(b, &self.b.path),
            "notes": [
                "Cosine similarity is the raw model score, not a probability.",
                "Region deltas are occlusion attribution, not a per-organ similarity readout.",
                "Region-only cosines are out-of-distribution; ranking only.",
                "Heavy occlusion and identical twins are known failure modes.",
                "No liveness detection was performed.",
                "All processing was local."
            ]
        });

        if let Some(p) = rfd::FileDialog::new()
            .set_file_name(&default_name)
            .add_filter("JSON report", &["json"])
            .save_file()
        {
            match serde_json::to_string_pretty(&doc)
                .map_err(anyhow::Error::from)
                .and_then(|s| std::fs::write(&p, s).map_err(anyhow::Error::from))
            {
                Ok(_) => {
                    self.last_report = Some(format!("{human} (json)"));
                    self.status = format!("report saved: {}", p.display());
                }
                Err(e) => self.status = format!("failed to save report: {e:#}"),
            }
        }
    }
}

fn quality_ui(ui: &mut egui::Ui, q: &Quality) {
    ui.label(format!(
        "size {:.0}px - det {:.2} - blur {:.0} - bright {:.0} - contrast {:.0}",
        q.size, q.det, q.blur, q.bright, q.contrast
    ));
    ui.label(format!(
        "pose: yaw {:+.2} - roll {:+.0} deg - pitch {:.2}",
        q.yaw, q.roll, q.pitch
    ));
    for w in &q.warnings {
        ui.colored_label(Color32::from_rgb(255, 190, 80), format!("- {w}"));
    }
    if q.warnings.is_empty() {
        ui.colored_label(Color32::from_rgb(120, 220, 140), "quality OK");
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _f: &mut eframe::Frame) {
        self.pump(ctx);
        self.handle_drops(ctx);
        self.pump_camera(ctx);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("AzFaceVerify");
                ui.label(RichText::new("built by Azis Ichwan").small().weak());
                ui.separator();
                ui.selectable_value(&mut self.tab, Tab::Photo, "Compare 2 Photos");
                ui.selectable_value(&mut self.tab, Tab::Webcam, "Webcam");
                ui.selectable_value(&mut self.tab, Tab::Analysis, "Deep Analysis");
                ui.selectable_value(&mut self.tab, Tab::Settings, "Settings");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(&self.status).small());
                });
            });
        });

        if self.tab == Tab::Photo {
            egui::TopBottomPanel::bottom("result_panel")
                .resizable(true)
                .default_height(240.0)
                .min_height(110.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| self.result_panel(ui));
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Photo => self.photo_tab(ui),
            Tab::Webcam => self.webcam_tab(ui, ctx),
            Tab::Analysis => self.analysis_tab(ui),
            Tab::Settings => self.settings_tab(ui),
        });

        if self.a.pending || self.b.pending || self.live_pending || self.deep_pending {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }
}