//! Deterministic toy "transformer" stage implementing the SWP/1 stage service.
//!
//! The math is a toy (no real model) but uses only exactly-rounded IEEE ops (+ * / abs), so every
//! conforming implementation, in any language, must reproduce it BIT-FOR-BIT. Each layer keeps a
//! per-position cache, so wrong positions, missed truncations or replay bugs change the output.
//! This validates transport, sequencing, cache control and rollback — not engine numerics.
use crate::frame::{read_frame, write_frame};
use crate::msg::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io;
use std::net::TcpStream;

use std::sync::atomic::{AtomicU64, Ordering};
/// Simulated per-forward compute (microseconds): base + per_row * rows. Lets benchmarks be exercised without GPUs.
pub static SIM_BASE_US: AtomicU64 = AtomicU64::new(0);
pub static SIM_ROW_US: AtomicU64 = AtomicU64::new(0);

pub const HIDDEN: usize = 16;
pub const VOCAB: usize = 64;
pub const LAYERS: usize = 8;

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
/// Embedding value in [-1, 1): 24 hash bits / 2^23 - 1 (exact in f32).
pub fn emb(token: u32, d: usize) -> f32 {
    let h = splitmix(((token as u64) << 32) | d as u64);
    ((h >> 40) as f32) / 8388608.0 - 1.0
}

#[derive(Clone)]
pub struct StageState {
    pub l0: usize,
    pub l1: usize,
    pub has_embed: bool,
    pub has_head: bool,
    pub hidden: usize,
    pub vocab: usize,
    cache: Vec<Vec<Vec<f32>>>,
    pub len: usize,
}

impl StageState {
    pub fn new(l0: usize, l1: usize, has_embed: bool, has_head: bool, hidden: usize, vocab: usize) -> Self {
        Self { l0, l1, has_embed, has_head, hidden, vocab, cache: vec![Vec::new(); l1 - l0], len: 0 }
    }
    pub fn truncate(&mut self, pos: usize) {
        for c in &mut self.cache {
            c.truncate(pos);
        }
        self.len = self.len.min(pos);
    }
    /// Token embedding plus an absolute-position embedding (so positions matter, as RoPE does in real models).
    pub fn embed(&self, tokens: &[u32], start_pos: usize) -> Vec<Vec<f32>> {
        tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| (0..self.hidden).map(|d| emb(t, d) + 0.5 * emb(0x10_0000 + (start_pos + i) as u32, d)).collect())
            .collect()
    }
    pub fn run_layers(&mut self, start_pos: usize, mut rows: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        for li in 0..(self.l1 - self.l0) {
            let scale = 0.05 + 0.01 * ((self.l0 + li) as f32);
            for (i, row) in rows.iter_mut().enumerate() {
                let p = start_pos + i;
                self.cache[li].push(row.clone());
                for d in 0..self.hidden {
                    let mut s = 0.0f32;
                    for j in 0..=p {
                        s += self.cache[li][j][d];
                    }
                    let m = s / ((p + 1) as f32);
                    row[d] += scale * (m / (1.0 + m.abs()));
                }
            }
        }
        self.len = start_pos + rows.len();
        rows
    }
    pub fn head_token(&self, row: &[f32]) -> u32 {
        let (mut best, mut bv) = (0u32, f32::NEG_INFINITY);
        for v in 0..self.vocab {
            let mut s = 0.0f32;
            for d in 0..self.hidden {
                s += row[d] * emb(4096 + v as u32, d); // untied output embedding
            }
            if s > bv {
                bv = s;
                best = v as u32;
            }
        }
        best
    }
}

fn bad(code: &str, msg: String) -> (String, String) {
    (code.to_string(), msg)
}

fn do_forward(st: &mut StageState, h: &Value, payload: &[u8]) -> Result<(Value, Vec<u8>), (String, String)> {
    let start = h["start_pos"].as_u64().ok_or_else(|| bad("bad_request", "start_pos".into()))? as usize;
    let n = h["n_rows"].as_u64().ok_or_else(|| bad("bad_request", "n_rows".into()))? as usize;
    let toks: Vec<u32> = h["token_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default();
    if toks.len() != n || n == 0 {
        return Err(bad("bad_request", "token_ids must have n_rows entries (every stage receives them)".into()));
    }
    if start != st.len {
        if h["rewrite"].as_bool() == Some(true) && start < st.len {
            st.truncate(start);
        } else {
            return Err(bad("pos_mismatch", format!("expected start_pos {} got {start}", st.len)));
        }
    }
    let t0 = std::time::Instant::now();
    let rows = if h["input"] == "tokens" {
        if !st.has_embed {
            return Err(bad("bad_request", "stage has no embedding".into()));
        }
        st.embed(&toks, start)
    } else {
        if payload.len() != n * st.hidden * 4 {
            return Err(bad("bad_request", format!("payload {} bytes, expected {}", payload.len(), n * st.hidden * 4)));
        }
        bytes_to_f32s(payload).chunks(st.hidden).map(|c| c.to_vec()).collect()
    };
    let out = st.run_layers(start, rows);
    let sim = SIM_BASE_US.load(Ordering::Relaxed) + SIM_ROW_US.load(Ordering::Relaxed) * n as u64;
    if sim > 0 {
        std::thread::sleep(std::time::Duration::from_micros(sim));
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    if st.has_head {
        let want = h["want"].as_str().unwrap_or("token");
        if want == "none" {
            return Ok((json!({"kind": "none", "stage_ms": ms}), vec![]));
        }
        let idx: Vec<usize> = if h["return_rows"] == "all" { (0..n).collect() } else { vec![n - 1] };
        let mut b = Vec::new();
        for &i in &idx {
            b.extend_from_slice(&(st.head_token(&out[i]) as i32).to_le_bytes());
        }
        Ok((json!({"kind": "tokens", "dtype": "i32", "shape": [idx.len()], "stage_ms": ms}), b))
    } else {
        let flat: Vec<f32> = out.into_iter().flatten().collect();
        Ok((json!({"kind": "hidden", "dtype": "f32", "shape": [n, st.hidden], "stage_ms": ms}), f32s_to_bytes(&flat)))
    }
}

pub fn serve_conn(mut s: TcpStream, node: &str, secret: Option<&str>) -> io::Result<()> {
    s.set_nodelay(true)?;
    server_handshake(&mut s, node, &["stage", "mock"], secret)?;
    let mut sessions: HashMap<String, StageState> = HashMap::new();
    while let Some(f) = read_frame(&mut s)? {
        let sid = f.header["session"].as_str().unwrap_or("").to_string();
        match f.kind {
            kind::PING => write_frame(&mut s, kind::PONG, f.seq, &f.header, &f.payload, false)?,
            kind::STAGE_CONFIGURE => {
                let l = f.header["layers"].as_array();
                let (Some(l), true) = (l, f.header["model"]["kind"] == "mock") else {
                    send_error(&mut s, f.seq, "bad_request", "mock stage needs layers and model.kind=mock")?;
                    continue;
                };
                let (a, b) = (l[0].as_u64().unwrap_or(0) as usize, l[1].as_u64().unwrap_or(0) as usize);
                let hidden = f.header["model"]["hidden"].as_u64().unwrap_or(HIDDEN as u64) as usize;
                let vocab = f.header["model"]["vocab"].as_u64().unwrap_or(VOCAB as u64) as usize;
                if b <= a {
                    send_error(&mut s, f.seq, "bad_request", "empty layer range")?;
                    continue;
                }
                sessions.insert(
                    sid.clone(),
                    StageState::new(a, b, f.header["has_embed"] == true, f.header["has_head"] == true, hidden, vocab),
                );
                write_frame(&mut s, kind::STAGE_READY, f.seq, &json!({"session": sid, "cache_len": 0}), &[], false)?;
            }
            kind::FORWARD => match sessions.get_mut(&sid) {
                None => send_error(&mut s, f.seq, "no_session", "configure the session first")?,
                Some(st) => match do_forward(st, &f.header, &f.payload) {
                    Ok((mut h, p)) => {
                        h["session"] = json!(sid);
                        h["step"] = f.header["step"].clone();
                        write_frame(&mut s, kind::FORWARD_OUT, f.seq, &h, &p, false)?
                    }
                    Err((c, m)) => send_error(&mut s, f.seq, &c, &m)?,
                },
            },
            kind::CACHE_OP => match sessions.get_mut(&sid) {
                None => send_error(&mut s, f.seq, "no_session", "configure the session first")?,
                Some(st) => {
                    match f.header["op"].as_str().unwrap_or("") {
                        "truncate" => st.truncate(f.header["pos"].as_u64().unwrap_or(0) as usize),
                        "clear" => st.truncate(0),
                        o => {
                            send_error(&mut s, f.seq, "unsupported", &format!("cache op {o}"))?;
                            continue;
                        }
                    }
                    write_frame(&mut s, kind::CACHE_OK, f.seq, &json!({"session": sid, "cache_len": st.len}), &[], false)?
                }
            },
            kind::SESSION_CLOSE => {
                sessions.remove(&sid);
                write_frame(&mut s, kind::SESSION_CLOSED, f.seq, &json!({"session": sid}), &[], false)?
            }
            k => send_error(&mut s, f.seq, "unsupported", &format!("kind {k:#x}"))?,
        }
    }
    Ok(())
}
