//! swp-drive: reference coordinator (star topology) for SWP/1 stage services.
//!
//!   swp-drive --selftest
//!   swp-drive --stages host:port,host:port --bounds 4 --prompt 1,2,3,4,5 --n 12 [--chunk 4] [--secret-env VAR]
//!
//! `--bounds` lists the interior layer boundaries (here: stage0 = layers 0..4, stage1 = 4..LAYERS).
//! Talks to `swp-mockstage` (or any adapter implementing the mock model kind) and can also be used
//! to exercise real adapters once they implement the same sequencing: configure -> prefill chunks
//! -> decode -> truncate/rewrite -> close.
use serde_json::{json, Value};
use splitwire::frame::{read_frame, write_frame};
use splitwire::mock::{serve_conn, StageState, HIDDEN, LAYERS, VOCAB};
use splitwire::msg::*;
use splitwire::util::secret_from_env;
use std::io;
use std::net::{TcpListener, TcpStream};

struct Chain {
    conns: Vec<TcpStream>,
    session: String,
    seq: u32,
    step: u64,
}

impl Chain {
    fn connect(addrs: &[String], bounds: &[usize], secret: Option<&str>) -> io::Result<Chain> {
        let n = addrs.len();
        assert_eq!(bounds.len() + 1, n, "need stages-1 boundaries");
        let mut edges = vec![0];
        edges.extend_from_slice(bounds);
        edges.push(LAYERS);
        let mut conns = Vec::new();
        let mut seq = 10;
        for (i, a) in addrs.iter().enumerate() {
            let mut s = TcpStream::connect(a)?;
            s.set_nodelay(true)?;
            client_handshake(&mut s, "drive", &["coordinator"], secret)?;
            seq += 1;
            write_frame(
                &mut s,
                kind::STAGE_CONFIGURE,
                seq,
                &json!({"session": "s1", "layers": [edges[i], edges[i + 1]], "has_embed": i == 0, "has_head": i == n - 1,
                        "wire_dtype": "f32", "model": {"kind": "mock", "hidden": HIDDEN, "vocab": VOCAB}}),
                &[],
                false,
            )?;
            expect(read_frame(&mut s)?.ok_or_else(|| io::Error::other("closed"))?, kind::STAGE_READY)?;
            conns.push(s);
        }
        Ok(Chain { conns, session: "s1".into(), seq, step: 0 })
    }

    /// Relay one forward through every stage. Returns head-stage tokens (empty for want="none").
    fn forward(&mut self, start_pos: usize, toks: &[u32], want: &str, rows: &str, rewrite: bool) -> io::Result<Vec<u32>> {
        self.step += 1;
        let n = self.conns.len();
        let mut hidden: Vec<u8> = vec![];
        let mut out = vec![];
        for i in 0..n {
            self.seq += 1;
            let last = i == n - 1;
            let h = json!({"session": self.session, "step": self.step, "mode": if toks.len() > 1 { "prefill" } else { "decode" },
                           "start_pos": start_pos, "n_rows": toks.len(), "token_ids": toks,
                           "input": if i == 0 { "tokens" } else { "hidden" }, "dtype": "f32", "shape": [toks.len(), HIDDEN],
                           "want": if last { want } else { "hidden" }, "return_rows": rows, "rewrite": rewrite});
            write_frame(&mut self.conns[i], kind::FORWARD, self.seq, &h, &hidden, false)?;
            let f = expect(read_frame(&mut self.conns[i])?.ok_or_else(|| io::Error::other("closed"))?, kind::FORWARD_OUT)?;
            if last {
                out = f.payload.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
            } else {
                hidden = f.payload;
            }
        }
        Ok(out)
    }

    fn truncate(&mut self, pos: usize) -> io::Result<()> {
        for c in &mut self.conns {
            self.seq += 1;
            write_frame(c, kind::CACHE_OP, self.seq, &json!({"session": self.session, "op": "truncate", "pos": pos}), &[], false)?;
            expect(read_frame(c)?.ok_or_else(|| io::Error::other("closed"))?, kind::CACHE_OK)?;
        }
        Ok(())
    }

    /// Prefill `prompt` in chunks, then greedy-decode `n_new` tokens.
    fn generate(&mut self, prompt: &[u32], n_new: usize, chunk: usize) -> io::Result<Vec<u32>> {
        let mut pos = 0;
        let mut first = vec![];
        for (ci, c) in prompt.chunks(chunk).enumerate() {
            let is_last = (ci + 1) * chunk >= prompt.len();
            first = self.forward(pos, c, if is_last { "token" } else { "none" }, "last", false)?;
            pos += c.len();
        }
        let mut out = vec![first[0]];
        while out.len() < n_new {
            let t = self.forward(pos, &[*out.last().unwrap()], "token", "last", false)?;
            pos += 1;
            out.push(t[0]);
        }
        Ok(out)
    }
}

/// Unsplit reference built from the same toy-model code.
struct Reference(StageState);
impl Reference {
    fn new() -> Self {
        Reference(StageState::new(0, LAYERS, true, true, HIDDEN, VOCAB))
    }
    fn forward(&mut self, start: usize, toks: &[u32], rewrite: bool) -> u32 {
        if rewrite {
            self.0.truncate(start);
        }
        let rows = self.0.embed(toks, start);
        let out = self.0.run_layers(start, rows);
        self.0.head_token(out.last().unwrap())
    }
    fn generate(&mut self, prompt: &[u32], n_new: usize, chunk: usize) -> Vec<u32> {
        let mut pos = 0;
        let mut first = 0;
        for c in prompt.chunks(chunk) {
            first = self.forward(pos, c, false);
            pos += c.len();
        }
        let mut out = vec![first];
        while out.len() < n_new {
            let t = self.forward(pos, &[*out.last().unwrap()], false);
            pos += 1;
            out.push(t);
        }
        out
    }
}

fn spawn_stage(secret: Option<String>) -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for c in l.incoming().flatten() {
            let s = secret.clone();
            std::thread::spawn(move || {
                let _ = serve_conn(c, "mock", s.as_deref());
            });
        }
    });
    addr
}

macro_rules! check {
    ($name:expr, $cond:expr) => {{
        let ok = $cond;
        println!("  [{}] {}", if ok { "PASS" } else { "FAIL" }, $name);
        if !ok {
            std::process::exit(1);
        }
    }};
}

fn selftest() -> io::Result<()> {
    let prompt: Vec<u32> = vec![5, 17, 3, 42, 9, 60, 21, 8, 33, 12, 1];
    let reference = Reference::new().generate(&prompt, 10, 4);
    println!("reference tokens: {reference:?}");
    let distinct = reference.iter().collect::<std::collections::BTreeSet<_>>().len();
    check!("toy model is discriminating (>= 5 distinct output tokens)", distinct >= 5);

    for (name, bounds) in [("1 stage", vec![]), ("2 stages, split 4", vec![4]), ("2 stages, split 1", vec![1]), ("3 stages, split 2/5", vec![2, 5]), ("4 stages", vec![1, 3, 7])] {
        let addrs: Vec<String> = (0..bounds.len() + 1).map(|_| spawn_stage(None)).collect();
        let mut ch = Chain::connect(&addrs, &bounds, None)?;
        check!(format!("split == unsplit: {name}"), ch.generate(&prompt, 10, 4)? == reference);
        let mut ch2 = Chain::connect(&addrs, &bounds, None)?;
        check!(format!("chunking-invariant (chunk=3 vs 4): {name}"), ch2.generate(&prompt, 10, 3)? == reference);
    }

    // Rollback: accept 2 drafted tokens, reject the rest, continue with different tokens.
    let addrs: Vec<String> = (0..3).map(|_| spawn_stage(None)).collect();
    let mut ch = Chain::connect(&addrs, &[2, 5], None)?;
    ch.generate(&prompt, 6, 4)?; // cache now holds prompt + 5 fed tokens
    let keep = prompt.len() + 2;
    let alt = [7u32, 11, 13];
    ch.truncate(keep)?;
    let got_explicit = ch.forward(keep, &alt, "token", "all", false)?;

    let mut rf = Reference::new();
    rf.generate(&prompt, 6, 4);
    let rows = {
        rf.0.truncate(keep);
        let r = rf.0.embed(&alt, keep);
        let o = rf.0.run_layers(keep, r);
        o.iter().map(|x| rf.0.head_token(x)).collect::<Vec<_>>()
    };
    check!("truncate + replay with alternative tokens matches reference", got_explicit == rows);
    // Sensitivity: continuing WITHOUT rolling back must give a different answer, else the test proves nothing.
    let mut stale = Reference::new();
    stale.generate(&prompt, 6, 4);
    let stale_pos = stale.0.len;
    let stale_rows = {
        let r = stale.0.embed(&alt, stale_pos);
        let o = stale.0.run_layers(stale_pos, r);
        o.iter().map(|x| stale.0.head_token(x)).collect::<Vec<_>>()
    };
    check!("test is sensitive to cache history (no-rollback differs)", stale_rows != rows);

    // Same rollback via rewrite=true (saves one round trip in speculative decoding).
    let mut ch = Chain::connect(&addrs, &[2, 5], None)?;
    ch.generate(&prompt, 6, 4)?;
    let got_rewrite = ch.forward(keep, &alt, "token", "all", true)?;
    check!("rewrite=true equals explicit truncate", got_rewrite == rows);

    // Desync detection: stale start_pos without rewrite must be rejected.
    let err = ch.forward(3, &[1], "token", "last", false);
    check!("stale start_pos rejected with pos_mismatch", err.is_err() && err.unwrap_err().to_string().contains("pos_mismatch"));

    // Auth.
    let a = spawn_stage(Some("s3cret".into()));
    check!("auth: correct secret accepted", Chain::connect(&[a.clone()], &[], Some("s3cret")).is_ok());
    check!("auth: wrong secret rejected", Chain::connect(&[a.clone()], &[], Some("nope")).is_err());
    check!("auth: missing secret rejected", Chain::connect(&[a], &[], None).is_err());

    println!("ALL CONFORMANCE CHECKS PASSED");
    Ok(())
}

fn main() {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| v.iter().position(|x| x == k).and_then(|i| v.get(i + 1)).cloned();
    if v.iter().any(|x| x == "--selftest") {
        if let Err(e) = selftest() {
            eprintln!("selftest error: {e}");
            std::process::exit(1);
        }
        return;
    }
    let addrs: Vec<String> = get("--stages").expect("--stages host:port,...").split(',').map(String::from).collect();
    let bounds: Vec<usize> = get("--bounds").map(|b| b.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()).unwrap_or_default();
    let prompt: Vec<u32> = get("--prompt").unwrap_or_else(|| "1,2,3,4,5".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let n: usize = get("--n").map(|x| x.parse().unwrap()).unwrap_or(8);
    let chunk: usize = get("--chunk").map(|x| x.parse().unwrap()).unwrap_or(4);
    let secret = secret_from_env(get("--secret-env").as_deref());
    let t0 = std::time::Instant::now();
    let mut ch = Chain::connect(&addrs, &bounds, secret.as_deref()).unwrap_or_else(|e| {
        eprintln!("connect: {e}");
        std::process::exit(1)
    });
    let toks = ch.generate(&prompt, n, chunk).unwrap_or_else(|e| {
        eprintln!("generate: {e}");
        std::process::exit(1)
    });
    let v: Value = json!(toks);
    println!("{v}  ({:.1} ms)", t0.elapsed().as_secs_f64() * 1e3);
}
