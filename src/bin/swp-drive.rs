//! swp-drive: reference coordinator (star topology) for SWP/1 stage services.
//!
//!   swp-drive --selftest
//!   swp-drive --bench --stages a:p,b:p --bounds 4 [--prompt-len 256] [--n 64] [--chunk 64] [--json out.json]
//!   swp-drive --probe --stages host:port[,host:port] [--hidden N] [--max-mib 64] [--secret-env VAR]
//!             [--assume-mbps M] [--decode-ms T] [--prefill-tok-s R]
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

/// One stage hop of one FORWARD: client-side round trip vs the stage's own reported compute time.
struct Hop {
    stage: usize,
    phase: u8, // 0 prefill, 1 decode
    rtt_ms: f64,
    stage_ms: f64,
}

struct Chain {
    conns: Vec<TcpStream>,
    session: String,
    seq: u32,
    step: u64,
    phase: u8,
    trace: Vec<Hop>,
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
        Ok(Chain { conns, session: "s1".into(), seq, step: 0, phase: 0, trace: vec![] })
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
            let t = std::time::Instant::now();
            write_frame(&mut self.conns[i], kind::FORWARD, self.seq, &h, &hidden, false)?;
            let f = expect(read_frame(&mut self.conns[i])?.ok_or_else(|| io::Error::other("closed"))?, kind::FORWARD_OUT)?;
            self.trace.push(Hop { stage: i, phase: self.phase, rtt_ms: t.elapsed().as_secs_f64() * 1e3, stage_ms: f.header["stage_ms"].as_f64().unwrap_or(0.0) });
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


// ------------------------------------------------------------------- benchmark

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

/// Time spent per stage ("machine") and tokens/s for prefill and decode.
/// compute  = stage_ms reported by the stage itself (it must sync its device before reading the clock)
/// wire+q   = client round trip minus that: serialization, network both ways, queueing, H2D/D2H not counted by the stage
/// coordinator = wall time not inside any hop (relay, sampling, bookkeeping)
fn bench(addrs: &[String], bounds: &[usize], secret: Option<&str>, prompt_len: usize, n_new: usize, chunk: usize, json_out: Option<String>) -> io::Result<()> {
    let mut ch = Chain::connect(addrs, bounds, secret)?;
    let prompt: Vec<u32> = (0..prompt_len).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect();
    let mut edges = vec![0];
    edges.extend_from_slice(bounds);
    edges.push(LAYERS);
    ch.phase = 0;
    let t0 = std::time::Instant::now();
    let (mut pos, mut first) = (0, vec![]);
    for (ci, c) in prompt.chunks(chunk).enumerate() {
        let last = (ci + 1) * chunk >= prompt.len();
        first = ch.forward(pos, c, if last { "token" } else { "none" }, "last", false)?;
        pos += c.len();
    }
    let pre_wall = t0.elapsed().as_secs_f64() * 1e3;
    ch.phase = 1;
    let (mut tok, mut lat) = (first[0], vec![]);
    let t1 = std::time::Instant::now();
    for _ in 1..n_new {
        let t = std::time::Instant::now();
        tok = ch.forward(pos, &[tok], "token", "last", false)?[0];
        pos += 1;
        lat.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let dec_wall = t1.elapsed().as_secs_f64() * 1e3;
    let n_dec = lat.len().max(1);

    let sums = |phase: u8, stage: usize| -> (f64, f64, usize) {
        ch.trace.iter().filter(|h| h.phase == phase && h.stage == stage).fold((0.0, 0.0, 0), |a, h| (a.0 + h.stage_ms, a.1 + (h.rtt_ms - h.stage_ms).max(0.0), a.2 + 1))
    };
    let ns = addrs.len();
    let mut js = json!({"prompt_tokens": prompt_len, "decode_tokens": n_dec, "stages": []});

    let pre_tps = prompt_len as f64 / (pre_wall / 1e3);
    println!("\nPREFILL  {prompt_len} tokens in {pre_wall:.1} ms (chunks of {chunk}) -> {pre_tps:.0} tok/s");
    println!("  stage  layers   compute ms   wire+queue ms   compute % of wall");
    let mut inside = 0.0;
    for i in 0..ns {
        let (c, o, _) = sums(0, i);
        inside += c + o;
        println!("  {i:>5}  {:>2}..{:<2}   {c:>10.1}   {o:>13.1}   {:>8.0}%", edges[i], edges[i + 1], c / pre_wall * 100.0);
    }
    println!("  coordinator (relay, local): {:.1} ms", (pre_wall - inside).max(0.0));

    let dec_tps = n_dec as f64 / (dec_wall / 1e3);
    let mean = dec_wall / n_dec as f64;
    println!("\nDECODE   {n_dec} tokens in {dec_wall:.1} ms -> {dec_tps:.1} tok/s   per token: mean {mean:.2} ms, p50 {:.2}, p95 {:.2}, max {:.2}", pct(&mut lat.clone(), 0.5), pct(&mut lat.clone(), 0.95), pct(&mut lat.clone(), 1.0));
    println!("  stage  layers   compute ms/tok   wire+queue ms/tok   busy (share of token time)");
    let mut inside = 0.0;
    let mut worst = (0usize, 0.0f64);
    for i in 0..ns {
        let (c, o, _) = sums(1, i);
        inside += c + o;
        if c > worst.1 {
            worst = (i, c);
        }
        println!("  {i:>5}  {:>2}..{:<2}   {:>14.3}   {:>17.3}   {:>8.0}%", edges[i], edges[i + 1], c / n_dec as f64, o / n_dec as f64, c / dec_wall * 100.0);
        js["stages"].as_array_mut().unwrap().push(json!({"stage": i, "layers": [edges[i], edges[i + 1]],
            "decode_compute_ms_per_tok": c / n_dec as f64, "decode_wire_ms_per_tok": o / n_dec as f64, "prefill_compute_ms": sums(0, i).0, "prefill_wire_ms": sums(0, i).1}));
    }
    println!("  coordinator (relay, sampling, local): {:.3} ms/tok", ((dec_wall - inside) / n_dec as f64).max(0.0));
    let wire: f64 = (0..ns).map(|i| sums(1, i).1).sum::<f64>();
    println!("\nserial decode: stages take turns, so busy shares add to <= 100% and each machine idles while the other works.");
    println!("slowest stage: {} ({:.0}% of token time); wire+queue is {:.0}% of token time.", worst.0, worst.1 / n_dec as f64 / mean * 100.0, wire / dec_wall * 100.0);
    println!("time to first token: {:.1} ms (prefill + first decode)", pre_wall + lat.first().copied().unwrap_or(0.0));
    js["prefill_tok_s"] = json!(pre_tps);
    js["decode_tok_s"] = json!(dec_tps);
    js["decode_ms_per_tok"] = json!(mean);
    if let Some(p) = json_out {
        std::fs::write(&p, serde_json::to_vec_pretty(&js).unwrap())?;
        println!("wrote {p}");
    }
    Ok(())
}

// ------------------------------------------------------------------ link probe

fn pstats(v: &mut Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    (v[0], v[n / 2], v[(n * 99 / 100).min(n - 1)])
}

/// Echo round trips via PING/PONG (spec A3), in microseconds.
fn ping_run(s: &mut TcpStream, seq: &mut u32, size: usize, iters: usize) -> io::Result<Vec<f64>> {
    let payload = vec![0xA5u8; size];
    let warm = (iters / 10).max(3);
    let mut out = Vec::with_capacity(iters);
    for i in 0..warm + iters {
        *seq += 1;
        let t = std::time::Instant::now();
        write_frame(s, kind::PING, *seq, &json!({"probe": i}), &payload, false)?;
        let f = read_frame(s)?.ok_or_else(|| io::Error::other("connection closed during probe"))?;
        if f.kind != kind::PONG || f.payload.len() != size {
            return Err(io::Error::other("bad PONG (peer must echo PING payloads)"));
        }
        if i >= warm {
            out.push(t.elapsed().as_secs_f64() * 1e6);
        }
    }
    Ok(out)
}

fn human(b: usize) -> String {
    if b >= 1 << 20 { format!("{:.1} MiB", b as f64 / 1048576.0) } else if b >= 1024 { format!("{:.1} KiB", b as f64 / 1024.0) } else { format!("{b} B") }
}

struct ProbeOpts {
    hidden: usize,
    max_mib: usize,
    assume_mbps: Option<f64>,
    assume_rtt_us: Option<f64>,
    decode_ms: Option<f64>,
    prefill_tok_s: Option<f64>,
}

fn probe(addrs: &[String], secret: Option<&str>, o: &ProbeOpts) -> io::Result<()> {
    let q8_row = o.hidden.div_ceil(32) * 34; // per 32 elements: f16 scale + 32 int8 (rotated blocks)
    let rows = [("f32", o.hidden * 4), ("bf16", o.hidden * 2), ("q8blk", q8_row)];
    for a in addrs {
        let mut s = TcpStream::connect(a)?;
        s.set_nodelay(true)?;
        let hello = client_handshake(&mut s, "probe", &["coordinator"], secret)?;
        println!("\n=== {a} (node '{}') ===", hello["node"].as_str().unwrap_or("?"));
        let mut seq = 100u32;

        let (rmin, rmed_measured, rp99) = pstats(&mut ping_run(&mut s, &mut seq, 0, 400)?);
        let rmed = o.assume_rtt_us.unwrap_or(rmed_measured);
        println!("round trip, empty message : min {rmin:.0} us   median {rmed_measured:.0} us   p99 {rp99:.0} us");

        // Decode-shaped messages: what one token's boundary looks like per dtype (echoed both ways).
        println!("\nechoed message size            median      p99     (decode-shaped, {} elements/row)", o.hidden);
        let mut decode_med = vec![];
        for (n, b) in rows {
            let (_, m, p) = pstats(&mut ping_run(&mut s, &mut seq, b, 300)?);
            decode_med.push(m);
            println!("  1 row {n:<6} {:>10}      {m:>7.0} us {p:>7.0} us", human(b));
        }

        // Bulk sizes -> sustained throughput. The peer reads a whole frame, then echoes it, so one-way ~ 2*size/time.
        println!("\n                               median        one-way estimate");
        let mut bw_best: f64 = 0.0;
        for (label, size, iters) in [("64 KiB", 64usize << 10, 200), ("1 MiB", 1 << 20, 60), ("16 MiB", 16 << 20, 12), ("64 MiB", 64 << 20, 5)] {
            if size > o.max_mib << 20 {
                continue;
            }
            let (_, m, _) = pstats(&mut ping_run(&mut s, &mut seq, size, iters)?);
            let mbps = 2.0 * size as f64 / (m * 1e-6) / 1e6;
            if size >= 1 << 20 {
                bw_best = bw_best.max(mbps);
            }
            println!("  {label:<8} {:>10}  {m:>10.0} us   {mbps:>9.0} MB/s  ({:.2} Gbit/s)", human(size), mbps * 8.0 / 1000.0);
        }
        let measured = bw_best;
        let mbps = o.assume_mbps.unwrap_or(measured).max(1e-3);
        if let Some(a) = o.assume_mbps {
            println!("\nWHAT-IF: using --assume-mbps {a} instead of the measured {measured:.0} MB/s");
        }
        if let Some(r) = o.assume_rtt_us {
            println!("WHAT-IF: using --assume-rtt-us {r} instead of the measured {rmed_measured:.0} us");
        }
        let bps = mbps * 1e6;
        println!("\nmodel at {mbps:.0} MB/s, rtt {rmed:.0} us (decode hop = rtt + one boundary row out; prefill ceiling = link-limited tokens/s)");
        println!("  dtype  bytes/row   decode hop     prefill ceiling");
        let mut hops = vec![];
        let mut ceil = vec![];
        for (n, b) in rows {
            let xfer = b as f64 / bps * 1e6;
            hops.push(rmed + xfer);
            ceil.push(bps / b as f64);
            println!("  {n:<6} {b:>9}   {:>7.0} us      {:>10.0} tok/s", rmed + xfer, bps / b as f64);
        }
        let saved = hops[0] - hops[2];
        print!("\nDECODE : q8blk saves ~{saved:.0} us/token vs f32 ({:.0}% of the hop)", saved / hops[0] * 100.0);
        match o.decode_ms {
            Some(t) => println!(", {:.2}% of a {t} ms token.", saved / (t * 1e3) * 100.0),
            None => println!(". Pass --decode-ms (your measured ms/token) to see it as a share of a token."),
        }
        let rtt_share = rmed / hops[0] * 100.0;
        println!("         latency (rtt) is {rtt_share:.0}% of the f32 hop: quantising does not touch it.");
        match o.prefill_tok_s {
            Some(r) => {
                println!("PREFILL: link utilisation at {r} tok/s (transfer time / compute time per chunk, if not overlapped):");
                for (i, (n, _)) in rows.iter().enumerate() {
                    let u = r / ceil[i] * 100.0;
                    let v = if u > 100.0 { "TRANSFER-BOUND" } else if u > 50.0 { "close: transfer visible" } else if u > 10.0 { "hidden if chunks are pipelined" } else { "negligible" };
                    println!("   {n:<6} {u:>6.1}%   {v}");
                }
                let u = r / ceil[0] * 100.0;
                println!("   heuristic: <10% leave f32; 10-70% try bf16 for prefill chunks only; >70% consider q8blk after a KL check (u(f32)={u:.0}%).");
            }
            None => println!("PREFILL: link-limited ceiling is {:.0} (f32) / {:.0} (bf16) / {:.0} (q8blk) tok/s. Pass --prefill-tok-s (your stage's prefill rate) for a verdict.", ceil[0], ceil[1], ceil[2]),
        }
        println!("planner input: \"link_rtt_ms\": {:.3}", rmed / 1000.0);
        let _ = decode_med;
    }
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
    if v.iter().any(|x| x == "--bench") {
        let addrs: Vec<String> = get("--stages").expect("--stages host:port,...").split(',').map(String::from).collect();
        let bounds: Vec<usize> = get("--bounds").map(|b| b.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()).unwrap_or_default();
        let n = |k: &str, d: usize| get(k).map(|x| x.parse().expect("bad number")).unwrap_or(d);
        let secret = secret_from_env(get("--secret-env").as_deref());
        if let Err(e) = bench(&addrs, &bounds, secret.as_deref(), n("--prompt-len", 256), n("--n", 64), n("--chunk", 64), get("--json")) {
            eprintln!("bench error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if v.iter().any(|x| x == "--probe") {
        let addrs: Vec<String> = get("--stages").expect("--stages host:port,...").split(',').map(String::from).collect();
        let num = |k: &str| get(k).map(|x| x.parse::<f64>().unwrap_or_else(|_| panic!("bad {k}")));
        let o = ProbeOpts {
            hidden: get("--hidden").map(|x| x.parse().expect("bad --hidden")).unwrap_or(4096),
            max_mib: get("--max-mib").map(|x| x.parse().expect("bad --max-mib")).unwrap_or(64),
            assume_mbps: num("--assume-mbps"),
            assume_rtt_us: num("--assume-rtt-us"),
            decode_ms: num("--decode-ms"),
            prefill_tok_s: num("--prefill-tok-s"),
        };
        if get("--hidden").is_none() {
            eprintln!("note: --hidden not given, assuming 4096 elements per row (GLM-5.3-Flash); pass the real hidden_size for exact numbers");
        }
        let secret = secret_from_env(get("--secret-env").as_deref());
        if let Err(e) = probe(&addrs, secret.as_deref(), &o) {
            eprintln!("probe error: {e}");
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
