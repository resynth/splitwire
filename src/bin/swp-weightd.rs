//! swp-weightd: weight manifest indexer, weight server, and shard puller (engine-free).
//!
//!   swp-weightd index --model DIR [--out FILE] [--no-hash] [--threads N] [--n-layers N]
//!   swp-weightd serve --model DIR [--listen ADDR] (--secret-env VAR | --insecure) [--threads N] [--n-layers N]
//!   swp-weightd pull  --server ADDR --out DIR (--layers A..B [--embed] [--head] [--extra] | --plan FILE --node NAME)
//!                     [--secret-env VAR] [--no-aux]
use serde_json::{json, Value};
use splitwire::client::WeightClient;
use splitwire::frame::{read_frame, write_frame};
use splitwire::manifest::*;
use splitwire::msg::{kind, send_error, server_handshake};
use splitwire::util::{read_exact_at, secret_from_env};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;

struct Args {
    kv: HashMap<String, String>,
}
const FLAGS: &[&str] = &["insecure", "embed", "head", "extra", "no-aux", "no-hash"];
impl Args {
    fn parse(it: impl Iterator<Item = String>) -> Args {
        let mut kv = HashMap::new();
        let v: Vec<String> = it.collect();
        let mut i = 0;
        while i < v.len() {
            if let Some(k) = v[i].strip_prefix("--") {
                if FLAGS.contains(&k) {
                    kv.insert(k.to_string(), "1".to_string());
                } else if i + 1 < v.len() {
                    kv.insert(k.to_string(), v[i + 1].clone());
                    i += 1;
                } else {
                    die(&format!("--{k} needs a value"));
                }
            } else {
                die(&format!("unexpected argument {}", v[i]));
            }
            i += 1;
        }
        Args { kv }
    }
    fn opt(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn flag(&self, k: &str) -> bool {
        self.kv.contains_key(k)
    }
    fn req(&self, k: &str) -> &str {
        self.opt(k).unwrap_or_else(|| die(&format!("missing --{k}")))
    }
    fn num<T: std::str::FromStr>(&self, k: &str) -> Option<T> {
        self.opt(k).map(|s| s.parse().unwrap_or_else(|_| die(&format!("bad --{k}"))))
    }
}
fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    exit(2)
}
fn threads(a: &Args) -> usize {
    a.num("threads").unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
}

fn main() {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_default();
    let a = Args::parse(it);
    let r = match cmd.as_str() {
        "index" => cmd_index(&a),
        "serve" => cmd_serve(&a),
        "pull" => cmd_pull(&a),
        _ => {
            eprintln!("usage: swp-weightd <index|serve|pull> [options]   (see source header)");
            exit(2)
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        exit(1);
    }
}

fn summarize(m: &Manifest) {
    let s = m.layer_sizes();
    let gb = |b: u64| b as f64 / 1e9;
    let d: u64 = s.dense.iter().sum();
    let r: u64 = s.routed.iter().sum();
    eprintln!(
        "format={} layers={:?} tensors={} | dense {:.2} GB, routed {:.2} GB, embed {:.2} GB, head {:.2} GB, extra(MTP) {:.2} GB, other {:.3} GB",
        m.format, m.n_layers, m.tensors.len(), gb(d), gb(r), gb(s.embed), gb(s.head), gb(s.extra), gb(s.other)
    );
    if let Some(h) = &m.manifest_hash {
        eprintln!("manifest_hash={h}");
    }
}

fn cmd_index(a: &Args) -> io::Result<()> {
    let dir = PathBuf::from(a.req("model"));
    let m = index_dir(&dir, !a.flag("no-hash"), threads(a), a.num("n-layers"))?;
    summarize(&m);
    let js = serde_json::to_vec_pretty(&m).unwrap();
    match a.opt("out") {
        Some(p) => std::fs::write(p, js)?,
        None => io::stdout().write_all(&js)?,
    }
    Ok(())
}

fn cmd_serve(a: &Args) -> io::Result<()> {
    let dir = Arc::new(PathBuf::from(a.req("model")));
    let secret = secret_from_env(a.opt("secret-env"));
    if secret.is_none() && !a.flag("insecure") {
        die("refusing to serve without --secret-env VAR (env var must be set) or an explicit --insecure");
    }
    eprintln!("indexing/hashing {} (cached in .swp-manifest.json) ...", dir.display());
    let m = Arc::new(load_or_index(&dir, threads(a), a.num("n-layers"))?);
    summarize(&m);
    let mj = Arc::new(serde_json::to_vec(&*m).unwrap());
    let listen = a.opt("listen").unwrap_or("0.0.0.0:7700");
    let l = TcpListener::bind(listen)?;
    eprintln!("swp-weightd listening on {listen} ({})", if secret.is_some() { "authenticated" } else { "INSECURE" });
    for c in l.incoming() {
        let Ok(s) = c else { continue };
        let (dir, m, mj, secret) = (dir.clone(), m.clone(), mj.clone(), secret.clone());
        std::thread::spawn(move || {
            let peer = s.peer_addr().map(|p| p.to_string()).unwrap_or_default();
            if let Err(e) = handle_conn(s, &dir, &m, &mj, secret.as_deref()) {
                eprintln!("[{peer}] closed: {e}");
            }
        });
    }
    Ok(())
}

const CHUNK: usize = 8 << 20;

fn handle_conn(mut s: TcpStream, dir: &Path, m: &Manifest, mj: &[u8], secret: Option<&str>) -> io::Result<()> {
    s.set_nodelay(true)?;
    server_handshake(&mut s, "weightd", &["weights"], secret)?;
    let idx: HashMap<&str, &TensorEntry> = m.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut files: HashMap<String, File> = HashMap::new();
    let mut buf = vec![0u8; CHUNK];
    while let Some(f) = read_frame(&mut s)? {
        match f.kind {
            kind::MANIFEST_GET => {
                write_frame(&mut s, kind::MANIFEST, f.seq, &json!({"manifest_hash": m.manifest_hash}), mj, false)?
            }
            kind::PING => write_frame(&mut s, kind::PONG, f.seq, &f.header, &f.payload, false)?,
            kind::TENSOR_GET => {
                let name = f.header["name"].as_str().unwrap_or("");
                let Some(t) = idx.get(name) else {
                    send_error(&mut s, f.seq, "not_found", &format!("no tensor {name}"))?;
                    continue;
                };
                if !files.contains_key(&t.file) {
                    files.insert(t.file.clone(), File::open(dir.join(&t.file))?);
                }
                let file = &files[&t.file];
                let mut h = blake3::Hasher::new();
                let mut done = 0u64;
                loop {
                    let n = ((t.len - done) as usize).min(CHUNK);
                    read_exact_at(file, &mut buf[..n], t.offset + done)?;
                    h.update(&buf[..n]);
                    let last = done + n as u64 == t.len;
                    let mut hd = json!({"name": name, "offset": done, "last": last});
                    if last {
                        hd["hash"] = json!(h.finalize().to_hex().to_string());
                    }
                    write_frame(&mut s, kind::TENSOR_DATA, f.seq, &hd, &buf[..n], false)?;
                    done += n as u64;
                    if last {
                        break;
                    }
                }
            }
            kind::AUX_GET => {
                let name = f.header["name"].as_str().unwrap_or("");
                if m.aux_files.iter().any(|x| x.name == name) {
                    let data = std::fs::read(dir.join(name))?;
                    write_frame(&mut s, kind::AUX_DATA, f.seq, &json!({"name": name}), &data, false)?;
                } else {
                    send_error(&mut s, f.seq, "not_found", &format!("no aux file {name}"))?;
                }
            }
            k => send_error(&mut s, f.seq, "unsupported", &format!("kind {k:#x} not served here"))?,
        }
    }
    Ok(())
}

fn parse_range(s: &str) -> (u32, u32) {
    let (a, b) = s.split_once("..").unwrap_or_else(|| die("--layers must look like A..B (half-open)"));
    (a.parse().unwrap_or_else(|_| die("bad layer start")), b.parse().unwrap_or_else(|_| die("bad layer end")))
}

fn cmd_pull(a: &Args) -> io::Result<()> {
    let out = PathBuf::from(a.req("out"));
    let secret = secret_from_env(a.opt("secret-env"));
    let (lo, hi, embed, head, extra) = if let Some(p) = a.opt("plan") {
        let node = a.req("node");
        let v: Value = serde_json::from_slice(&std::fs::read(p)?).map_err(io::Error::other)?;
        let st = v["stages"]
            .as_array()
            .and_then(|s| s.iter().find(|x| x["node"] == node))
            .unwrap_or_else(|| die(&format!("node {node} not in plan")));
        (
            st["layers"][0].as_u64().unwrap() as u32,
            st["layers"][1].as_u64().unwrap() as u32,
            st["embed"].as_bool().unwrap_or(false),
            st["head"].as_bool().unwrap_or(false),
            st["extra"].as_bool().unwrap_or(false),
        )
    } else {
        let (l, h) = parse_range(a.req("layers"));
        (l, h, a.flag("embed"), a.flag("head"), a.flag("extra"))
    };
    let mut c = WeightClient::connect(a.req("server"), "pull", secret.as_deref())?;
    let m = c.manifest()?;
    if m.format != "safetensors" {
        die("pull currently writes safetensors shards only (GGUF sources can be served/indexed, not pulled)");
    }
    let sel = m.select(lo, hi, embed, head, extra);
    let total: u64 = sel.iter().map(|t| t.len).sum();
    eprintln!("pulling {} tensors, {:.2} GB: layers {lo}..{hi} embed={embed} head={head} extra={extra}", sel.len(), total as f64 / 1e9);
    std::fs::create_dir_all(&out)?;
    let tag = format!("{}{}{}", if embed { "-embed" } else { "" }, if head { "-head" } else { "" }, if extra { "-extra" } else { "" });
    let name = format!("swp-shard-L{lo}-{hi}{tag}.safetensors");
    let part = out.join(format!("{name}.part"));
    let meta = json!({"swp_source": m.manifest_hash, "swp_layers": format!("{lo}..{hi}"), "format": "pt"});
    let (header, _offs) = shard_header(&sel, meta);
    let mut w = BufWriter::with_capacity(8 << 20, File::create(&part)?);
    w.write_all(&header)?;
    let mut order: Vec<&&TensorEntry> = sel.iter().collect();
    order.sort_by(|x, y| x.name.cmp(&y.name)); // must match header layout (name order)
    let t0 = std::time::Instant::now();
    let (mut done, mut next_report) = (0u64, 1u64 << 30);
    for t in order {
        let (n, h) = c.fetch_tensor(&t.name, &mut w)?;
        if n != t.len || t.hash.as_deref().map_or(false, |x| x != h) {
            return Err(io::Error::other(format!("{}: size/hash mismatch vs manifest", t.name)));
        }
        done += n;
        if done >= next_report {
            eprintln!("  {:.1} / {:.1} GB  ({:.2} GB/s)", done as f64 / 1e9, total as f64 / 1e9, done as f64 / 1e9 / t0.elapsed().as_secs_f64());
            next_report += 1 << 30;
        }
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&part, out.join(&name))?;
    if !a.flag("no-aux") {
        for f in &m.aux_files {
            let p = out.join(&f.name);
            if std::fs::metadata(&p).map(|md| md.len() == f.len).unwrap_or(false) {
                continue;
            }
            std::fs::write(p, c.fetch_aux(&f.name)?)?;
        }
    }
    std::fs::write(
        out.join(format!("{name}.swp.json")),
        serde_json::to_vec_pretty(&json!({"source_manifest_hash": m.manifest_hash, "layers": [lo, hi], "embed": embed, "head": head, "extra": extra, "tensors": sel.len(), "bytes": total})).unwrap(),
    )?;
    eprintln!("done: {} in {:.1}s", out.join(&name).display(), t0.elapsed().as_secs_f64());
    Ok(())
}
