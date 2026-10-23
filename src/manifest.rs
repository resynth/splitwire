//! Weight manifests: tensor-level index of a model directory (safetensors or GGUF),
//! per-tensor blake3 hashes, and layer/group classification.
use crate::util::{hex, read_exact_at};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FileEntry {
    pub name: String,
    pub len: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub file: String,
    /// absolute byte offset of the tensor data within `file`
    pub offset: u64,
    /// extent in bytes (for GGUF this includes alignment padding up to the next tensor)
    pub len: u64,
    #[serde(default)]
    pub hash: Option<String>,
    /// layer | extra | embed | head | other
    pub group: String,
    #[serde(default)]
    pub layer: Option<u32>,
    /// routed-expert tensor (MoE); used by the planner
    #[serde(default)]
    pub routed: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    pub swp_manifest: u32,
    pub format: String,
    pub n_layers: Option<u32>,
    pub files: Vec<FileEntry>,
    pub aux_files: Vec<FileEntry>,
    pub tensors: Vec<TensorEntry>,
    #[serde(default)]
    pub manifest_hash: Option<String>,
}

pub struct LayerSizes {
    pub dense: Vec<u64>,
    pub routed: Vec<u64>,
    pub embed: u64,
    pub head: u64,
    pub extra: u64,
    pub other: u64,
}

impl Manifest {
    pub fn compute_hash(&self) -> String {
        let mut ts: Vec<&TensorEntry> = self.tensors.iter().collect();
        ts.sort_by(|a, b| a.name.cmp(&b.name));
        let mut h = blake3::Hasher::new();
        for t in ts {
            h.update(t.name.as_bytes());
            h.update(t.dtype.as_bytes());
            for d in &t.shape {
                h.update(&d.to_le_bytes());
            }
            h.update(&t.len.to_le_bytes());
            h.update(t.hash.as_deref().unwrap_or("").as_bytes());
        }
        h.finalize().to_hex().to_string()
    }

    pub fn layer_sizes(&self) -> LayerSizes {
        let n = self.n_layers.map(|x| x as usize).unwrap_or_else(|| {
            self.tensors.iter().filter_map(|t| t.layer).max().map(|m| m as usize + 1).unwrap_or(0)
        });
        let mut s = LayerSizes { dense: vec![0; n], routed: vec![0; n], embed: 0, head: 0, extra: 0, other: 0 };
        for t in &self.tensors {
            match t.group.as_str() {
                "layer" => {
                    let l = t.layer.unwrap() as usize;
                    if l < n {
                        if t.routed { s.routed[l] += t.len } else { s.dense[l] += t.len }
                    }
                }
                "embed" => s.embed += t.len,
                "head" => s.head += t.len,
                "extra" => s.extra += t.len,
                _ => s.other += t.len,
            }
        }
        s
    }

    /// Tensors for a stage: layers in [a, b), plus optional embed/head/extra groups. "other" always included.
    pub fn select(&self, a: u32, b: u32, embed: bool, head: bool, extra: bool) -> Vec<&TensorEntry> {
        self.tensors
            .iter()
            .filter(|t| match t.group.as_str() {
                "layer" => t.layer.map_or(false, |l| l >= a && l < b),
                "embed" => embed,
                "head" => head,
                "extra" => extra,
                _ => true,
            })
            .collect()
    }
}

pub fn classify(name: &str, n_layers: Option<u32>) -> (String, Option<u32>, bool) {
    let toks: Vec<&str> = name.split('.').collect();
    let mut layer = None;
    for w in toks.windows(2) {
        if matches!(w[0], "layers" | "blk" | "h") {
            if let Ok(n) = w[1].parse::<u32>() {
                layer = Some(n);
                break;
            }
        }
    }
    let routed = name.contains(".experts.") || name.contains("_exps");
    let lower = name.to_ascii_lowercase();
    let group = match layer {
        Some(l) => match n_layers {
            Some(n) if l >= n => "extra",
            _ => "layer",
        },
        None => {
            if lower.contains("embed_tokens") || lower.contains("token_embd") || lower.contains("wte") {
                "embed"
            } else if lower.starts_with("lm_head")
                || lower.starts_with("model.norm")
                || lower.starts_with("output")
                || lower.starts_with("norm.")
            {
                "head"
            } else {
                "other"
            }
        }
    };
    (group.to_string(), layer, routed)
}

// ---------------------------------------------------------------- safetensors

fn index_safetensors_file(dir: &Path, fname: &str, out: &mut Vec<TensorEntry>) -> io::Result<u64> {
    let mut f = File::open(dir.join(fname))?;
    let flen = f.metadata()?.len();
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n);
    if n > (256 << 20) || 8 + n > flen {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{fname}: bad safetensors header")));
    }
    let mut h = vec![0u8; n as usize];
    f.read_exact(&mut h)?;
    let v: Value = serde_json::from_slice(&h)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{fname}: {e}")))?;
    for (k, val) in v.as_object().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "header not object"))? {
        if k == "__metadata__" {
            continue;
        }
        let bad = || io::Error::new(io::ErrorKind::InvalidData, format!("{fname}: bad entry {k}"));
        let off = val["data_offsets"].as_array().ok_or_else(bad)?;
        let (b, e) = (off[0].as_u64().ok_or_else(bad)?, off[1].as_u64().ok_or_else(bad)?);
        out.push(TensorEntry {
            name: k.clone(),
            dtype: val["dtype"].as_str().unwrap_or("?").to_string(),
            shape: val["shape"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default(),
            file: fname.to_string(),
            offset: 8 + n + b,
            len: e - b,
            hash: None,
            group: String::new(),
            layer: None,
            routed: false,
        });
    }
    Ok(flen)
}

// ----------------------------------------------------------------------- gguf

fn rd_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn rd_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn rd_str<R: Read>(r: &mut R) -> io::Result<String> {
    let n = rd_u64(r)?;
    if n > (1 << 24) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "gguf string too long"));
    }
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}
fn skip<R: Read>(r: &mut R, n: u64) -> io::Result<()> {
    io::copy(&mut r.take(n), &mut io::sink()).map(|_| ())
}
fn skip_value<R: Read>(r: &mut R, ty: u32) -> io::Result<()> {
    match ty {
        0 | 1 | 7 => skip(r, 1),
        2 | 3 => skip(r, 2),
        4 | 5 | 6 => skip(r, 4),
        10 | 11 | 12 => skip(r, 8),
        8 => {
            let n = rd_u64(r)?;
            skip(r, n)
        }
        9 => {
            let et = rd_u32(r)?;
            let n = rd_u64(r)?;
            for _ in 0..n {
                skip_value(r, et)?;
            }
            Ok(())
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("gguf: unknown value type {ty}"))),
    }
}

pub fn ggml_type_name(id: u32) -> String {
    let n = match id {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K",
        16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S",
        22 => "IQ2_S", 23 => "IQ4_XS", 24 => "I8", 25 => "I16", 26 => "I32", 27 => "I64", 28 => "F64",
        29 => "IQ1_M", 30 => "BF16",
        _ => return format!("ggml:{id}"),
    };
    n.to_string()
}

/// Returns (file_len, n_layers_from_kv)
fn index_gguf_file(dir: &Path, fname: &str, out: &mut Vec<TensorEntry>) -> io::Result<(u64, Option<u32>)> {
    let f = File::open(dir.join(fname))?;
    let flen = f.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{fname}: not GGUF")));
    }
    let ver = rd_u32(&mut r)?;
    if ver < 2 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "GGUF v1 unsupported"));
    }
    let n_tensors = rd_u64(&mut r)?;
    let n_kv = rd_u64(&mut r)?;
    let mut align: u64 = 32;
    let mut blocks = None;
    for _ in 0..n_kv {
        let key = rd_str(&mut r)?;
        let ty = rd_u32(&mut r)?;
        if key == "general.alignment" && ty == 4 {
            align = rd_u32(&mut r)? as u64;
        } else if key.ends_with(".block_count") && ty == 4 {
            blocks = Some(rd_u32(&mut r)?);
        } else {
            skip_value(&mut r, ty)?;
        }
    }
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = rd_str(&mut r)?;
        let nd = rd_u32(&mut r)?;
        let mut shape = Vec::new();
        for _ in 0..nd {
            shape.push(rd_u64(&mut r)?);
        }
        let ty = rd_u32(&mut r)?;
        let off = rd_u64(&mut r)?;
        infos.push((name, shape, ty, off));
    }
    let pos = r.stream_position()?;
    let data_start = pos.div_ceil(align) * align;
    infos.sort_by_key(|i| i.3);
    for i in 0..infos.len() {
        let end = if i + 1 < infos.len() { infos[i + 1].3 } else { flen.saturating_sub(data_start) };
        let (name, shape, ty, off) = infos[i].clone();
        out.push(TensorEntry {
            name,
            dtype: ggml_type_name(ty),
            shape,
            file: fname.to_string(),
            offset: data_start + off,
            len: end.saturating_sub(off),
            hash: None,
            group: String::new(),
            layer: None,
            routed: false,
        });
    }
    Ok((flen, blocks))
}

// ------------------------------------------------------------------- indexing

fn n_layers_from_config(dir: &Path) -> Option<u32> {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    v["num_hidden_layers"].as_u64().or_else(|| v["text_config"]["num_hidden_layers"].as_u64()).map(|x| x as u32)
}

pub fn index_dir(dir: &Path, with_hash: bool, threads: usize, n_layers: Option<u32>) -> io::Result<Manifest> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let st: Vec<&String> = names.iter().filter(|n| n.ends_with(".safetensors")).collect();
    let gg: Vec<&String> = names.iter().filter(|n| n.ends_with(".gguf")).collect();
    let mut tensors = Vec::new();
    let mut files = Vec::new();
    let mut kv_layers = None;
    let format;
    if !st.is_empty() {
        format = "safetensors";
        for n in st {
            let len = index_safetensors_file(dir, n, &mut tensors)?;
            files.push(FileEntry { name: n.clone(), len });
        }
    } else if !gg.is_empty() {
        format = "gguf";
        for n in gg {
            let (len, b) = index_gguf_file(dir, n, &mut tensors)?;
            kv_layers = kv_layers.or(b);
            files.push(FileEntry { name: n.clone(), len });
        }
    } else {
        return Err(io::Error::new(io::ErrorKind::NotFound, "no .safetensors or .gguf files found"));
    }
    let n_layers = n_layers.or_else(|| n_layers_from_config(dir)).or(kv_layers);
    for t in &mut tensors {
        let (g, l, r) = classify(&t.name, n_layers);
        t.group = g;
        t.layer = l;
        t.routed = r;
    }
    let mut aux_files = Vec::new();
    if format == "safetensors" {
        for n in &names {
            let ext = Path::new(n).extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "json" | "txt" | "jinja" | "model" | "tiktoken")
                && n != "model.safetensors.index.json"
                && !n.starts_with(".swp")
            {
                let len = std::fs::metadata(dir.join(n))?.len();
                if len <= (64 << 20) {
                    aux_files.push(FileEntry { name: n.clone(), len });
                }
            }
        }
    }
    if with_hash {
        hash_tensors(dir, &mut tensors, threads)?;
    }
    let mut m = Manifest {
        swp_manifest: 1,
        format: format.to_string(),
        n_layers,
        files,
        aux_files,
        tensors,
        manifest_hash: None,
    };
    if with_hash {
        m.manifest_hash = Some(m.compute_hash());
    }
    Ok(m)
}

pub fn hash_range(f: &File, off: u64, len: u64, buf: &mut [u8]) -> io::Result<String> {
    let mut h = blake3::Hasher::new();
    let mut done = 0u64;
    while done < len {
        let n = ((len - done) as usize).min(buf.len());
        read_exact_at(f, &mut buf[..n], off + done)?;
        h.update(&buf[..n]);
        done += n as u64;
    }
    Ok(h.finalize().to_hex().to_string())
}

pub fn hash_tensors(dir: &Path, tensors: &mut [TensorEntry], threads: usize) -> io::Result<()> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());
    let first_err: Mutex<Option<io::Error>> = Mutex::new(None);
    {
        let ro: &[TensorEntry] = &*tensors;
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| {
                    let mut buf = vec![0u8; 4 << 20];
                    let mut files: HashMap<&str, File> = HashMap::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= ro.len() || first_err.lock().unwrap().is_some() {
                            break;
                        }
                        let t = &ro[i];
                        let r = (|| -> io::Result<String> {
                            if !files.contains_key(t.file.as_str()) {
                                files.insert(t.file.as_str(), File::open(dir.join(&t.file))?);
                            }
                            hash_range(&files[t.file.as_str()], t.offset, t.len, &mut buf)
                        })();
                        match r {
                            Ok(h) => results.lock().unwrap().push((i, h)),
                            Err(e) => {
                                *first_err.lock().unwrap() = Some(e);
                                break;
                            }
                        }
                    }
                });
            }
        });
    }
    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }
    for (i, h) in results.into_inner().unwrap() {
        tensors[i].hash = Some(h);
    }
    Ok(())
}

/// Load `.swp-manifest.json` if still valid for the directory, else index (with hashes) and cache it.
pub fn load_or_index(dir: &Path, threads: usize, n_layers: Option<u32>) -> io::Result<Manifest> {
    let cache = dir.join(".swp-manifest.json");
    if let Ok(b) = std::fs::read(&cache) {
        if let Ok(m) = serde_json::from_slice::<Manifest>(&b) {
            let ok = m.manifest_hash.is_some()
                && m.files.iter().all(|f| std::fs::metadata(dir.join(&f.name)).map(|md| md.len() == f.len).unwrap_or(false))
                && n_layers.map_or(true, |n| m.n_layers == Some(n));
            if ok {
                return Ok(m);
            }
        }
    }
    let m = index_dir(dir, true, threads, n_layers)?;
    if let Err(e) = std::fs::write(&cache, serde_json::to_vec_pretty(&m).unwrap()) {
        eprintln!("warning: could not cache manifest at {}: {e}", cache.display());
    }
    Ok(m)
}

/// Build a safetensors header for the given tensors laid out back to back (sorted by name).
/// Returns (header_bytes_including_8_byte_length_prefix, offsets_in_data_section_by_input_index).
pub fn shard_header(tensors: &[&TensorEntry], meta: Value) -> (Vec<u8>, Vec<u64>) {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by(|&a, &b| tensors[a].name.cmp(&tensors[b].name));
    let mut offs = vec![0u64; tensors.len()];
    let mut obj = serde_json::Map::new();
    obj.insert("__metadata__".into(), meta);
    let mut cur = 0u64;
    for &i in &order {
        let t = tensors[i];
        offs[i] = cur;
        obj.insert(
            t.name.clone(),
            json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [cur, cur + t.len]}),
        );
        cur += t.len;
    }
    let mut h = serde_json::to_vec(&Value::Object(obj)).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    (out, offs)
}

pub fn short(h: &str) -> String {
    h.chars().take(12).collect::<String>()
}
#[allow(dead_code)]
fn _unused(_: &[u8]) -> String {
    hex(&[])
}
