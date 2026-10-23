//! Message kinds, handshake and small header helpers.
use crate::frame::{read_frame, write_frame, Frame};
use serde_json::{json, Value};
use std::io::{self, Read, Write};

pub const PROTO: &str = "swp/1";

pub mod kind {
    pub const HELLO: u8 = 0x01;
    pub const HELLO_OK: u8 = 0x02;
    pub const ERROR: u8 = 0x0F;
    pub const PING: u8 = 0x10;
    pub const PONG: u8 = 0x11;
    pub const STAGE_CONFIGURE: u8 = 0x20;
    pub const STAGE_READY: u8 = 0x21;
    pub const FORWARD: u8 = 0x22;
    pub const FORWARD_OUT: u8 = 0x23;
    pub const CACHE_OP: u8 = 0x24;
    pub const CACHE_OK: u8 = 0x25;
    pub const SESSION_CLOSE: u8 = 0x26;
    pub const SESSION_CLOSED: u8 = 0x27;
    pub const MANIFEST_GET: u8 = 0x40;
    pub const MANIFEST: u8 = 0x41;
    pub const TENSOR_GET: u8 = 0x42;
    pub const TENSOR_DATA: u8 = 0x43;
    pub const AUX_GET: u8 = 0x44;
    pub const AUX_DATA: u8 = 0x45;
}

pub fn err_header(code: &str, message: &str) -> Value {
    json!({"code": code, "message": message})
}

pub fn send_error<W: Write>(w: &mut W, seq: u32, code: &str, message: &str) -> io::Result<()> {
    write_frame(w, kind::ERROR, seq, &err_header(code, message), &[], false)
}

pub fn random_nonce() -> String {
    let mut b = [0u8; 16];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut b).is_ok() {
            return crate::util::hex(&b);
        }
    }
    // Fallback (non-unix): time + pid + address entropy, hashed.
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let local = 0u8;
    let mut h = blake3::Hasher::new();
    h.update(&t.to_le_bytes());
    h.update(&std::process::id().to_le_bytes());
    h.update(&(&local as *const u8 as usize).to_le_bytes());
    crate::util::hex(&h.finalize().as_bytes()[..16])
}

pub fn proof(secret: &str, nonce: &str, node: &str) -> String {
    let key = blake3::hash(secret.as_bytes());
    let mut h = blake3::Hasher::new_keyed(key.as_bytes());
    h.update(nonce.as_bytes());
    h.update(b"|");
    h.update(node.as_bytes());
    h.finalize().to_hex().to_string()
}

/// Server side: server speaks first (HELLO with nonce), client answers with HELLO+proof.
/// Authentication is a keyed-hash challenge; it does NOT encrypt the connection.
pub fn server_handshake<S: Read + Write>(
    s: &mut S,
    node: &str,
    caps: &[&str],
    secret: Option<&str>,
) -> io::Result<Value> {
    let nonce = random_nonce();
    write_frame(
        s,
        kind::HELLO,
        0,
        &json!({"proto": PROTO, "node": node, "caps": caps, "nonce": nonce, "auth": secret.is_some()}),
        &[],
        false,
    )?;
    let f = read_frame(s)?.ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "no client hello"))?;
    if f.kind != kind::HELLO || f.header["proto"] != PROTO {
        send_error(s, f.seq, "proto", "expected swp/1 HELLO")?;
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad client hello"));
    }
    if let Some(sec) = secret {
        let peer = f.header["node"].as_str().unwrap_or("");
        let want = proof(sec, &nonce, peer);
        if f.header["proof"].as_str() != Some(want.as_str()) {
            send_error(s, f.seq, "auth", "authentication failed")?;
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "auth failed"));
        }
    }
    write_frame(s, kind::HELLO_OK, f.seq, &json!({"proto": PROTO, "node": node, "caps": caps}), &[], false)?;
    Ok(f.header)
}

pub fn client_handshake<S: Read + Write>(
    s: &mut S,
    node: &str,
    caps: &[&str],
    secret: Option<&str>,
) -> io::Result<Value> {
    let sh = read_frame(s)?.ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "no server hello"))?;
    if sh.kind != kind::HELLO || sh.header["proto"] != PROTO {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "server is not swp/1"));
    }
    let mut hello = json!({"proto": PROTO, "node": node, "caps": caps});
    if sh.header["auth"].as_bool() == Some(true) {
        let sec = secret.ok_or_else(|| {
            io::Error::new(io::ErrorKind::PermissionDenied, "server requires a secret (set the env var)")
        })?;
        let nonce = sh.header["nonce"].as_str().unwrap_or("");
        hello["proof"] = json!(proof(sec, nonce, node));
    }
    write_frame(s, kind::HELLO, 1, &hello, &[], false)?;
    let r = read_frame(s)?.ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "no hello_ok"))?;
    match r.kind {
        kind::HELLO_OK => Ok(sh.header),
        kind::ERROR => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            r.header["message"].as_str().unwrap_or("handshake rejected").to_string(),
        )),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected handshake reply")),
    }
}

/// Expect a specific kind; turn ERROR frames into io errors.
pub fn expect(f: Frame, want: u8) -> io::Result<Frame> {
    if f.kind == want {
        Ok(f)
    } else if f.kind == kind::ERROR {
        Err(io::Error::other(format!(
            "{}: {}",
            f.header["code"].as_str().unwrap_or("error"),
            f.header["message"].as_str().unwrap_or("")
        )))
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, format!("unexpected frame kind {:#x}", f.kind)))
    }
}

pub fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

pub fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
