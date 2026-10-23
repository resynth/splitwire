//! SWP/1 framing. All integers little-endian.
//!
//! ```text
//!  0  4  magic "SWP1"
//!  4  1  kind
//!  5  1  flags   bit0: 16-byte blake3 trailer follows the payload
//!  6  2  reserved (0)
//!  8  4  seq     request id; responses echo the request's seq
//! 12  4  hdr_len (<= 1 MiB)
//! 16  8  pay_len
//! 24  hdr_len bytes   JSON object (UTF-8)
//!     pay_len bytes   raw payload
//!     [16 bytes       blake3(payload)[..16] if flag bit0]
//! ```
use serde_json::Value;
use std::io::{self, Read, Write};

pub const MAGIC: [u8; 4] = *b"SWP1";
pub const FIXED_LEN: usize = 24;
pub const FLAG_HASH: u8 = 0x01;
pub const MAX_HEADER: u32 = 1 << 20;
pub const MAX_PAYLOAD: u64 = 8 << 30;

#[derive(Debug, Clone)]
pub struct Frame {
    pub kind: u8,
    pub flags: u8,
    pub seq: u32,
    pub header: Value,
    pub payload: Vec<u8>,
}

pub fn write_frame<W: Write>(
    w: &mut W,
    kind: u8,
    seq: u32,
    header: &Value,
    payload: &[u8],
    hash: bool,
) -> io::Result<()> {
    let hdr = serde_json::to_vec(header).map_err(io::Error::other)?;
    if hdr.len() as u64 > MAX_HEADER as u64 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "header too large"));
    }
    let mut head = Vec::with_capacity(FIXED_LEN + hdr.len());
    head.extend_from_slice(&MAGIC);
    head.push(kind);
    head.push(if hash { FLAG_HASH } else { 0 });
    head.extend_from_slice(&[0, 0]);
    head.extend_from_slice(&seq.to_le_bytes());
    head.extend_from_slice(&(hdr.len() as u32).to_le_bytes());
    head.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    head.extend_from_slice(&hdr);
    w.write_all(&head)?;
    w.write_all(payload)?;
    if hash {
        w.write_all(&blake3::hash(payload).as_bytes()[..16])?;
    }
    w.flush()
}

/// Ok(None) = clean EOF before a frame started.
pub fn read_frame<R: Read>(r: &mut R) -> io::Result<Option<Frame>> {
    let mut fixed = [0u8; FIXED_LEN];
    loop {
        match r.read(&mut fixed[..1]) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    r.read_exact(&mut fixed[1..])?;
    if fixed[..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let kind = fixed[4];
    let flags = fixed[5];
    let seq = u32::from_le_bytes(fixed[8..12].try_into().unwrap());
    let hdr_len = u32::from_le_bytes(fixed[12..16].try_into().unwrap());
    let pay_len = u64::from_le_bytes(fixed[16..24].try_into().unwrap());
    if hdr_len > MAX_HEADER || pay_len > MAX_PAYLOAD {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    let mut hdr = vec![0u8; hdr_len as usize];
    r.read_exact(&mut hdr)?;
    let header: Value = serde_json::from_slice(&hdr)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad header json: {e}")))?;
    let mut payload = vec![0u8; pay_len as usize];
    r.read_exact(&mut payload)?;
    if flags & FLAG_HASH != 0 {
        let mut t = [0u8; 16];
        r.read_exact(&mut t)?;
        if t != blake3::hash(&payload).as_bytes()[..16] {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "payload hash mismatch"));
        }
    }
    Ok(Some(Frame { kind, flags, seq, header, payload }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn roundtrip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, 0x22, 7, &json!({"a": 1, "b": "x"}), b"hello", true).unwrap();
        let f = read_frame(&mut &buf[..]).unwrap().unwrap();
        assert_eq!((f.kind, f.seq, f.payload.as_slice()), (0x22, 7, &b"hello"[..]));
        assert_eq!(f.header["b"], "x");
    }
    #[test]
    fn detects_corruption() {
        let mut buf = Vec::new();
        write_frame(&mut buf, 1, 1, &json!({}), b"payload", true).unwrap();
        let n = buf.len();
        buf[n - 20] ^= 0xff;
        assert!(read_frame(&mut &buf[..]).is_err());
    }
}
