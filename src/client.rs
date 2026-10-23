//! Weight-service client.
use crate::frame::{read_frame, write_frame};
use crate::manifest::Manifest;
use crate::msg::{client_handshake, expect, kind};
use serde_json::json;
use std::io::{self, Write};
use std::net::TcpStream;

pub struct WeightClient {
    s: TcpStream,
    seq: u32,
}

impl WeightClient {
    pub fn connect(addr: &str, node: &str, secret: Option<&str>) -> io::Result<Self> {
        let mut s = TcpStream::connect(addr)?;
        s.set_nodelay(true)?;
        client_handshake(&mut s, node, &["weights"], secret)?;
        Ok(Self { s, seq: 2 })
    }

    fn next(&mut self) -> u32 {
        self.seq += 1;
        self.seq
    }

    pub fn manifest(&mut self) -> io::Result<Manifest> {
        let q = self.next();
        write_frame(&mut self.s, kind::MANIFEST_GET, q, &json!({}), &[], false)?;
        let f = expect(read_frame(&mut self.s)?.ok_or_else(eof)?, kind::MANIFEST)?;
        serde_json::from_slice(&f.payload).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Stream one tensor into `out`; returns (bytes, blake3 hex computed locally). Verifies the server's hash too.
    pub fn fetch_tensor<W: Write>(&mut self, name: &str, out: &mut W) -> io::Result<(u64, String)> {
        let q = self.next();
        write_frame(&mut self.s, kind::TENSOR_GET, q, &json!({"name": name}), &[], false)?;
        let mut h = blake3::Hasher::new();
        let mut total = 0u64;
        loop {
            let f = expect(read_frame(&mut self.s)?.ok_or_else(eof)?, kind::TENSOR_DATA)?;
            h.update(&f.payload);
            out.write_all(&f.payload)?;
            total += f.payload.len() as u64;
            if f.header["last"].as_bool() == Some(true) {
                let local = h.finalize().to_hex().to_string();
                if let Some(sh) = f.header["hash"].as_str() {
                    if sh != local {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{name}: transfer hash mismatch")));
                    }
                }
                return Ok((total, local));
            }
        }
    }

    pub fn fetch_aux(&mut self, name: &str) -> io::Result<Vec<u8>> {
        let q = self.next();
        write_frame(&mut self.s, kind::AUX_GET, q, &json!({"name": name}), &[], false)?;
        Ok(expect(read_frame(&mut self.s)?.ok_or_else(eof)?, kind::AUX_DATA)?.payload)
    }
}

fn eof() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed")
}
