use std::fs::File;
use std::io;

/// Positional read that works on unix and windows.
pub fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_at(buf, off)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        f.seek_read(buf, off)
    }
}

pub fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = read_at(f, buf, off)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
        }
        off += n as u64;
        let tmp = buf;
        buf = &mut tmp[n..];
    }
    Ok(())
}

/// Secret from an env var name, or None. Never accept secrets on the command line.
pub fn secret_from_env(var: Option<&str>) -> Option<String> {
    var.and_then(|v| std::env::var(v).ok()).filter(|s| !s.is_empty())
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
