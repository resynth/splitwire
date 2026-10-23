//! swp-mockstage: deterministic toy stage server for adapter conformance testing.
//!   swp-mockstage --listen 127.0.0.1:7801 [--node NAME] [--secret-env VAR] [--sim-base-ms X] [--sim-row-us Y]
//! The sim flags make each FORWARD take X ms + Y us per row, to exercise swp-drive --bench without GPUs.
use splitwire::{mock::serve_conn, util::secret_from_env};
use std::net::TcpListener;

fn main() {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| v.iter().position(|x| x == k).and_then(|i| v.get(i + 1)).cloned();
    let listen = get("--listen").unwrap_or_else(|| "127.0.0.1:7801".into());
    let node = get("--node").unwrap_or_else(|| "mockstage".into());
    let secret = secret_from_env(get("--secret-env").as_deref());
    let f = |k: &str| get(k).map(|x| x.parse::<f64>().expect("bad number"));
    splitwire::mock::SIM_BASE_US.store((f("--sim-base-ms").unwrap_or(0.0) * 1000.0) as u64, std::sync::atomic::Ordering::Relaxed);
    splitwire::mock::SIM_ROW_US.store(f("--sim-row-us").unwrap_or(0.0) as u64, std::sync::atomic::Ordering::Relaxed);
    let l = TcpListener::bind(&listen).expect("bind");
    eprintln!("swp-mockstage '{node}' on {listen} ({})", if secret.is_some() { "authenticated" } else { "no auth" });
    for c in l.incoming().flatten() {
        let (n, s) = (node.clone(), secret.clone());
        std::thread::spawn(move || {
            if let Err(e) = serve_conn(c, &n, s.as_deref()) {
                eprintln!("connection closed: {e}");
            }
        });
    }
}
