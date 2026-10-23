//! swp-plan: assign contiguous layer ranges to nodes, minimizing estimated decode time per token.
//!
//!   swp-plan --manifest manifest.json --nodes nodes.json [--experts 8/288] [--order a,b] [--replicate-embed-last] [--out plan.json]
//!
//! nodes.json:
//!   {"nodes":[{"name":"strix","reserve_gb":8,"link_rtt_ms":0.0,
//!              "tiers":[{"name":"gpu","gb":120,"bw_gbps":230}]},
//!             {"name":"razer","reserve_gb":2,"link_rtt_ms":0.4,
//!              "tiers":[{"name":"vram","gb":16,"bw_gbps":400},{"name":"ram","gb":56,"bw_gbps":60}]}]}
//! Tiers are listed fastest-first. `reserve_gb` is taken from tier 0 (KV cache + workspace).
//! `link_rtt_ms` is the round-trip to reach this node from the previous stage; the coordinator is
//! assumed to sit on the first node. Estimates are a bandwidth-bound model of ONE decode step:
//! use them to rank placements, then measure.
use serde::Deserialize;
use serde_json::json;
use splitwire::manifest::Manifest;
use std::process::exit;

#[derive(Deserialize, Clone)]
struct Tier {
    name: String,
    gb: f64,
    bw_gbps: f64,
}
#[derive(Deserialize, Clone)]
struct Node {
    name: String,
    tiers: Vec<Tier>,
    #[serde(default)]
    reserve_gb: f64,
    #[serde(default)]
    link_rtt_ms: f64,
}
#[derive(Deserialize)]
struct NodesFile {
    nodes: Vec<Node>,
}

#[derive(Clone)]
struct Eval {
    time_s: f64,
    tier_bytes: Vec<f64>,
}

fn fill(caps: &mut [f64], mut amount: f64) -> Vec<f64> {
    let mut out = vec![0.0; caps.len()];
    for (i, c) in caps.iter_mut().enumerate() {
        let take = amount.min(*c);
        out[i] = take;
        *c -= take;
        amount -= take;
    }
    out
}

/// read_hot: bytes read every token (dense + head); cap_only: resident but rarely read (embed, MTP, other)
fn eval(node: &Node, read_hot: f64, cap_only: f64, routed: f64, active: f64) -> Option<Eval> {
    let mut caps: Vec<f64> = node.tiers.iter().map(|t| t.gb * 1e9).collect();
    caps[0] -= node.reserve_gb * 1e9;
    if caps[0] < 0.0 {
        return None;
    }
    if read_hot + cap_only + routed > caps.iter().sum::<f64>() {
        return None;
    }
    let hot_total = read_hot + cap_only;
    let hot = fill(&mut caps, hot_total);
    let rt = fill(&mut caps, routed);
    let frac = if hot_total > 0.0 { read_hot / hot_total } else { 0.0 };
    let mut time = 0.0;
    let mut tb = vec![0.0; node.tiers.len()];
    for (i, t) in node.tiers.iter().enumerate() {
        time += (hot[i] * frac + rt[i] * active) / (t.bw_gbps * 1e9);
        tb[i] = hot[i] + rt[i];
    }
    Some(Eval { time_s: time, tier_bytes: tb })
}

struct Ctx {
    pd: Vec<f64>,
    pr: Vec<f64>,
    embed: f64,
    head: f64,
    extra: f64,
    other: f64,
    active: f64,
    rep_embed: bool,
    l: usize,
}

fn stage_eval(c: &Ctx, node: &Node, a: usize, b: usize, first: bool, last: bool) -> Option<Eval> {
    let dense = c.pd[b] - c.pd[a];
    let routed = c.pr[b] - c.pr[a];
    let mut read = dense;
    let mut cap = 0.0;
    if first {
        cap += c.embed + c.other;
    }
    if last {
        read += c.head;
        cap += c.extra + if c.rep_embed && !first { c.embed } else { 0.0 };
    }
    eval(node, read, cap, routed, c.active)
}

type Solution = (f64, Vec<(usize, usize)>, Vec<Eval>);

fn solve(c: &Ctx, nodes: &[Node], order: &[usize]) -> Option<Solution> {
    let n = order.len();
    let l = c.l;
    let inf = f64::INFINITY;
    let mut f = vec![vec![inf; l + 1]; n + 1];
    let mut back = vec![vec![0usize; l + 1]; n + 1];
    f[0][0] = 0.0;
    for i in 0..n {
        for a in 0..l {
            if !f[i][a].is_finite() {
                continue;
            }
            for b in a + 1..=l {
                if i == n - 1 && b != l {
                    continue;
                }
                let node = &nodes[order[i]];
                if let Some(e) = stage_eval(c, node, a, b, i == 0, i == n - 1) {
                    let hop = if i > 0 { node.link_rtt_ms / 1000.0 } else { 0.0 };
                    let t = f[i][a] + e.time_s + hop;
                    if t < f[i + 1][b] {
                        f[i + 1][b] = t;
                        back[i + 1][b] = a;
                    }
                }
            }
        }
    }
    if !f[n][l].is_finite() {
        return None;
    }
    let mut ranges = vec![(0, 0); n];
    let mut b = l;
    for i in (1..=n).rev() {
        let a = back[i][b];
        ranges[i - 1] = (a, b);
        b = a;
    }
    let evals: Vec<Eval> = ranges
        .iter()
        .enumerate()
        .map(|(i, &(a, b))| stage_eval(c, &nodes[order[i]], a, b, i == 0, i == n - 1).unwrap())
        .collect();
    Some((f[n][l], ranges, evals))
}

fn perms(n: usize) -> Vec<Vec<usize>> {
    if n == 1 {
        return vec![vec![0]];
    }
    let mut out = vec![];
    for p in perms(n - 1) {
        for i in 0..=p.len() {
            let mut q = p.clone();
            q.insert(i, n - 1);
            out.push(q);
        }
    }
    out
}

fn die(m: &str) -> ! {
    eprintln!("error: {m}");
    exit(2)
}

fn main() {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| v.iter().position(|x| x == k).and_then(|i| v.get(i + 1)).cloned();
    let rep = v.iter().any(|x| x == "--replicate-embed-last");
    let m: Manifest = serde_json::from_slice(
        &std::fs::read(get("--manifest").unwrap_or_else(|| die("missing --manifest"))).unwrap_or_else(|e| die(&e.to_string())),
    )
    .unwrap_or_else(|e| die(&format!("manifest: {e}")));
    let nf: NodesFile = serde_json::from_slice(
        &std::fs::read(get("--nodes").unwrap_or_else(|| die("missing --nodes"))).unwrap_or_else(|e| die(&e.to_string())),
    )
    .unwrap_or_else(|e| die(&format!("nodes: {e}")));
    let s = m.layer_sizes();
    let l = s.dense.len();
    let routed_total: u64 = s.routed.iter().sum();
    let active = match get("--experts") {
        Some(x) => {
            let (a, b) = x.split_once('/').unwrap_or_else(|| die("--experts must look like 8/288"));
            a.parse::<f64>().unwrap_or_else(|_| die("bad --experts")) / b.parse::<f64>().unwrap_or_else(|_| die("bad --experts"))
        }
        None if routed_total > 0 => die("manifest has routed-expert tensors: pass --experts ACTIVE/TOTAL (e.g. 8/288)"),
        None => 1.0,
    };
    let mut pd = vec![0.0; l + 1];
    let mut pr = vec![0.0; l + 1];
    for i in 0..l {
        pd[i + 1] = pd[i] + s.dense[i] as f64;
        pr[i + 1] = pr[i] + s.routed[i] as f64;
    }
    let ctx = Ctx { pd, pr, embed: s.embed as f64, head: s.head as f64, extra: s.extra as f64, other: s.other as f64, active, rep_embed: rep, l };
    let nodes = nf.nodes;
    let orders: Vec<Vec<usize>> = match get("--order") {
        Some(o) => vec![o
            .split(',')
            .map(|n| nodes.iter().position(|x| x.name == n).unwrap_or_else(|| die(&format!("unknown node {n}"))))
            .collect()],
        None if nodes.len() <= 6 => perms(nodes.len()),
        None => die("more than 6 nodes: pass --order"),
    };
    let mut best: Option<(f64, Vec<usize>, Vec<(usize, usize)>, Vec<Eval>)> = None;
    for o in orders {
        if let Some((t, r, e)) = solve(&ctx, &nodes, &o) {
            if best.as_ref().map_or(true, |b| t < b.0) {
                best = Some((t, o, r, e));
            }
        }
    }
    let Some((t, order, ranges, evals)) = best else {
        let need: f64 = (ctx.pd[l] + ctx.pr[l] + ctx.embed + ctx.head + ctx.extra + ctx.other) / 1e9;
        let have: f64 = nodes.iter().map(|n| n.tiers.iter().map(|t| t.gb).sum::<f64>() - n.reserve_gb).sum();
        die(&format!("no feasible plan: model needs {need:.1} GB, nodes offer {have:.1} GB after reserves (or a stage cannot hold one layer)"));
    };
    let n = order.len();
    let mut stages = vec![];
    for i in 0..n {
        let node = &nodes[order[i]];
        let (a, b) = ranges[i];
        let tb: serde_json::Map<String, serde_json::Value> =
            node.tiers.iter().zip(&evals[i].tier_bytes).map(|(t, x)| (t.name.clone(), json!(*x as u64))).collect();
        eprintln!(
            "stage {i}: {:<10} layers {a:>3}..{b:<3} {}  est {:.1} ms/tok{}",
            node.name,
            node.tiers.iter().zip(&evals[i].tier_bytes).map(|(t, x)| format!("{} {:.1} GB", t.name, x / 1e9)).collect::<Vec<_>>().join(", "),
            evals[i].time_s * 1e3,
            if i == 0 { "  [embed]" } else if i == n - 1 { "  [head+MTP]" } else { "" },
        );
        stages.push(json!({
            "node": node.name, "layers": [a, b],
            "embed": i == 0 || (rep && i == n - 1),
            "head": i == n - 1, "extra": i == n - 1,
            "tier_bytes": tb, "est_ms": evals[i].time_s * 1e3
        }));
    }
    eprintln!(
        "estimated decode: {:.1} ms/token ({:.1} tok/s) without speculation; bandwidth-bound model, measure before trusting",
        t * 1e3,
        1.0 / t
    );
    let plan = json!({
        "swp_plan": 1, "manifest_hash": m.manifest_hash, "n_layers": l,
        "order": order.iter().map(|&i| nodes[i].name.clone()).collect::<Vec<_>>(),
        "stages": stages, "est_decode_ms_per_token": t * 1e3, "est_tok_s": 1.0 / t,
        "notes": ["KV cache and workspace are only covered by reserve_gb; verify with a real load",
                  "layers >= n_layers (MTP/drafter) are assigned to the last stage"]
    });
    let js = serde_json::to_string_pretty(&plan).unwrap();
    match get("--out") {
        Some(p) => std::fs::write(p, js).unwrap(),
        None => println!("{js}"),
    }
}
