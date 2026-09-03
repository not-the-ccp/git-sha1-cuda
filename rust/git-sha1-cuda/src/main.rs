use std::{
    env,
    error::Error,
    ffi::OsString,
    fs,
    io::{self, Read, Write},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use git_sha1_cuda::{
    device_count, device_info, Context, GitJob, PrintableHeaderJob, SearchResult, TargetPrefix,
};

const RAW_OUTER_BATCH: u64 = 1 << 22;
const RAW_OUTER_DOMAIN: u64 = 1 << 32;
const PRINTABLE_BATCH: u64 = 1 << 30;
const PRINTABLE_DOMAIN: u64 = 1 << 40;

struct CommitArgs {
    prefix: String,
    messages: Vec<String>,
    message_file: Option<OsString>,
    device: i32,
    carrier: CommitCarrier,
    update_ref: bool,
    allow_empty: bool,
    amend: bool,
    start_epoch: u64,
}

#[derive(Clone, Copy)]
enum CommitCarrier {
    Header,
    Trailer,
}

impl CommitCarrier {
    fn eligible_fraction(self) -> f64 {
        match self {
            Self::Header => 1.0,
            Self::Trailer => (255.0_f64 / 256.0).powi(5),
        }
    }
}

enum PreparedJob {
    Printable(PrintableHeaderJob),
    Raw(GitJob),
}

impl PreparedJob {
    fn create_context(&self, device: i32) -> Result<Context, Box<dyn Error>> {
        Ok(match self {
            Self::Printable(job) => job.create_context(device)?,
            Self::Raw(job) => job.create_context(device)?,
        })
    }

    fn configure_context(&self, context: &mut Context) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Printable(job) => job.configure_context(context)?,
            Self::Raw(job) => {
                job.configure_context_with_policy(context, git_sha1_cuda::NoncePolicy::NoNul)?
            }
        }
        Ok(())
    }

    fn batch_size(&self) -> u64 {
        match self {
            Self::Printable(_) => PRINTABLE_BATCH,
            Self::Raw(_) => RAW_OUTER_BATCH,
        }
    }

    fn domain_size(&self) -> u64 {
        match self {
            Self::Printable(_) => PRINTABLE_DOMAIN,
            Self::Raw(_) => RAW_OUTER_DOMAIN,
        }
    }

    fn search(
        &self,
        context: &mut Context,
        base: u64,
        count: u64,
    ) -> Result<SearchResult, Box<dyn Error>> {
        Ok(match self {
            Self::Printable(_) => context.search_masked_header(base, count)?,
            Self::Raw(_) => context.search(base, count)?,
        })
    }

    fn verify_candidate(&self, candidate: u64) -> Result<[u8; 20], Box<dyn Error>> {
        Ok(match self {
            Self::Printable(job) => job.verify_candidate(candidate)?,
            Self::Raw(job) => job.verify_candidate(candidate)?,
        })
    }

    fn materialize_payload(&self, candidate: u64) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(match self {
            Self::Printable(job) => job.materialize_payload(candidate)?,
            Self::Raw(job) => job.materialize_payload(candidate)?,
        })
    }
}

fn usage() -> &'static str {
    concat!(
        "git-sha1-cuda ",
        env!("CARGO_PKG_VERSION"),
        r#"
Create an unsigned Git commit whose SHA-1 begins with a chosen prefix.

USAGE:
    git-sha1-cuda commit --prefix HEX -m MESSAGE [OPTIONS]
    git-sha1-cuda devices
    git-sha1-cuda benchmark [--device N]

OPTIONS:
    -p, --prefix HEX     Required leading hexadecimal digits (1 to 12)
    -m, --message TEXT   Commit message; repeat for additional paragraphs
    -F, --file PATH      Read the commit message from PATH, or stdin with -
        --carrier TYPE   Nonce location: header or trailer [default: header]
        --device N       CUDA device index [default: 0]
        --no-update-ref  Write the commit object without advancing HEAD
        --allow-empty    Create a commit when the index tree is unchanged
        --amend          Replace HEAD while preserving its author and parents
        --start-epoch N  Begin with nonce epoch N [default: 0]
    -h, --help           Print help
    -V, --version        Print version
"#
    )
}

fn parse_benchmark_args() -> Result<i32, String> {
    let mut args = env::args_os().skip(2);
    let mut device = 0;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--device") => {
                device = value(&mut args, "--device")?
                    .parse()
                    .map_err(|_| "--device must be an integer".to_owned())?;
            }
            Some("-h" | "--help") => {
                print!(
                    r#"Measure commit-search throughput on both nonce carriers.

USAGE:
    git-sha1-cuda benchmark [--device N]

OPTIONS:
        --device N  CUDA device index [default: 0]
    -h, --help      Print help
"#
                );
                std::process::exit(0);
            }
            _ => {
                return Err(format!(
                    "unknown benchmark option: {}",
                    arg.to_string_lossy()
                ))
            }
        }
    }
    Ok(device)
}

fn parse_args() -> Result<CommitArgs, String> {
    let mut args = env::args_os().skip(1);
    match args.next().as_deref() {
        Some(command) if command == "commit" => {}
        Some(flag) if flag == "-h" || flag == "--help" => {
            print!("{}", usage());
            std::process::exit(0);
        }
        Some(flag) if flag == "-V" || flag == "--version" => {
            println!("git-sha1-cuda {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        Some(command) => return Err(format!("unknown command: {}", command.to_string_lossy())),
        None => return Err("missing command".to_owned()),
    }

    let mut prefix = None;
    let mut messages = Vec::new();
    let mut message_file = None;
    let mut device = 0;
    let mut carrier = CommitCarrier::Header;
    let mut update_ref = true;
    let mut allow_empty = false;
    let mut amend = false;
    let mut start_epoch = 0;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-p" | "--prefix") => {
                prefix = Some(value(&mut args, "--prefix")?);
            }
            Some("-m" | "--message") => messages.push(value(&mut args, "--message")?),
            Some("-F" | "--file") => {
                message_file = Some(value_os(&mut args, "--file")?);
            }
            Some("--device") => {
                device = value(&mut args, "--device")?
                    .parse()
                    .map_err(|_| "--device must be an integer".to_owned())?;
            }
            Some("--carrier") => {
                carrier = match value(&mut args, "--carrier")?.as_str() {
                    "header" => CommitCarrier::Header,
                    "trailer" => CommitCarrier::Trailer,
                    _ => return Err("--carrier must be header or trailer".to_owned()),
                };
            }
            Some("--no-update-ref") => update_ref = false,
            Some("--allow-empty") => allow_empty = true,
            Some("--amend") => amend = true,
            Some("--start-epoch") => {
                start_epoch = value(&mut args, "--start-epoch")?
                    .parse()
                    .map_err(|_| "--start-epoch must be an unsigned integer".to_owned())?;
            }
            Some("-h" | "--help") => {
                print!("{}", usage());
                std::process::exit(0);
            }
            Some("-V" | "--version") => {
                println!("git-sha1-cuda {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option: {}", arg.to_string_lossy())),
        }
    }
    let prefix = prefix.ok_or_else(|| "--prefix is required".to_owned())?;
    let digits = prefix.strip_prefix("0x").unwrap_or(&prefix);
    if digits.is_empty() || digits.len() > 12 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("--prefix must contain 1 to 12 hexadecimal digits".to_owned());
    }
    if !messages.is_empty() && message_file.is_some() {
        return Err("-m/--message and -F/--file cannot be combined".to_owned());
    }
    if messages.is_empty() && message_file.is_none() {
        return Err("-m/--message or -F/--file is required".to_owned());
    }
    Ok(CommitArgs {
        prefix,
        messages,
        message_file,
        device,
        carrier,
        update_ref,
        allow_empty,
        amend,
        start_epoch,
    })
}

fn value_os(args: &mut impl Iterator<Item = OsString>, option: &str) -> Result<OsString, String> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn read_message(args: &CommitArgs) -> Result<Vec<u8>, Box<dyn Error>> {
    let message = if let Some(path) = &args.message_file {
        if path == "-" {
            let mut bytes = Vec::new();
            io::stdin().read_to_end(&mut bytes)?;
            bytes
        } else {
            fs::read(path)?
        }
    } else {
        args.messages.join("\n\n").into_bytes()
    };
    if message.is_empty() {
        return Err("commit message is empty".into());
    }
    if message.contains(&0) {
        return Err("commit message contains NUL".into());
    }
    Ok(message)
}

fn value(args: &mut impl Iterator<Item = OsString>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value"))?
        .into_string()
        .map_err(|_| format!("{option} must be valid UTF-8"))
}

fn git_bytes(arguments: &[&str]) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = Command::new("git").args(arguments).output()?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git {} failed: {}", arguments.join(" "), detail.trim()).into());
    }
    Ok(output.stdout)
}

fn git_output(arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    Ok(String::from_utf8(git_bytes(arguments)?)?
        .trim_end()
        .to_owned())
}

fn git_optional(arguments: &[&str]) -> Result<Option<String>, Box<dyn Error>> {
    let output = Command::new("git").args(arguments).output()?;
    if output.status.success() {
        Ok(Some(
            String::from_utf8(output.stdout)?.trim_end().to_owned(),
        ))
    } else {
        Ok(None)
    }
}

fn git_input(arguments: &[&str], input: &[u8]) -> Result<String, Box<dyn Error>> {
    let mut child = Command::new("git")
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(input)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git {} failed: {}", arguments.join(" "), detail.trim()).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim_end().to_owned())
}

fn commit_payload(
    message: &[u8],
    allow_empty: bool,
    amend: bool,
) -> Result<(Vec<u8>, Option<String>), Box<dyn Error>> {
    git_output(&["rev-parse", "--git-dir"])?;
    let format = git_output(&["rev-parse", "--show-object-format"])?;
    if format != "sha1" {
        return Err(format!("repository uses {format} objects; SHA-1 is required").into());
    }

    let tree = git_output(&["write-tree"])?;
    let head = git_optional(&["rev-parse", "--verify", "HEAD"])?;
    if amend && head.is_none() {
        return Err("--amend requires an existing HEAD commit".into());
    }
    if !allow_empty && !amend {
        if let Some(head_id) = &head {
            let parent_tree = git_output(&["show", "-s", "--format=%T", head_id])?;
            if parent_tree == tree {
                return Err("the index has no changes to commit".into());
            }
        }
    }
    let (author, parents) = if amend {
        let old_payload = git_bytes(&["cat-file", "commit", head.as_ref().unwrap()])?;
        let old_headers = old_payload
            .windows(2)
            .position(|bytes| bytes == b"\n\n")
            .map_or(old_payload.as_slice(), |offset| &old_payload[..offset]);
        let author = old_headers
            .split(|byte| *byte == b'\n')
            .find_map(|line| line.strip_prefix(b"author "))
            .ok_or("HEAD commit has no author header")?
            .to_vec();
        let parents = old_headers
            .split(|byte| *byte == b'\n')
            .filter_map(|line| line.strip_prefix(b"parent ").map(|parent| parent.to_vec()))
            .collect::<Vec<_>>();
        (author, parents)
    } else {
        (
            git_output(&["var", "GIT_AUTHOR_IDENT"])?.into_bytes(),
            head.iter()
                .map(|parent| parent.as_bytes().to_vec())
                .collect(),
        )
    };
    let committer = git_output(&["var", "GIT_COMMITTER_IDENT"])?;
    let mut payload = format!("tree {tree}\n").into_bytes();
    for parent_id in parents {
        payload.extend_from_slice(b"parent ");
        payload.extend_from_slice(&parent_id);
        payload.push(b'\n');
    }
    payload.extend_from_slice(b"author ");
    payload.extend_from_slice(&author);
    payload.extend_from_slice(format!("\ncommitter {committer}\n\n").as_bytes());
    payload.extend_from_slice(message);
    if !payload.ends_with(b"\n") {
        payload.push(b'\n');
    }
    Ok((payload, head))
}

fn hex_digest(digest: &[u8; 20]) -> String {
    let mut text = String::with_capacity(40);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut text, "{byte:02x}").unwrap();
    }
    text
}

fn prepare_job(
    payload: &[u8],
    target: TargetPrefix,
    carrier: CommitCarrier,
    epoch: u64,
) -> Result<PreparedJob, Box<dyn Error>> {
    Ok(match (carrier, epoch) {
        (CommitCarrier::Header, 0) => {
            PreparedJob::Printable(PrintableHeaderJob::header(payload, target)?)
        }
        (CommitCarrier::Header, epoch) => {
            PreparedJob::Printable(PrintableHeaderJob::header_epoch(payload, target, epoch)?)
        }
        (CommitCarrier::Trailer, 0) => PreparedJob::Raw(GitJob::message_trailer(payload, target)?),
        (CommitCarrier::Trailer, epoch) => {
            PreparedJob::Raw(GitJob::message_trailer_epoch(payload, target, epoch)?)
        }
    })
}

fn list_devices() -> Result<(), Box<dyn Error>> {
    let count = device_count()?;
    if count == 0 {
        println!("No CUDA devices found.");
        return Ok(());
    }
    for index in 0..count {
        let info = device_info(index as i32)?;
        println!(
            "{}  {}  sm_{}{}  {} SMs  {:.1} GiB",
            info.index,
            info.name,
            info.compute_major,
            info.compute_minor,
            info.multiprocessor_count,
            info.global_memory_bytes as f64 / (1_u64 << 30) as f64,
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct BenchmarkResult {
    carrier: CommitCarrier,
    billions_per_second: f64,
}

fn benchmark_carrier(
    device: i32,
    carrier: CommitCarrier,
) -> Result<BenchmarkResult, Box<dyn Error>> {
    const PAYLOAD: &[u8] = b"tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\
author Benchmark <benchmark@localhost> 0 +0000\n\
committer Benchmark <benchmark@localhost> 0 +0000\n\n\
benchmark\n";
    const SAMPLES: u64 = 3;

    let target = TargetPrefix::from_hex("ffffffffffffffffffffffffffffffffffffffff")?;
    let job = prepare_job(PAYLOAD, target, carrier, 0)?;
    let mut context = job.create_context(device)?;
    let batch = job.batch_size();

    // Warm up CUDA context initialization, code loading, and clock ramp-up.
    let _ = job.search(&mut context, 0, batch)?;

    let mut candidates = 0_u64;
    let mut milliseconds = 0.0_f64;
    for sample in 0..SAMPLES {
        let base = sample * batch;
        let result = job.search(&mut context, base, batch)?;
        if result.candidate.is_some() {
            return Err("benchmark unexpectedly found the 160-bit sentinel target".into());
        }
        candidates = candidates.saturating_add(result.candidates_hashed);
        milliseconds += f64::from(result.milliseconds);
    }
    Ok(BenchmarkResult {
        carrier,
        billions_per_second: candidates as f64 / milliseconds / 1.0e6,
    })
}

fn human_duration(seconds: f64) -> String {
    if seconds < 0.001 {
        format!("{:.0} us", seconds * 1_000_000.0)
    } else if seconds < 1.0 {
        format!("{:.1} ms", seconds * 1_000.0)
    } else if seconds < 60.0 {
        format!("{seconds:.1} s")
    } else if seconds < 3_600.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{:.1} h", seconds / 3_600.0)
    }
}

fn run_benchmark(device: i32) -> Result<(), Box<dyn Error>> {
    let info = device_info(device)?;
    eprintln!("Benchmarking {} (CUDA device {})...", info.name, device);
    let header = benchmark_carrier(device, CommitCarrier::Header)?;
    let trailer = benchmark_carrier(device, CommitCarrier::Trailer)?;
    let effective_rate =
        |result: BenchmarkResult| result.billions_per_second * result.carrier.eligible_fraction();

    println!("\nThroughput");
    println!("  printable header  {:>7.2} GH/s", effective_rate(header));
    println!("  message trailer   {:>7.2} GH/s", effective_rate(trailer));
    println!("\nAverage search time");
    println!("  prefix       header      trailer");
    for digits in 7..=12 {
        let candidates = 16.0_f64.powi(digits);
        let header_time = candidates / (effective_rate(header) * 1.0e9);
        let trailer_time = candidates / (effective_rate(trailer) * 1.0e9);
        println!(
            "  {digits:>2} hex   {:>10}   {:>10}",
            human_duration(header_time),
            human_duration(trailer_time)
        );
    }
    println!("\nActual searches vary randomly; half finish within 69% of the average.");
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut top_level = env::args_os().skip(1);
    match top_level.next().as_deref() {
        Some(command) if command == "devices" => {
            if top_level.next().is_some() {
                return Err("devices takes no arguments".into());
            }
            return list_devices();
        }
        Some(command) if command == "benchmark" => {
            let device =
                parse_benchmark_args().map_err(|message| format!("{message}\n\n{}", usage()))?;
            return run_benchmark(device);
        }
        _ => {}
    }
    let args = parse_args().map_err(|message| format!("{message}\n\n{}", usage()))?;
    let message = read_message(&args)?;
    let (payload, parent) = commit_payload(&message, args.allow_empty, args.amend)?;
    let target = TargetPrefix::from_hex(&args.prefix)?;
    let mut epoch = args.start_epoch;
    let mut job = prepare_job(&payload, target, args.carrier, epoch)?;
    let mut context = job.create_context(args.device)?;

    eprintln!(
        "Searching CUDA device {} for prefix {}...",
        args.device, args.prefix
    );
    let started = Instant::now();
    let mut last_progress = started;
    let mut search_base = 0;
    let mut total_hashed = 0_u64;
    let expected_hashes = 2.0_f64.powi(target.bits() as i32) / args.carrier.eligible_fraction();
    let candidate = loop {
        let search_count = job.batch_size().min(job.domain_size() - search_base);
        let result = job.search(&mut context, search_base, search_count)?;
        if let Some(candidate) = result.candidate {
            break candidate;
        }
        total_hashed = total_hashed.saturating_add(result.candidates_hashed);
        search_base += search_count;
        if search_base == job.domain_size() {
            epoch = epoch.checked_add(1).ok_or("nonce epoch overflow")?;
            job = prepare_job(&payload, target, args.carrier, epoch)?;
            job.configure_context(&mut context)?;
            search_base = 0;
            eprintln!("  continuing with nonce epoch {epoch}");
        }
        if last_progress.elapsed() >= Duration::from_secs(1) {
            let match_probability = -(-(total_hashed as f64) / expected_hashes).exp_m1();
            eprintln!(
                "  {:.1}% match probability ({:.2} GH/s, epoch {epoch})",
                match_probability * 100.0,
                total_hashed as f64 / started.elapsed().as_secs_f64() / 1e9
            );
            last_progress = Instant::now();
        }
    };

    let digest = job.verify_candidate(candidate)?;
    let expected_id = hex_digest(&digest);
    let finished_payload = job.materialize_payload(candidate)?;
    let object_id = git_input(
        &["hash-object", "-t", "commit", "-w", "--stdin"],
        &finished_payload,
    )?;
    if object_id != expected_id {
        return Err(
            format!("Git wrote {object_id}, but verification produced {expected_id}").into(),
        );
    }

    let subject_bytes = message
        .split(|byte| *byte == b'\n')
        .next()
        .unwrap_or_default();
    let subject = String::from_utf8_lossy(subject_bytes);
    let reflog = format!("commit: {subject}");
    if args.update_ref {
        let old_id =
            parent.unwrap_or_else(|| "0000000000000000000000000000000000000000".to_owned());
        git_output(&["update-ref", "-m", &reflog, "HEAD", &object_id, &old_id])?;
    }
    eprintln!("Found in {:.2?}", started.elapsed());
    if args.update_ref {
        println!("[{object_id}] {subject}");
    } else {
        println!("{object_id}");
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
