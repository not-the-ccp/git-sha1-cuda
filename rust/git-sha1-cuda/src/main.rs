use std::{
    env,
    error::Error,
    ffi::OsString,
    fs,
    io::{self, Read, Write},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use git_sha1_cuda::{GitJob, TargetPrefix};

const DEFAULT_OUTER_BATCH: u64 = 1 << 22;
const OUTER_DOMAIN: u64 = 1 << 32;

struct CommitArgs {
    prefix: String,
    messages: Vec<String>,
    message_file: Option<OsString>,
    device: i32,
    carrier: CommitCarrier,
    update_ref: bool,
}

#[derive(Clone, Copy)]
enum CommitCarrier {
    Header,
    Trailer,
}

fn usage() -> &'static str {
    r#"git-sha1-cuda 0.1.0
Create an unsigned Git commit whose SHA-1 begins with a chosen prefix.

USAGE:
    git-sha1-cuda commit --prefix HEX -m MESSAGE [OPTIONS]

OPTIONS:
    -p, --prefix HEX     Required leading hexadecimal digits (1 to 10)
    -m, --message TEXT   Commit message; repeat for additional paragraphs
    -F, --file PATH      Read the commit message from PATH, or stdin with -
        --carrier TYPE   Nonce location: header or trailer [default: header]
        --device N       CUDA device index [default: 0]
        --no-update-ref  Write the commit object without advancing HEAD
    -h, --help           Print help
    -V, --version        Print version
"#
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
            println!("git-sha1-cuda 0.1.0");
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
            Some("-h" | "--help") => {
                print!("{}", usage());
                std::process::exit(0);
            }
            Some("-V" | "--version") => {
                println!("git-sha1-cuda 0.1.0");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option: {}", arg.to_string_lossy())),
        }
    }
    let prefix = prefix.ok_or_else(|| "--prefix is required".to_owned())?;
    let digits = prefix.strip_prefix("0x").unwrap_or(&prefix);
    if digits.is_empty() || digits.len() > 10 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("--prefix must contain 1 to 10 hexadecimal digits".to_owned());
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

fn git_output(arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git").args(arguments).output()?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git {} failed: {}", arguments.join(" "), detail.trim()).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim_end().to_owned())
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

fn commit_payload(message: &[u8]) -> Result<(Vec<u8>, Option<String>), Box<dyn Error>> {
    git_output(&["rev-parse", "--git-dir"])?;
    let format = git_output(&["rev-parse", "--show-object-format"])?;
    if format != "sha1" {
        return Err(format!("repository uses {format} objects; SHA-1 is required").into());
    }

    let tree = git_output(&["write-tree"])?;
    let parent = git_optional(&["rev-parse", "--verify", "HEAD"])?;
    if let Some(parent_id) = &parent {
        let parent_tree = git_output(&["show", "-s", "--format=%T", parent_id])?;
        if parent_tree == tree {
            return Err("the index has no changes to commit".into());
        }
    }
    let author = git_output(&["var", "GIT_AUTHOR_IDENT"])?;
    let committer = git_output(&["var", "GIT_COMMITTER_IDENT"])?;
    let mut payload = format!("tree {tree}\n").into_bytes();
    if let Some(parent_id) = &parent {
        payload.extend_from_slice(format!("parent {parent_id}\n").as_bytes());
    }
    payload.extend_from_slice(format!("author {author}\ncommitter {committer}\n\n").as_bytes());
    payload.extend_from_slice(message);
    if !payload.ends_with(b"\n") {
        payload.push(b'\n');
    }
    Ok((payload, parent))
}

fn hex_digest(digest: &[u8; 20]) -> String {
    let mut text = String::with_capacity(40);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut text, "{byte:02x}").unwrap();
    }
    text
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args().map_err(|message| format!("{message}\n\n{}", usage()))?;
    let message = read_message(&args)?;
    let (payload, parent) = commit_payload(&message)?;
    let target = TargetPrefix::from_hex(&args.prefix)?;
    let job = match args.carrier {
        CommitCarrier::Header => GitJob::header(&payload, target)?,
        CommitCarrier::Trailer => GitJob::message_trailer(&payload, target)?,
    };
    let mut context = job.create_context(args.device)?;

    eprintln!(
        "Searching CUDA device {} for prefix {}...",
        args.device, args.prefix
    );
    let started = Instant::now();
    let mut last_progress = started;
    let mut outer_base = 0;
    let candidate = loop {
        let outer_count = DEFAULT_OUTER_BATCH.min(OUTER_DOMAIN - outer_base);
        let result = context.search(outer_base, outer_count)?;
        if let Some(candidate) = result.candidate {
            break candidate;
        }
        outer_base += outer_count;
        if outer_base == OUTER_DOMAIN {
            return Err("the complete 40-bit candidate space contained no match".into());
        }
        if last_progress.elapsed() >= Duration::from_secs(1) {
            let candidates = outer_base.saturating_mul(256);
            eprintln!(
                "  {:.1}% searched ({:.2} GH/s)",
                outer_base as f64 * 100.0 / OUTER_DOMAIN as f64,
                candidates as f64 / started.elapsed().as_secs_f64() / 1e9
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
