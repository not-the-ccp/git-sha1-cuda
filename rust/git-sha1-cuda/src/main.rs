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
const RAW_OUTER_START: u64 = 0x0101_0101;
const PRINTABLE_BATCH: u64 = 1 << 30;
const PRINTABLE_DOMAIN: u64 = 1 << 40;
type IdentParts<'a> = (&'a [u8], &'a [u8]);

fn candidate_batch_size(target_bits: u32) -> u64 {
    let exponent = target_bits.saturating_sub(2).clamp(20, 30);
    1_u64 << exponent
}

struct CommitArgs {
    prefix: String,
    messages: Vec<String>,
    message_file: Option<OsString>,
    trailers: Vec<String>,
    signoff: bool,
    device: i32,
    carrier: CommitCarrier,
    update_ref: bool,
    allow_empty: bool,
    amend: bool,
    author: Option<String>,
    author_date: Option<String>,
    reset_author: bool,
    start_epoch: u64,
    resume: Option<String>,
}

struct CommitPayload {
    bytes: Vec<u8>,
    expected_head: Option<String>,
    merging: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommitCarrier {
    Header,
    Trailer,
}

impl CommitCarrier {
    fn name(self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::Trailer => "trailer",
        }
    }

    fn eligible_fraction(self) -> f64 {
        match self {
            Self::Header => 1.0,
            Self::Trailer => (255.0_f64 / 256.0).powi(5),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ResumePoint {
    carrier: CommitCarrier,
    prefix: String,
    epoch: u64,
    offset: u64,
    author_date: String,
    committer_date: String,
    payload_id: String,
}

fn normalized_prefix(prefix: &str) -> String {
    prefix
        .strip_prefix("0x")
        .unwrap_or(prefix)
        .to_ascii_lowercase()
}

fn valid_timezone(text: &str) -> bool {
    text.len() == 5
        && matches!(text.as_bytes()[0], b'+' | b'-')
        && text.as_bytes()[1..].iter().all(u8::is_ascii_digit)
}

fn parse_resume_token(text: &str) -> Result<ResumePoint, String> {
    let fields: Vec<_> = text.split(':').collect();
    if fields.len() != 10 || fields[0] != "gsv1" {
        return Err("invalid resume token".to_owned());
    }
    let carrier = match fields[1] {
        "header" => CommitCarrier::Header,
        "trailer" => CommitCarrier::Trailer,
        _ => return Err("resume token has an unknown carrier".to_owned()),
    };
    let prefix = normalized_prefix(fields[2]);
    if prefix.is_empty()
        || prefix.len() > 12
        || !prefix.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("resume token has an invalid prefix".to_owned());
    }
    let epoch = fields[3]
        .parse()
        .map_err(|_| "resume token has an invalid epoch".to_owned())?;
    let offset = fields[4]
        .parse()
        .map_err(|_| "resume token has an invalid offset".to_owned())?;
    if fields[5].parse::<i64>().is_err() || !valid_timezone(fields[6]) {
        return Err("resume token has an invalid author date".to_owned());
    }
    if fields[7].parse::<i64>().is_err() || !valid_timezone(fields[8]) {
        return Err("resume token has an invalid committer date".to_owned());
    }
    if fields[9].len() != 40 || !fields[9].bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("resume token has an invalid payload fingerprint".to_owned());
    }
    Ok(ResumePoint {
        carrier,
        prefix,
        epoch,
        offset,
        author_date: format!("{} {}", fields[5], fields[6]),
        committer_date: format!("{} {}", fields[7], fields[8]),
        payload_id: fields[9].to_ascii_lowercase(),
    })
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

    fn maximum_batch_size(&self) -> u64 {
        match self {
            Self::Printable(_) => PRINTABLE_BATCH,
            Self::Raw(_) => RAW_OUTER_BATCH,
        }
    }

    fn batch_size(&self, target_bits: u32) -> u64 {
        let candidates = candidate_batch_size(target_bits);
        match self {
            Self::Printable(_) => candidates,
            Self::Raw(_) => candidates / 256,
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
            Self::Raw(_) => context.search(raw_outer_base(base), count)?,
        })
    }

    fn contiguous_count(&self, base: u64, requested: u64) -> u64 {
        match self {
            Self::Printable(_) => requested,
            Self::Raw(_) => requested.min(RAW_OUTER_DOMAIN - raw_outer_base(base)),
        }
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

fn raw_outer_base(logical_base: u64) -> u64 {
    (logical_base + RAW_OUTER_START) & (RAW_OUTER_DOMAIN - 1)
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
        --trailer T=V    Add a Git message trailer; repeat as needed
    -s, --signoff        Add Signed-off-by for the committer
        --carrier TYPE   Nonce location: header or trailer [default: header]
        --device N       CUDA device index [default: 0]
        --no-update-ref  Write the commit object without advancing HEAD
        --allow-empty    Create a commit when the index tree is unchanged
        --amend          Replace HEAD while preserving its author and parents
        --author IDENT   Set the author as "Name <email>"
        --date DATE      Set the author date using a Git date expression
        --reset-author   Reset author identity and date when amending
        --start-epoch N  Begin with nonce epoch N [default: 0]
        --resume TOKEN   Resume from a token printed by an earlier search
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
    let mut trailers = Vec::new();
    let mut signoff = false;
    let mut device = 0;
    let mut carrier = CommitCarrier::Header;
    let mut update_ref = true;
    let mut allow_empty = false;
    let mut amend = false;
    let mut author = None;
    let mut author_date = None;
    let mut reset_author = false;
    let mut start_epoch = 0;
    let mut start_epoch_set = false;
    let mut resume = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-p" | "--prefix") => {
                prefix = Some(value(&mut args, "--prefix")?);
            }
            Some("-m" | "--message") => messages.push(value(&mut args, "--message")?),
            Some("-F" | "--file") => {
                message_file = Some(value_os(&mut args, "--file")?);
            }
            Some("--trailer") => trailers.push(value(&mut args, "--trailer")?),
            Some("-s" | "--signoff") => signoff = true,
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
            Some("--author") => author = Some(value(&mut args, "--author")?),
            Some("--date") => author_date = Some(value(&mut args, "--date")?),
            Some("--reset-author") => reset_author = true,
            Some("--start-epoch") => {
                start_epoch = value(&mut args, "--start-epoch")?
                    .parse()
                    .map_err(|_| "--start-epoch must be an unsigned integer".to_owned())?;
                start_epoch_set = true;
            }
            Some("--resume") => resume = Some(value(&mut args, "--resume")?),
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
    if reset_author && !amend {
        return Err("--reset-author requires --amend".to_owned());
    }
    if reset_author && (author.is_some() || author_date.is_some()) {
        return Err("--reset-author cannot be combined with --author or --date".to_owned());
    }
    if resume.is_some() && start_epoch_set {
        return Err("--resume cannot be combined with --start-epoch".to_owned());
    }
    Ok(CommitArgs {
        prefix,
        messages,
        message_file,
        trailers,
        signoff,
        device,
        carrier,
        update_ref,
        allow_empty,
        amend,
        author,
        author_date,
        reset_author,
        start_epoch,
        resume,
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

fn git_bytes_with_env(
    arguments: &[&str],
    variables: &[(&str, &str)],
) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = Command::new("git")
        .args(arguments)
        .envs(variables.iter().copied())
        .output()?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git {} failed: {}", arguments.join(" "), detail.trim()).into());
    }
    Ok(output.stdout)
}

fn parse_author_identity(author: &str) -> Result<(&str, &str), Box<dyn Error>> {
    if author.contains(['\0', '\n', '\r']) || !author.ends_with('>') {
        return Err("--author must have the form Name <email>".into());
    }
    let separator = author
        .rfind(" <")
        .ok_or("--author must have the form Name <email>")?;
    let name = &author[..separator];
    let email = &author[separator + 2..author.len() - 1];
    if name.is_empty() || email.is_empty() || email.contains(['<', '>']) {
        return Err("--author must have the form Name <email>".into());
    }
    Ok((name, email))
}

fn current_author_ident(
    author: Option<&str>,
    date: Option<&str>,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut variables = Vec::with_capacity(3);
    if let Some(author) = author {
        let (name, email) = parse_author_identity(author)?;
        variables.push(("GIT_AUTHOR_NAME", name));
        variables.push(("GIT_AUTHOR_EMAIL", email));
    }
    if let Some(date) = date {
        if date.contains(['\0', '\n', '\r']) {
            return Err("--date contains an invalid character".into());
        }
        variables.push(("GIT_AUTHOR_DATE", date));
    }
    let output = git_bytes_with_env(&["var", "GIT_AUTHOR_IDENT"], &variables)?;
    Ok(output.strip_suffix(b"\n").unwrap_or(&output).to_vec())
}

fn current_committer_ident(date: Option<&str>) -> Result<Vec<u8>, Box<dyn Error>> {
    let variables = date.map_or_else(Vec::new, |date| vec![("GIT_COMMITTER_DATE", date)]);
    let output = git_bytes_with_env(&["var", "GIT_COMMITTER_IDENT"], &variables)?;
    Ok(output.strip_suffix(b"\n").unwrap_or(&output).to_vec())
}

fn apply_message_trailers(
    message: &[u8],
    trailers: &[String],
    signoff: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    if trailers.is_empty() && !signoff {
        return Ok(message.to_vec());
    }

    let mut additions = trailers.to_vec();
    if signoff {
        let committer = current_committer_ident(None)?;
        let (identity, _) = ident_parts(&committer)?;
        additions.push(format!(
            "Signed-off-by={}",
            String::from_utf8(identity.to_vec())?
        ));
    }
    let mut arguments = vec!["interpret-trailers"];
    for trailer in &additions {
        arguments.push("--trailer");
        arguments.push(trailer);
    }
    let mut input = message.to_vec();
    if !input.ends_with(b"\n") {
        input.push(b'\n');
    }
    git_input_bytes(&arguments, &input)
}

fn ident_parts(ident: &[u8]) -> Result<IdentParts<'_>, Box<dyn Error>> {
    let close = ident
        .iter()
        .rposition(|byte| *byte == b'>')
        .ok_or("author identity has no closing email bracket")?;
    if close + 2 > ident.len() || ident.get(close + 1) != Some(&b' ') {
        return Err("author identity has no date".into());
    }
    Ok((&ident[..=close], &ident[close + 2..]))
}

fn amended_author_ident(
    original: &[u8],
    author: Option<&str>,
    date: Option<&str>,
    reset: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    if reset {
        return current_author_ident(None, date);
    }
    if author.is_none() && date.is_none() {
        return Ok(original.to_vec());
    }
    let replacement = current_author_ident(author, date)?;
    let (old_identity, old_date) = ident_parts(original)?;
    let (new_identity, new_date) = ident_parts(&replacement)?;
    let mut result = Vec::new();
    result.extend_from_slice(if author.is_some() {
        new_identity
    } else {
        old_identity
    });
    result.push(b' ');
    result.extend_from_slice(if date.is_some() { new_date } else { old_date });
    Ok(result)
}

fn payload_ident_date(payload: &[u8], header: &[u8]) -> Result<String, Box<dyn Error>> {
    let header_end = payload
        .windows(2)
        .position(|bytes| bytes == b"\n\n")
        .ok_or("commit payload has no header separator")?;
    let value = payload[..header_end]
        .split(|byte| *byte == b'\n')
        .find_map(|line| line.strip_prefix(header))
        .ok_or("commit payload is missing an identity header")?;
    let (_, date) = ident_parts(value)?;
    Ok(String::from_utf8(date.to_vec())?)
}

fn resume_token(
    carrier: CommitCarrier,
    prefix: &str,
    epoch: u64,
    offset: u64,
    payload: &[u8],
    payload_id: &str,
) -> Result<String, Box<dyn Error>> {
    let author_date = payload_ident_date(payload, b"author ")?;
    let committer_date = payload_ident_date(payload, b"committer ")?;
    let (author_timestamp, author_timezone) = author_date
        .split_once(' ')
        .ok_or("author date has no timezone")?;
    let (committer_timestamp, committer_timezone) = committer_date
        .split_once(' ')
        .ok_or("committer date has no timezone")?;
    Ok(format!(
        "gsv1:{}:{}:{epoch}:{offset}:{author_timestamp}:{author_timezone}:{}:{}:{}",
        carrier.name(),
        normalized_prefix(prefix),
        committer_timestamp,
        committer_timezone,
        payload_id
    ))
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

fn git_input_bytes(arguments: &[&str], input: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
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
    Ok(output.stdout)
}

fn git_input(arguments: &[&str], input: &[u8]) -> Result<String, Box<dyn Error>> {
    Ok(String::from_utf8(git_input_bytes(arguments, input)?)?
        .trim_end()
        .to_owned())
}

fn git_path(name: &str) -> Result<std::path::PathBuf, Box<dyn Error>> {
    Ok(git_output(&["rev-parse", "--git-path", name])?.into())
}

fn merge_heads() -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let path = git_path("MERGE_HEAD")?;
    let contents = match fs::read(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut heads = Vec::new();
    for line in contents.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        if line.len() != 40 || !line.iter().all(u8::is_ascii_hexdigit) {
            return Err("MERGE_HEAD contains an invalid object ID".into());
        }
        let id = String::from_utf8(line.to_vec())?;
        let commit = format!("{id}^{{commit}}");
        git_output(&["rev-parse", "--verify", &commit])?;
        heads.push(line.to_vec());
    }
    if heads.is_empty() {
        return Err("MERGE_HEAD contains no commit IDs".into());
    }
    Ok(heads)
}

fn clear_merge_state() -> Result<(), Box<dyn Error>> {
    for name in ["MERGE_HEAD", "MERGE_MSG", "MERGE_MODE", "MERGE_RR"] {
        match fs::remove_file(git_path(name)?) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("could not remove {name}: {error}").into()),
        }
    }
    git_output(&["update-ref", "-d", "AUTO_MERGE"])?;
    Ok(())
}

fn commit_payload(
    message: &[u8],
    allow_empty: bool,
    amend: bool,
    author_override: Option<&str>,
    author_date: Option<&str>,
    reset_author: bool,
    committer_date: Option<&str>,
) -> Result<CommitPayload, Box<dyn Error>> {
    git_output(&["rev-parse", "--git-dir"])?;
    let format = git_output(&["rev-parse", "--show-object-format"])?;
    if format != "sha1" {
        return Err(format!("repository uses {format} objects; SHA-1 is required").into());
    }

    let tree = git_output(&["write-tree"])?;
    let head = git_optional(&["rev-parse", "--verify", "HEAD"])?;
    let merge_heads = if amend { Vec::new() } else { merge_heads()? };
    let merging = !merge_heads.is_empty();
    if merging && head.is_none() {
        return Err("MERGE_HEAD exists in a repository without HEAD".into());
    }
    if merging && git_path("MERGE_AUTOSTASH")?.try_exists()? {
        return Err("merge commits with MERGE_AUTOSTASH are not supported yet".into());
    }
    if amend && head.is_none() {
        return Err("--amend requires an existing HEAD commit".into());
    }
    if !allow_empty && !amend && !merging {
        if let Some(head_id) = &head {
            let parent_tree = git_output(&["show", "-s", "--format=%T", head_id])?;
            if parent_tree == tree {
                return Err("the index has no changes to commit".into());
            }
        }
    }
    let (author, mut parents) = if amend {
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
        (
            amended_author_ident(&author, author_override, author_date, reset_author)?,
            parents,
        )
    } else {
        (
            current_author_ident(author_override, author_date)?,
            head.iter()
                .map(|parent| parent.as_bytes().to_vec())
                .collect(),
        )
    };
    parents.extend(merge_heads);
    let committer = current_committer_ident(committer_date)?;
    let mut payload = format!("tree {tree}\n").into_bytes();
    for parent_id in parents {
        payload.extend_from_slice(b"parent ");
        payload.extend_from_slice(&parent_id);
        payload.push(b'\n');
    }
    payload.extend_from_slice(b"author ");
    payload.extend_from_slice(&author);
    payload.extend_from_slice(b"\ncommitter ");
    payload.extend_from_slice(&committer);
    payload.extend_from_slice(b"\n\n");
    payload.extend_from_slice(message);
    if !payload.ends_with(b"\n") {
        payload.push(b'\n');
    }
    Ok(CommitPayload {
        bytes: payload,
        expected_head: head,
        merging,
    })
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
    let batch = job.maximum_batch_size();

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
    println!("\nIdeal average at measured throughput");
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
    println!(
        "\nActual searches vary randomly; short searches also include batch and launch overhead."
    );
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
    let resume = args.resume.as_deref().map(parse_resume_token).transpose()?;
    if let Some(point) = &resume {
        if point.carrier != args.carrier {
            return Err(format!(
                "resume token uses the {} carrier; select it with --carrier {}",
                point.carrier.name(),
                point.carrier.name()
            )
            .into());
        }
        if point.prefix != normalized_prefix(&args.prefix) {
            return Err("resume token prefix does not match --prefix".into());
        }
    }
    let message = apply_message_trailers(&read_message(&args)?, &args.trailers, args.signoff)?;
    let author_date = resume
        .as_ref()
        .map_or(args.author_date.as_deref(), |point| {
            Some(point.author_date.as_str())
        });
    let committer_date = resume.as_ref().map(|point| point.committer_date.as_str());
    let CommitPayload {
        bytes: payload,
        expected_head: parent,
        merging,
    } = commit_payload(
        &message,
        args.allow_empty,
        args.amend,
        args.author.as_deref(),
        author_date,
        args.reset_author,
        committer_date,
    )?;
    let payload_id = git_input(&["hash-object", "-t", "commit", "--stdin"], &payload)?;
    if resume
        .as_ref()
        .is_some_and(|point| point.payload_id != payload_id)
    {
        return Err(
            "resume token does not match the prepared commit; repository state or commit options changed"
                .into(),
        );
    }
    let target = TargetPrefix::from_hex(&args.prefix)?;
    let mut epoch = resume
        .as_ref()
        .map_or(args.start_epoch, |point| point.epoch);
    let mut job = prepare_job(&payload, target, args.carrier, epoch)?;
    let mut search_base = resume.as_ref().map_or(0, |point| point.offset);
    if search_base >= job.domain_size() {
        return Err("resume token offset is outside the carrier domain".into());
    }
    let mut context = job.create_context(args.device)?;

    eprintln!(
        "Searching CUDA device {} for prefix {}...",
        args.device, args.prefix
    );
    let started = Instant::now();
    let mut last_progress = started;
    let mut total_hashed = 0_u64;
    let expected_hashes = 2.0_f64.powi(target.bits() as i32) / args.carrier.eligible_fraction();
    let prior_hashed = resume.as_ref().map_or(0.0, |point| {
        point.epoch as f64 * PRINTABLE_DOMAIN as f64
            + point.offset as f64
                * match args.carrier {
                    CommitCarrier::Header => 1.0,
                    CommitCarrier::Trailer => 256.0,
                }
    });
    let batch_size = job.batch_size(target.bits());
    let candidate = loop {
        let search_count =
            job.contiguous_count(search_base, batch_size.min(job.domain_size() - search_base));
        let result = job.search(&mut context, search_base, search_count)?;
        total_hashed = total_hashed.saturating_add(result.candidates_hashed);
        if let Some(candidate) = result.candidate {
            break candidate;
        }
        search_base += search_count;
        if search_base == job.domain_size() {
            epoch = epoch.checked_add(1).ok_or("nonce epoch overflow")?;
            job = prepare_job(&payload, target, args.carrier, epoch)?;
            job.configure_context(&mut context)?;
            search_base = 0;
            eprintln!("  continuing with nonce epoch {epoch}");
        }
        if last_progress.elapsed() >= Duration::from_secs(1) {
            let match_probability =
                -(-((prior_hashed + total_hashed as f64) / expected_hashes)).exp_m1();
            eprintln!(
                "  {:.1}% match probability ({:.2} GH/s, epoch {epoch})",
                match_probability * 100.0,
                total_hashed as f64 / started.elapsed().as_secs_f64() / 1e9
            );
            eprintln!(
                "  resume token: {}",
                resume_token(
                    args.carrier,
                    &args.prefix,
                    epoch,
                    search_base,
                    &payload,
                    &payload_id
                )?
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
        if merging {
            clear_merge_state()?;
        }
    }
    eprintln!(
        "Found in {:.2?} after {} candidates ({:.2} GH/s)",
        started.elapsed(),
        total_hashed,
        total_hashed as f64 / started.elapsed().as_secs_f64() / 1e9
    );
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

#[cfg(test)]
mod tests {
    use super::{
        amended_author_ident, apply_message_trailers, candidate_batch_size, ident_parts,
        parse_author_identity, parse_resume_token, raw_outer_base, resume_token, CommitCarrier,
        RAW_OUTER_DOMAIN, RAW_OUTER_START,
    };

    #[test]
    fn search_batches_scale_with_target_width() {
        assert_eq!(candidate_batch_size(4), 1 << 20);
        assert_eq!(candidate_batch_size(20), 1 << 20);
        assert_eq!(candidate_batch_size(24), 1 << 22);
        assert_eq!(candidate_batch_size(28), 1 << 26);
        assert_eq!(candidate_batch_size(32), 1 << 30);
        assert_eq!(candidate_batch_size(48), 1 << 30);
    }

    #[test]
    fn raw_domain_rotation_starts_with_nonzero_bytes_and_wraps() {
        assert_eq!(raw_outer_base(0), RAW_OUTER_START);
        assert_eq!(raw_outer_base(RAW_OUTER_DOMAIN - RAW_OUTER_START), 0);
        assert_eq!(raw_outer_base(RAW_OUTER_DOMAIN - 1), RAW_OUTER_START - 1);
    }

    #[test]
    fn resume_tokens_round_trip_search_and_payload_state() {
        let payload = b"tree 0000000000000000000000000000000000000000\n\
author Ada <ada@example.com> 1234567890 +0130\n\
committer Grace <grace@example.com> 1234567891 -0700\n\nmessage\n";
        let token = resume_token(
            CommitCarrier::Trailer,
            "0xAbCd",
            7,
            12345,
            payload,
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        let point = parse_resume_token(&token).unwrap();
        assert_eq!(point.carrier, CommitCarrier::Trailer);
        assert_eq!(point.prefix, "abcd");
        assert_eq!(point.epoch, 7);
        assert_eq!(point.offset, 12345);
        assert_eq!(point.author_date, "1234567890 +0130");
        assert_eq!(point.committer_date, "1234567891 -0700");
        assert_eq!(point.payload_id, "0123456789abcdef0123456789abcdef01234567");
    }

    #[test]
    fn resume_tokens_reject_malformed_state() {
        assert!(parse_resume_token("gsv1:header:0000000:0:1").is_err());
        assert!(parse_resume_token(
            "gsv1:unknown:0000000:0:1:1:+0000:1:+0000:0123456789abcdef0123456789abcdef01234567"
        )
        .is_err());
    }

    #[test]
    fn git_formats_repeated_message_trailers() {
        let result = apply_message_trailers(
            b"Subject\n\nBody\n",
            &[
                "Reviewed-by=Ada <ada@example.com>".to_owned(),
                "Acked-by=Grace <grace@example.com>".to_owned(),
            ],
            false,
        )
        .unwrap();
        assert_eq!(
            result,
            b"Subject\n\nBody\n\nReviewed-by: Ada <ada@example.com>\n\
Acked-by: Grace <grace@example.com>\n"
        );
    }

    #[test]
    fn git_avoids_an_identical_trailer() {
        let result = apply_message_trailers(
            b"Subject\n\nReviewed-by: Ada <ada@example.com>\n",
            &["Reviewed-by=Ada <ada@example.com>".to_owned()],
            false,
        )
        .unwrap();
        assert_eq!(result, b"Subject\n\nReviewed-by: Ada <ada@example.com>\n");
    }

    #[test]
    fn parses_explicit_author_identity() {
        assert_eq!(
            parse_author_identity("Ada Lovelace <ada@example.com>").unwrap(),
            ("Ada Lovelace", "ada@example.com")
        );
        assert!(parse_author_identity("Ada Lovelace").is_err());
        assert!(parse_author_identity("<ada@example.com>").is_err());
        assert!(parse_author_identity("Ada <>").is_err());
    }

    #[test]
    fn splits_git_identity_and_date() {
        assert_eq!(
            ident_parts(b"Ada Lovelace <ada@example.com> 1234567890 +0130").unwrap(),
            (
                b"Ada Lovelace <ada@example.com>".as_slice(),
                b"1234567890 +0130".as_slice()
            )
        );
    }

    #[test]
    fn trailer_block_is_separated_from_a_message_without_newline() {
        let result = apply_message_trailers(b"Subject", &["Issue=42".to_owned()], false).unwrap();
        assert_eq!(result, b"Subject\n\nIssue: 42\n");
    }

    #[test]
    fn amend_preserves_original_author_by_default() {
        let original = b"Original <original@example.com> 1234567890 +0130";
        assert_eq!(
            amended_author_ident(original, None, None, false).unwrap(),
            original
        );
    }
}
