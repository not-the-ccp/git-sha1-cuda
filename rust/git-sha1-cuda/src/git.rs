use std::{error::Error, fmt};

use crate::{sha1, Context, CudaError, Job, NoncePolicy};

const NONCE_BYTES: usize = 5;
const NONCE_BITS: u32 = 40;
const NONCE_BLOCK_OFFSET: usize = 48;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparationError(String);

impl PreparationError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for PreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for PreparationError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetPrefix {
    bits: u32,
    words: [u32; 5],
}

impl TargetPrefix {
    pub fn from_hex(text: &str) -> Result<Self, PreparationError> {
        let digits = text.strip_prefix("0x").unwrap_or(text);
        Self::from_hex_bits(digits, (digits.len() * 4) as u32)
    }

    pub fn from_hex_bits(text: &str, bits: u32) -> Result<Self, PreparationError> {
        let digits = text.strip_prefix("0x").unwrap_or(text);
        if digits.is_empty() || digits.len() > 40 {
            return Err(PreparationError::new(
                "target must contain between 1 and 40 hexadecimal digits",
            ));
        }
        let supplied_bits = (digits.len() * 4) as u32;
        if bits == 0 || bits > supplied_bits || bits > 160 {
            return Err(PreparationError::new(
                "target width must select between 1 and 160 supplied leading bits",
            ));
        }

        let mut aligned = [0_u8; 20];
        for (index, byte) in digits.bytes().enumerate() {
            let nibble = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => {
                    return Err(PreparationError::new(
                        "target contains a non-hexadecimal digit",
                    ))
                }
            };
            aligned[index / 2] |= if index % 2 == 0 { nibble << 4 } else { nibble };
        }
        let whole_bytes = (bits / 8) as usize;
        let remaining_bits = (bits % 8) as usize;
        if remaining_bits != 0 {
            aligned[whole_bytes] &= 0xff << (8 - remaining_bits);
            aligned[whole_bytes + 1..].fill(0);
        } else {
            aligned[whole_bytes..].fill(0);
        }

        let mut words = [0; 5];
        for (word, bytes) in words.iter_mut().zip(aligned.chunks_exact(4)) {
            *word = u32::from_be_bytes(bytes.try_into().unwrap());
        }
        Ok(Self { bits, words })
    }

    pub fn from_digest(mut digest: [u8; 20], bits: u32) -> Result<Self, PreparationError> {
        if !(1..=160).contains(&bits) {
            return Err(PreparationError::new(
                "target width must be between 1 and 160 bits",
            ));
        }
        let whole_bytes = (bits / 8) as usize;
        let remaining_bits = (bits % 8) as usize;
        if remaining_bits != 0 {
            digest[whole_bytes] &= 0xff << (8 - remaining_bits);
            digest[whole_bytes + 1..].fill(0);
        } else {
            digest[whole_bytes..].fill(0);
        }
        let mut words = [0; 5];
        for (word, bytes) in words.iter_mut().zip(digest.chunks_exact(4)) {
            *word = u32::from_be_bytes(bytes.try_into().unwrap());
        }
        Ok(Self { bits, words })
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn words(&self) -> [u32; 5] {
        self.words
    }

    pub fn matches(&self, digest: &[u8; 20]) -> bool {
        let full_bytes = (self.bits / 8) as usize;
        let partial_bits = self.bits % 8;
        let wanted = self.aligned_bytes();
        if digest[..full_bytes] != wanted[..full_bytes] {
            return false;
        }
        if partial_bits == 0 {
            return true;
        }
        let mask = 0xff << (8 - partial_bits);
        digest[full_bytes] & mask == wanted[full_bytes] & mask
    }

    fn aligned_bytes(&self) -> [u8; 20] {
        let mut result = [0; 20];
        for (bytes, word) in result.chunks_exact_mut(4).zip(self.words) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        result
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Carrier {
    Header,
    MessageTrailer,
}

#[derive(Clone, Debug)]
pub struct GitJob {
    carrier: Carrier,
    source_payload: Vec<u8>,
    payload_template: Vec<u8>,
    object_template: Vec<u8>,
    nonce_payload_offset: usize,
    nonce_object_offset: usize,
    mutable_block: usize,
    filler_bytes: usize,
    mutable_block_template: [u8; 64],
    suffix_words: Vec<[u32; 16]>,
    prestate: [u32; 5],
    target: TargetPrefix,
}

impl GitJob {
    pub fn header(payload: &[u8], target: TargetPrefix) -> Result<Self, PreparationError> {
        Self::header_named(payload, target, b"x")
    }

    pub fn header_named(
        payload: &[u8],
        target: TargetPrefix,
        header_name: &[u8],
    ) -> Result<Self, PreparationError> {
        Self::header_with_value_prefix(payload, target, header_name, b"")
    }

    pub fn header_epoch(
        payload: &[u8],
        target: TargetPrefix,
        epoch: u64,
    ) -> Result<Self, PreparationError> {
        let value_prefix = format!("{epoch:016x} ");
        Self::header_with_value_prefix(payload, target, b"x", value_prefix.as_bytes())
    }

    pub fn header_with_value_prefix(
        payload: &[u8],
        target: TargetPrefix,
        header_name: &[u8],
        value_prefix: &[u8],
    ) -> Result<Self, PreparationError> {
        if header_name.is_empty() || !header_name.iter().all(|byte| (0x21..=0x7e).contains(byte)) {
            return Err(PreparationError::new(
                "header name must be a non-empty printable ASCII token",
            ));
        }
        if value_prefix.contains(&0) || value_prefix.contains(&b'\n') {
            return Err(PreparationError::new(
                "header value prefix cannot contain NUL or newline",
            ));
        }
        let separator = payload
            .windows(2)
            .position(|bytes| bytes == b"\n\n")
            .ok_or_else(|| {
                PreparationError::new("commit payload has no blank line before its message")
            })?;
        let headers = &payload[..separator];
        let message = &payload[separator + 2..];
        let mut prefix =
            Vec::with_capacity(headers.len() + header_name.len() + value_prefix.len() + 2);
        prefix.extend_from_slice(headers);
        prefix.push(b'\n');
        prefix.extend_from_slice(header_name);
        prefix.push(b' ');
        prefix.extend_from_slice(value_prefix);

        for filler_bytes in 0..4096 {
            let nonce_payload_offset = prefix.len() + filler_bytes;
            let mut candidate_payload =
                Vec::with_capacity(prefix.len() + filler_bytes + NONCE_BYTES + 2 + message.len());
            candidate_payload.extend_from_slice(&prefix);
            candidate_payload.resize(candidate_payload.len() + filler_bytes, b' ');
            candidate_payload.extend_from_slice(b"PPPPP\n\n");
            candidate_payload.extend_from_slice(message);
            if let Some(job) = Self::finish(
                Carrier::Header,
                payload,
                candidate_payload,
                nonce_payload_offset,
                filler_bytes,
                target,
            ) {
                return Ok(job);
            }
        }
        Err(PreparationError::new(
            "could not align the custom-header nonce within 4096 filler bytes",
        ))
    }

    pub fn message_trailer(payload: &[u8], target: TargetPrefix) -> Result<Self, PreparationError> {
        Self::message_trailer_labeled(payload, target, b"X: ")
    }

    pub fn message_trailer_labeled(
        payload: &[u8],
        target: TargetPrefix,
        label: &[u8],
    ) -> Result<Self, PreparationError> {
        if label.is_empty() || label.contains(&0) || label.contains(&b'\n') {
            return Err(PreparationError::new(
                "trailer label must be non-empty and contain no NUL or newline",
            ));
        }
        let mut prefix = payload.to_vec();
        if !prefix.ends_with(b"\n") {
            prefix.push(b'\n');
        }
        prefix.extend_from_slice(label);
        for filler_bytes in 0..4096 {
            let nonce_payload_offset = prefix.len() + filler_bytes;
            let mut candidate_payload = Vec::with_capacity(prefix.len() + filler_bytes + 6);
            candidate_payload.extend_from_slice(&prefix);
            candidate_payload.resize(candidate_payload.len() + filler_bytes, b' ');
            candidate_payload.extend_from_slice(b"PPPPP\n");
            if let Some(job) = Self::finish(
                Carrier::MessageTrailer,
                payload,
                candidate_payload,
                nonce_payload_offset,
                filler_bytes,
                target,
            ) {
                if job.suffix_words.is_empty() {
                    return Ok(job);
                }
            }
        }
        Err(PreparationError::new(
            "could not align the message-trailer nonce within 4096 filler bytes",
        ))
    }

    fn finish(
        carrier: Carrier,
        source_payload: &[u8],
        payload_template: Vec<u8>,
        nonce_payload_offset: usize,
        filler_bytes: usize,
        target: TargetPrefix,
    ) -> Option<Self> {
        let object_template = serialize_commit(&payload_template);
        let object_header_bytes = object_template.len() - payload_template.len();
        let nonce_object_offset = object_header_bytes + nonce_payload_offset;
        if nonce_object_offset % 64 != NONCE_BLOCK_OFFSET {
            return None;
        }
        let padded = sha1::pad(&object_template);
        let mutable_block = nonce_object_offset / 64;
        let mutable_start = mutable_block * 64;
        let suffix_start = mutable_start + 64;
        if suffix_start > padded.len() {
            return None;
        }
        if carrier == Carrier::MessageTrailer && suffix_start != padded.len() {
            return None;
        }
        let mut mutable_block_template: [u8; 64] = padded[mutable_start..suffix_start]
            .try_into()
            .expect("one complete mutable block");
        mutable_block_template[NONCE_BLOCK_OFFSET..NONCE_BLOCK_OFFSET + NONCE_BYTES].fill(0);
        let suffix_words = padded[suffix_start..]
            .chunks_exact(64)
            .map(sha1::block_words)
            .collect();
        Some(Self {
            carrier,
            source_payload: source_payload.to_vec(),
            payload_template,
            object_template,
            nonce_payload_offset,
            nonce_object_offset,
            mutable_block,
            filler_bytes,
            mutable_block_template,
            suffix_words,
            prestate: sha1::prestate(&padded, mutable_block),
            target,
        })
    }

    pub fn carrier(&self) -> Carrier {
        self.carrier
    }

    pub fn source_payload(&self) -> &[u8] {
        &self.source_payload
    }

    pub fn payload_template(&self) -> &[u8] {
        &self.payload_template
    }

    pub fn object_template(&self) -> &[u8] {
        &self.object_template
    }

    pub fn nonce_payload_offset(&self) -> usize {
        self.nonce_payload_offset
    }

    pub fn nonce_object_offset(&self) -> usize {
        self.nonce_object_offset
    }

    pub fn mutable_block(&self) -> usize {
        self.mutable_block
    }

    pub fn filler_bytes(&self) -> usize {
        self.filler_bytes
    }

    pub fn suffix_blocks(&self) -> usize {
        self.suffix_words.len()
    }

    pub fn target(&self) -> TargetPrefix {
        self.target
    }

    pub fn low_level_job(&self) -> Result<Job, CudaError> {
        Job::new(
            &self.prestate,
            &sha1::block_words(&self.mutable_block_template),
            self.target.bits,
            &self.target.words,
        )
    }

    pub fn create_context(&self, device: i32) -> Result<Context, CudaError> {
        let policy = match self.carrier {
            Carrier::Header => NoncePolicy::HeaderSafe,
            Carrier::MessageTrailer => NoncePolicy::NoNul,
        };
        self.create_context_with_policy(device, policy)
    }

    pub fn create_context_with_policy(
        &self,
        device: i32,
        policy: NoncePolicy,
    ) -> Result<Context, CudaError> {
        let job = self.low_level_job()?;
        let mut context = Context::new(device, &job)?;
        self.configure_context_with_policy(&mut context, policy)?;
        Ok(context)
    }

    pub fn configure_context_with_policy(
        &self,
        context: &mut Context,
        policy: NoncePolicy,
    ) -> Result<(), CudaError> {
        let job = self.low_level_job()?;
        if self.carrier == Carrier::Header {
            context.set_header_job(&job, &self.suffix_words)?;
        } else {
            context.set_job(&job)?;
        }
        context.set_nonce_policy(policy)?;
        Ok(())
    }

    pub fn materialize_payload(&self, candidate: u64) -> Result<Vec<u8>, PreparationError> {
        let nonce = candidate_bytes(candidate)?;
        let mut payload = self.payload_template.clone();
        payload[self.nonce_payload_offset..self.nonce_payload_offset + NONCE_BYTES]
            .copy_from_slice(&nonce);
        Ok(payload)
    }

    pub fn materialize_object(&self, candidate: u64) -> Result<Vec<u8>, PreparationError> {
        Ok(serialize_commit(&self.materialize_payload(candidate)?))
    }

    pub fn digest(&self, candidate: u64) -> Result<[u8; 20], PreparationError> {
        Ok(sha1::digest(&self.materialize_object(candidate)?))
    }

    pub fn verify_candidate(&self, candidate: u64) -> Result<[u8; 20], PreparationError> {
        let policy = match self.carrier {
            Carrier::Header => NoncePolicy::HeaderSafe,
            Carrier::MessageTrailer => NoncePolicy::NoNul,
        };
        self.verify_candidate_with_policy(candidate, policy)
    }

    pub fn verify_candidate_with_policy(
        &self,
        candidate: u64,
        policy: NoncePolicy,
    ) -> Result<[u8; 20], PreparationError> {
        let nonce = candidate_bytes(candidate)?;
        let safe = match policy {
            NoncePolicy::NoNul => !nonce.contains(&0),
            NoncePolicy::HeaderSafe => !nonce.contains(&0) && !nonce.contains(&b'\n'),
            NoncePolicy::PrintableAscii => nonce.iter().all(|byte| (0x20..=0x7e).contains(byte)),
        };
        if !safe {
            return Err(PreparationError::new(match policy {
                NoncePolicy::NoNul => "candidate contains NUL",
                NoncePolicy::HeaderSafe => "candidate contains NUL or newline",
                NoncePolicy::PrintableAscii => "candidate contains a non-printable ASCII byte",
            }));
        }
        let digest = self.digest(candidate)?;
        if !self.target.matches(&digest) {
            return Err(PreparationError::new(
                "candidate digest does not match the requested target prefix",
            ));
        }
        Ok(digest)
    }
}

fn candidate_bytes(candidate: u64) -> Result<[u8; 5], PreparationError> {
    if candidate >= (1_u64 << NONCE_BITS) {
        return Err(PreparationError::new("candidate must fit in 40 bits"));
    }
    let bytes = candidate.to_be_bytes();
    Ok(bytes[3..].try_into().unwrap())
}

fn serialize_commit(payload: &[u8]) -> Vec<u8> {
    let header = format!("commit {}\0", payload.len());
    let mut object = Vec::with_capacity(header.len() + payload.len());
    object.extend_from_slice(header.as_bytes());
    object.extend_from_slice(payload);
    object
}

#[cfg(test)]
mod tests {
    use super::{Carrier, GitJob, TargetPrefix};
    use crate::device_count;

    const COMMIT: &[u8] = b"tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\
author Test <test@localhost> 0 +0000\n\
committer Test <test@localhost> 0 +0000\n\n\
visible subject\n";

    #[test]
    fn target_prefix_supports_partial_nibbles() {
        let target = TargetPrefix::from_hex_bits("a8", 5).unwrap();
        assert_eq!(target.bits(), 5);
        assert_eq!(target.words()[0], 0xa800_0000);
        let mut matching = [0; 20];
        matching[0] = 0xab;
        assert!(target.matches(&matching));
        matching[0] = 0xa0;
        assert!(!target.matches(&matching));
    }

    #[test]
    fn header_layout_matches_cross_language_fixture() {
        let target = TargetPrefix::from_hex("fb542602a40a02bb2fe6613a1ca13d7489d83246").unwrap();
        let job = GitJob::header(COMMIT, target).unwrap();
        assert_eq!(job.carrier(), Carrier::Header);
        assert_eq!(job.nonce_object_offset() % 64, 48);
        assert_eq!(job.suffix_blocks(), 1);
        assert_eq!(job.filler_bytes(), 40);
        assert_eq!(
            job.verify_candidate(0x6162_6364_65).unwrap(),
            target.aligned_bytes()
        );
        let payload = job.materialize_payload(0x6162_6364_65).unwrap();
        let headers = &payload[..payload.windows(2).position(|x| x == b"\n\n").unwrap()];
        assert!(headers.windows(3).any(|x| x == b"\nx "));
        assert!(headers.ends_with(b"abcde"));
        assert!(payload.ends_with(b"\n\nvisible subject\n"));
    }

    #[test]
    fn trailer_layout_has_no_suffix() {
        let target = TargetPrefix::from_hex("0").unwrap();
        let job = GitJob::message_trailer(COMMIT, target).unwrap();
        assert_eq!(job.carrier(), Carrier::MessageTrailer);
        assert_eq!(job.nonce_object_offset() % 64, 48);
        assert_eq!(job.suffix_blocks(), 0);
    }

    #[test]
    fn header_epochs_produce_distinct_aligned_domains() {
        let target = TargetPrefix::from_hex("0123456789a").unwrap();
        let first = GitJob::header_epoch(COMMIT, target, 0).unwrap();
        let second = GitJob::header_epoch(COMMIT, target, 1).unwrap();
        assert_eq!(first.nonce_object_offset() % 64, 48);
        assert_eq!(second.nonce_object_offset() % 64, 48);
        assert_ne!(first.object_template(), second.object_template());
        assert_ne!(
            first.digest(0x6162_6364_65).unwrap(),
            second.digest(0x6162_6364_65).unwrap()
        );
    }

    #[test]
    fn cuda_header_digest_matches_cpu() {
        if !matches!(device_count(), Ok(count) if count > 0) {
            return;
        }
        let target = TargetPrefix::from_hex("fb542602a40a02bb2fe6613a1ca13d7489d83246").unwrap();
        let job = GitJob::header(COMMIT, target).unwrap();
        let mut context = job.create_context(0).unwrap();
        let words = context.digest(0x6162_6364_65).unwrap();
        let mut digest = [0; 20];
        for (bytes, word) in digest.chunks_exact_mut(4).zip(words) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        assert_eq!(digest, job.digest(0x6162_6364_65).unwrap());
        let result = context.search(0x6162_6364, 1).unwrap();
        assert_eq!(result.candidate, Some(0x6162_6364_65));

        let mut long_commit = COMMIT.to_vec();
        long_commit.extend(std::iter::repeat_n(b'z', 512));
        let planning = GitJob::header(&long_commit, TargetPrefix::from_hex("0").unwrap()).unwrap();
        let expected = planning.digest(0x6162_6364_65).unwrap();
        let target = TargetPrefix::from_digest(expected, 160).unwrap();
        let long_job = GitJob::header(&long_commit, target).unwrap();
        assert!(long_job.suffix_blocks() > 1);
        let mut long_context = long_job.create_context(0).unwrap();
        let words = long_context.digest(0x6162_6364_65).unwrap();
        let mut actual = [0; 20];
        for (bytes, word) in actual.chunks_exact_mut(4).zip(words) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        assert_eq!(actual, expected);
    }
}
