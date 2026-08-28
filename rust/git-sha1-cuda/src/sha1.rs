pub(crate) const IV: [u32; 5] = [
    0x6745_2301,
    0xefcd_ab89,
    0x98ba_dcfe,
    0x1032_5476,
    0xc3d2_e1f0,
];

pub(crate) fn pad(message: &[u8]) -> Vec<u8> {
    let mut padded = Vec::with_capacity((message.len() + 72) & !63);
    padded.extend_from_slice(message);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&((message.len() as u64) * 8).to_be_bytes());
    padded
}

pub(crate) fn block_words(block: &[u8]) -> [u32; 16] {
    assert_eq!(block.len(), 64);
    let mut words = [0; 16];
    for (word, bytes) in words.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes(bytes.try_into().unwrap());
    }
    words
}

pub(crate) fn compress(state: [u32; 5], block: &[u8]) -> [u32; 5] {
    assert_eq!(block.len(), 64);
    let mut w = [0_u32; 80];
    w[..16].copy_from_slice(&block_words(block));
    for t in 16..80 {
        w[t] = (w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]).rotate_left(1);
    }

    let [mut a, mut b, mut c, mut d, mut e] = state;
    for (t, word) in w.into_iter().enumerate() {
        let (function, constant) = match t {
            0..=19 => (d ^ (b & (c ^ d)), 0x5a82_7999),
            20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
            40..=59 => ((b & c) | (d & (b | c)), 0x8f1b_bcdc),
            _ => (b ^ c ^ d, 0xca62_c1d6),
        };
        let next = a
            .rotate_left(5)
            .wrapping_add(function)
            .wrapping_add(e)
            .wrapping_add(constant)
            .wrapping_add(word);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = next;
    }
    [
        state[0].wrapping_add(a),
        state[1].wrapping_add(b),
        state[2].wrapping_add(c),
        state[3].wrapping_add(d),
        state[4].wrapping_add(e),
    ]
}

pub(crate) fn digest(message: &[u8]) -> [u8; 20] {
    let mut state = IV;
    let padded = pad(message);
    for block in padded.chunks_exact(64) {
        state = compress(state, block);
    }
    state_to_digest(state)
}

pub(crate) fn prestate(padded: &[u8], block_index: usize) -> [u32; 5] {
    assert_eq!(padded.len() % 64, 0);
    assert!(block_index <= padded.len() / 64);
    let mut state = IV;
    for block in padded[..block_index * 64].chunks_exact(64) {
        state = compress(state, block);
    }
    state
}

pub(crate) fn state_to_digest(state: [u32; 5]) -> [u8; 20] {
    let mut digest = [0; 20];
    for (bytes, word) in digest.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::digest;

    #[test]
    fn standard_vectors() {
        assert_eq!(digest(b""), hex("da39a3ee5e6b4b0d3255bfef95601890afd80709"));
        assert_eq!(
            digest(b"abc"),
            hex("a9993e364706816aba3e25717850c26c9cd0d89d")
        );
    }

    fn hex(text: &str) -> [u8; 20] {
        let mut result = [0; 20];
        for (index, byte) in result.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).unwrap();
        }
        result
    }
}
