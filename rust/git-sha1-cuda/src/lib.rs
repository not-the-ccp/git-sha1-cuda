//! Safe bindings to the optimized five-byte Git SHA-1 CUDA search backend.

use std::{error::Error, ffi::CStr, fmt, os::raw::c_char, ptr};

const OK: i32 = 0;
const FOUND: i32 = 1;
const NOT_FOUND: i32 = 2;
pub const NO_WINNER: u64 = u64::MAX;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RawJob {
    pub abi_version: u32,
    pub prestate: [u32; 5],
    pub pre12: [u32; 5],
    pub base_words: [u32; 16],
    pub target_words: [u32; 5],
    pub target_masks: [u32; 5],
    pub target_bits: u32,
    pub h0_gate_base: u32,
    pub h0_gate_span: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawSearchResult {
    found: u32,
    reserved: u32,
    candidate: u64,
    candidates_hashed: u64,
    milliseconds: f32,
    billions_per_second: f32,
}

const _: [(); 160] = [(); std::mem::size_of::<RawJob>()];
const _: [(); 32] = [(); std::mem::size_of::<RawSearchResult>()];

#[repr(C)]
struct RawContext {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn gsv_device_count() -> i32;
    fn gsv_job_init(
        job: *mut RawJob,
        prestate: *const u32,
        base_words: *const u32,
        target_bits: u32,
        target_words: *const u32,
    ) -> i32;
    fn gsv_context_create(device: i32, job: *const RawJob, out: *mut *mut RawContext) -> i32;
    fn gsv_context_destroy(context: *mut RawContext);
    fn gsv_context_set_job(context: *mut RawContext, job: *const RawJob) -> i32;
    fn gsv_search(
        context: *mut RawContext,
        outer_base: u64,
        outer_count: u64,
        result: *mut RawSearchResult,
    ) -> i32;
    fn gsv_digest(context: *mut RawContext, candidate: u64, digest: *mut u32) -> i32;
    fn gsv_last_error(context: *const RawContext) -> *const c_char;
    fn gsv_status_string(status: i32) -> *const c_char;
}

#[derive(Debug, Clone)]
pub struct CudaError {
    pub status: i32,
    pub message: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (status {})", self.message, self.status)
    }
}

impl Error for CudaError {}

fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn error(status: i32, context: *const RawContext) -> CudaError {
    let detail = unsafe { c_string(gsv_last_error(context)) };
    let name = unsafe { c_string(gsv_status_string(status)) };
    CudaError {
        status,
        message: if detail.is_empty() {
            name
        } else {
            format!("{name}: {detail}")
        },
    }
}

pub fn device_count() -> Result<u32, CudaError> {
    let count = unsafe { gsv_device_count() };
    if count < 0 {
        Err(error(-2, ptr::null()))
    } else {
        Ok(count as u32)
    }
}

#[derive(Clone, Copy)]
pub struct Job(RawJob);

impl Job {
    /// `base_words` is the final padded SHA-1 block in big-endian word order.
    /// W12 and W13's high byte must be zero for the five-byte candidate.
    pub fn new(
        prestate: &[u32; 5],
        base_words: &[u32; 16],
        target_bits: u32,
        target_words: &[u32; 5],
    ) -> Result<Self, CudaError> {
        let mut raw = RawJob {
            abi_version: 0,
            prestate: [0; 5],
            pre12: [0; 5],
            base_words: [0; 16],
            target_words: [0; 5],
            target_masks: [0; 5],
            target_bits: 0,
            h0_gate_base: 0,
            h0_gate_span: 0,
        };
        let status = unsafe {
            gsv_job_init(
                &mut raw,
                prestate.as_ptr(),
                base_words.as_ptr(),
                target_bits,
                target_words.as_ptr(),
            )
        };
        if status == OK {
            Ok(Self(raw))
        } else {
            Err(error(status, ptr::null()))
        }
    }

    pub fn as_raw(&self) -> &RawJob {
        &self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub candidate: Option<u64>,
    pub candidates_hashed: u64,
    pub milliseconds: f32,
    pub billions_per_second: f32,
}

pub struct Context {
    raw: *mut RawContext,
}

impl Context {
    pub fn new(device: i32, job: &Job) -> Result<Self, CudaError> {
        let mut raw = ptr::null_mut();
        let status = unsafe { gsv_context_create(device, &job.0, &mut raw) };
        if status == OK {
            Ok(Self { raw })
        } else {
            Err(error(status, ptr::null()))
        }
    }

    pub fn set_job(&mut self, job: &Job) -> Result<(), CudaError> {
        let status = unsafe { gsv_context_set_job(self.raw, &job.0) };
        if status == OK {
            Ok(())
        } else {
            Err(error(status, self.raw))
        }
    }

    /// Searches `outer_count * 256` candidates, starting at `outer_base << 8`.
    pub fn search(&mut self, outer_base: u64, outer_count: u64) -> Result<SearchResult, CudaError> {
        let mut raw = RawSearchResult::default();
        let status = unsafe { gsv_search(self.raw, outer_base, outer_count, &mut raw) };
        if status != FOUND && status != NOT_FOUND {
            return Err(error(status, self.raw));
        }
        Ok(SearchResult {
            candidate: (raw.found != 0 && raw.candidate != NO_WINNER).then_some(raw.candidate),
            candidates_hashed: raw.candidates_hashed,
            milliseconds: raw.milliseconds,
            billions_per_second: raw.billions_per_second,
        })
    }

    pub fn digest(&mut self, candidate: u64) -> Result<[u32; 5], CudaError> {
        let mut digest = [0; 5];
        let status = unsafe { gsv_digest(self.raw, candidate, digest.as_mut_ptr()) };
        if status == OK {
            Ok(digest)
        } else {
            Err(error(status, self.raw))
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { gsv_context_destroy(self.raw) }
    }
}
