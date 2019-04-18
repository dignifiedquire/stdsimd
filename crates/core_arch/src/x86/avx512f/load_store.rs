//! Intrinsics for Load and Store Operations

use crate::{
    core_arch::{simd::*, x86::*},
    mem::{self, transmute},
    ptr,
};

/// Load 512-bits of integer data from memory into `dst.mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_loadu_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovdqu32))]
pub unsafe fn _mm512_loadu_si512(mem_addr: *const __m512i) -> __m512i {
    let mut dst = _mm512_undefined();
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m512i as *mut u8,
        mem::size_of::<__m512i>(),
    );
    dst
}

/// Returns vector of type __m512i with undefined elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_undefined)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_undefined() -> __m512i {
    // FIXME: this function should return MaybeUninit<__m512i>
    mem::MaybeUninit::<__m512i>::uninit().assume_init()
}
