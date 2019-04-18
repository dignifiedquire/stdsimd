//! Intrinsics for Bit Manipulation Operations

use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
};

/// Computes the bitwise XOR of 512 bits (representing integer data) in `a` and `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_xor_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpxord))]
pub unsafe fn _mm512_xor_si512(a: __m512i, b: __m512i) -> __m512i {
    simd_xor(a, b)
}

/// Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in `imm8`, and store the results in `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_ror_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprord))]
pub unsafe fn _mm512_ror_epi32(a: __m512i, imm8: i32) -> __m512i {
    prord128(a, imm8)
}

/// LLVM intrinsics used in the above functions
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.pror.d.128"]
    fn prord128(a: __m512i, imm8: i32) -> __m512i;
}
