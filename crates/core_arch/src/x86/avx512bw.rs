use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Shuffles bytes from `a` according to the content of `b`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_shuffle_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpshufb))]
pub unsafe fn _mm512_shuffle_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(pshufb(a.as_i8x64(), b.as_i8x64()))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.pshuf.b.512"]
    fn pshufb(a: i8x64, b: i8x64) -> i8x64;
}

#[cfg(test)]
mod tests {
    use std;

    use stdsimd_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_shuffle_epi8() {
        let a = _mm512_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let b = _mm512_setzero_si512();
        let expected = _mm512_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r: __m512i = _mm512_shuffle_epi8(a, b);
        assert_eq_m512i(r, expected);
    }
}
