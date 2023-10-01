/* automatically generated by rust-bindgen 0.68.1 */

pub const MKL_DOMAIN_ALL: u32 = 0;
pub const MKL_DOMAIN_BLAS: u32 = 1;
pub const MKL_DOMAIN_FFT: u32 = 2;
pub const MKL_DOMAIN_VML: u32 = 3;
pub const MKL_DOMAIN_PARDISO: u32 = 4;
pub const MKL_DOMAIN_LAPACK: u32 = 5;
pub const MKL_CBWR_BRANCH: u32 = 1;
pub const MKL_CBWR_ALL: i32 = -1;
pub const MKL_CBWR_STRICT: u32 = 65536;
pub const MKL_CBWR_OFF: u32 = 0;
pub const MKL_CBWR_BRANCH_OFF: u32 = 1;
pub const MKL_CBWR_AUTO: u32 = 2;
pub const MKL_CBWR_COMPATIBLE: u32 = 3;
pub const MKL_CBWR_SSE2: u32 = 4;
pub const MKL_CBWR_SSSE3: u32 = 6;
pub const MKL_CBWR_SSE4_1: u32 = 7;
pub const MKL_CBWR_SSE4_2: u32 = 8;
pub const MKL_CBWR_AVX: u32 = 9;
pub const MKL_CBWR_AVX2: u32 = 10;
pub const MKL_CBWR_AVX512_MIC: u32 = 11;
pub const MKL_CBWR_AVX512: u32 = 12;
pub const MKL_CBWR_AVX512_MIC_E1: u32 = 13;
pub const MKL_CBWR_AVX512_E1: u32 = 14;
pub const MKL_CBWR_SUCCESS: u32 = 0;
pub const MKL_CBWR_ERR_INVALID_SETTINGS: i32 = -1;
pub const MKL_CBWR_ERR_INVALID_INPUT: i32 = -2;
pub const MKL_CBWR_ERR_UNSUPPORTED_BRANCH: i32 = -3;
pub const MKL_CBWR_ERR_UNKNOWN_BRANCH: i32 = -4;
pub const MKL_CBWR_ERR_MODE_CHANGE_FAILURE: i32 = -8;
pub const MKL_CBWR_SSE3: u32 = 5;
pub const MKL_DSS_DEFAULTS: u32 = 0;
pub const MKL_DSS_OOC_VARIABLE: u32 = 1024;
pub const MKL_DSS_OOC_STRONG: u32 = 2048;
pub const MKL_DSS_REFINEMENT_OFF: u32 = 4096;
pub const MKL_DSS_REFINEMENT_ON: u32 = 8192;
pub const MKL_DSS_FORWARD_SOLVE: u32 = 16384;
pub const MKL_DSS_DIAGONAL_SOLVE: u32 = 32768;
pub const MKL_DSS_BACKWARD_SOLVE: u32 = 49152;
pub const MKL_DSS_TRANSPOSE_SOLVE: u32 = 262144;
pub const MKL_DSS_CONJUGATE_SOLVE: u32 = 524288;
pub const MKL_DSS_SINGLE_PRECISION: u32 = 65536;
pub const MKL_DSS_ZERO_BASED_INDEXING: u32 = 131072;
pub const MKL_DSS_MSG_LVL_SUCCESS: i32 = -2147483647;
pub const MKL_DSS_MSG_LVL_DEBUG: i32 = -2147483646;
pub const MKL_DSS_MSG_LVL_INFO: i32 = -2147483645;
pub const MKL_DSS_MSG_LVL_WARNING: i32 = -2147483644;
pub const MKL_DSS_MSG_LVL_ERROR: i32 = -2147483643;
pub const MKL_DSS_MSG_LVL_FATAL: i32 = -2147483642;
pub const MKL_DSS_TERM_LVL_SUCCESS: u32 = 1073741832;
pub const MKL_DSS_TERM_LVL_DEBUG: u32 = 1073741840;
pub const MKL_DSS_TERM_LVL_INFO: u32 = 1073741848;
pub const MKL_DSS_TERM_LVL_WARNING: u32 = 1073741856;
pub const MKL_DSS_TERM_LVL_ERROR: u32 = 1073741864;
pub const MKL_DSS_TERM_LVL_FATAL: u32 = 1073741872;
pub const MKL_DSS_SYMMETRIC: u32 = 536870976;
pub const MKL_DSS_SYMMETRIC_STRUCTURE: u32 = 536871040;
pub const MKL_DSS_NON_SYMMETRIC: u32 = 536871104;
pub const MKL_DSS_SYMMETRIC_COMPLEX: u32 = 536871168;
pub const MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX: u32 = 536871232;
pub const MKL_DSS_NON_SYMMETRIC_COMPLEX: u32 = 536871296;
pub const MKL_DSS_AUTO_ORDER: u32 = 268435520;
pub const MKL_DSS_MY_ORDER: u32 = 268435584;
pub const MKL_DSS_OPTION1_ORDER: u32 = 268435648;
pub const MKL_DSS_GET_ORDER: u32 = 268435712;
pub const MKL_DSS_METIS_ORDER: u32 = 268435776;
pub const MKL_DSS_METIS_OPENMP_ORDER: u32 = 268435840;
pub const MKL_DSS_POSITIVE_DEFINITE: u32 = 134217792;
pub const MKL_DSS_INDEFINITE: u32 = 134217856;
pub const MKL_DSS_HERMITIAN_POSITIVE_DEFINITE: u32 = 134217920;
pub const MKL_DSS_HERMITIAN_INDEFINITE: u32 = 134217984;
pub const MKL_DSS_SUCCESS: u32 = 0;
pub const MKL_DSS_ZERO_PIVOT: i32 = -1;
pub const MKL_DSS_OUT_OF_MEMORY: i32 = -2;
pub const MKL_DSS_FAILURE: i32 = -3;
pub const MKL_DSS_ROW_ERR: i32 = -4;
pub const MKL_DSS_COL_ERR: i32 = -5;
pub const MKL_DSS_TOO_FEW_VALUES: i32 = -6;
pub const MKL_DSS_TOO_MANY_VALUES: i32 = -7;
pub const MKL_DSS_NOT_SQUARE: i32 = -8;
pub const MKL_DSS_STATE_ERR: i32 = -9;
pub const MKL_DSS_INVALID_OPTION: i32 = -10;
pub const MKL_DSS_OPTION_CONFLICT: i32 = -11;
pub const MKL_DSS_MSG_LVL_ERR: i32 = -12;
pub const MKL_DSS_TERM_LVL_ERR: i32 = -13;
pub const MKL_DSS_STRUCTURE_ERR: i32 = -14;
pub const MKL_DSS_REORDER_ERR: i32 = -15;
pub const MKL_DSS_VALUES_ERR: i32 = -16;
pub const MKL_DSS_STATISTICS_INVALID_MATRIX: i32 = -17;
pub const MKL_DSS_STATISTICS_INVALID_STATE: i32 = -18;
pub const MKL_DSS_STATISTICS_INVALID_STRING: i32 = -19;
pub const MKL_DSS_REORDER1_ERR: i32 = -20;
pub const MKL_DSS_PREORDER_ERR: i32 = -21;
pub const MKL_DSS_DIAG_ERR: i32 = -22;
pub const MKL_DSS_I32BIT_ERR: i32 = -23;
pub const MKL_DSS_OOC_MEM_ERR: i32 = -24;
pub const MKL_DSS_OOC_OC_ERR: i32 = -25;
pub const MKL_DSS_OOC_RW_ERR: i32 = -26;
pub const PARDISO_NO_ERROR: u32 = 0;
pub const PARDISO_UNIMPLEMENTED: i32 = -101;
pub const PARDISO_NULL_HANDLE: i32 = -102;
pub const PARDISO_MEMORY_ERROR: i32 = -103;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _MKL_Complex8 {
    pub real: f32,
    pub imag: f32,
}
#[test]
fn bindgen_test_layout__MKL_Complex8() {
    const UNINIT: ::std::mem::MaybeUninit<_MKL_Complex8> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<_MKL_Complex8>(),
        8usize,
        concat!("Size of: ", stringify!(_MKL_Complex8))
    );
    assert_eq!(
        ::std::mem::align_of::<_MKL_Complex8>(),
        4usize,
        concat!("Alignment of ", stringify!(_MKL_Complex8))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).real) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(_MKL_Complex8),
            "::",
            stringify!(real)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).imag) as usize - ptr as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(_MKL_Complex8),
            "::",
            stringify!(imag)
        )
    );
}
pub type MKL_Complex8 = _MKL_Complex8;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _MKL_Complex16 {
    pub real: f64,
    pub imag: f64,
}
#[test]
fn bindgen_test_layout__MKL_Complex16() {
    const UNINIT: ::std::mem::MaybeUninit<_MKL_Complex16> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<_MKL_Complex16>(),
        16usize,
        concat!("Size of: ", stringify!(_MKL_Complex16))
    );
    assert_eq!(
        ::std::mem::align_of::<_MKL_Complex16>(),
        8usize,
        concat!("Alignment of ", stringify!(_MKL_Complex16))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).real) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(_MKL_Complex16),
            "::",
            stringify!(real)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).imag) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(_MKL_Complex16),
            "::",
            stringify!(imag)
        )
    );
}
pub type MKL_Complex16 = _MKL_Complex16;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MKLVersion {
    pub MajorVersion: ::std::os::raw::c_int,
    pub MinorVersion: ::std::os::raw::c_int,
    pub UpdateVersion: ::std::os::raw::c_int,
    pub ProductStatus: *mut ::std::os::raw::c_char,
    pub Build: *mut ::std::os::raw::c_char,
    pub Processor: *mut ::std::os::raw::c_char,
    pub Platform: *mut ::std::os::raw::c_char,
}
#[test]
fn bindgen_test_layout_MKLVersion() {
    const UNINIT: ::std::mem::MaybeUninit<MKLVersion> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<MKLVersion>(),
        48usize,
        concat!("Size of: ", stringify!(MKLVersion))
    );
    assert_eq!(
        ::std::mem::align_of::<MKLVersion>(),
        8usize,
        concat!("Alignment of ", stringify!(MKLVersion))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).MajorVersion) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(MKLVersion),
            "::",
            stringify!(MajorVersion)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).MinorVersion) as usize - ptr as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(MKLVersion),
            "::",
            stringify!(MinorVersion)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).UpdateVersion) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(MKLVersion),
            "::",
            stringify!(UpdateVersion)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).ProductStatus) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(MKLVersion),
            "::",
            stringify!(ProductStatus)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).Build) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(MKLVersion),
            "::",
            stringify!(Build)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).Processor) as usize - ptr as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(MKLVersion),
            "::",
            stringify!(Processor)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).Platform) as usize - ptr as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(MKLVersion),
            "::",
            stringify!(Platform)
        )
    );
}
pub const MKL_LAYOUT_MKL_ROW_MAJOR: MKL_LAYOUT = 101;
pub const MKL_LAYOUT_MKL_COL_MAJOR: MKL_LAYOUT = 102;
pub type MKL_LAYOUT = ::std::os::raw::c_uint;
pub const MKL_TRANSPOSE_MKL_NOTRANS: MKL_TRANSPOSE = 111;
pub const MKL_TRANSPOSE_MKL_TRANS: MKL_TRANSPOSE = 112;
pub const MKL_TRANSPOSE_MKL_CONJTRANS: MKL_TRANSPOSE = 113;
pub const MKL_TRANSPOSE_MKL_CONJ: MKL_TRANSPOSE = 114;
pub type MKL_TRANSPOSE = ::std::os::raw::c_uint;
pub const MKL_UPLO_MKL_UPPER: MKL_UPLO = 121;
pub const MKL_UPLO_MKL_LOWER: MKL_UPLO = 122;
pub type MKL_UPLO = ::std::os::raw::c_uint;
pub const MKL_DIAG_MKL_NONUNIT: MKL_DIAG = 131;
pub const MKL_DIAG_MKL_UNIT: MKL_DIAG = 132;
pub type MKL_DIAG = ::std::os::raw::c_uint;
pub const MKL_SIDE_MKL_LEFT: MKL_SIDE = 141;
pub const MKL_SIDE_MKL_RIGHT: MKL_SIDE = 142;
pub type MKL_SIDE = ::std::os::raw::c_uint;
pub const MKL_COMPACT_PACK_MKL_COMPACT_SSE: MKL_COMPACT_PACK = 181;
pub const MKL_COMPACT_PACK_MKL_COMPACT_AVX: MKL_COMPACT_PACK = 182;
pub const MKL_COMPACT_PACK_MKL_COMPACT_AVX512: MKL_COMPACT_PACK = 183;
pub type MKL_COMPACT_PACK = ::std::os::raw::c_uint;
pub type sgemm_jit_kernel_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ::std::os::raw::c_void,
        arg2: *mut f32,
        arg3: *mut f32,
        arg4: *mut f32,
    ),
>;
pub type dgemm_jit_kernel_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ::std::os::raw::c_void,
        arg2: *mut f64,
        arg3: *mut f64,
        arg4: *mut f64,
    ),
>;
pub type cgemm_jit_kernel_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ::std::os::raw::c_void,
        arg2: *mut MKL_Complex8,
        arg3: *mut MKL_Complex8,
        arg4: *mut MKL_Complex8,
    ),
>;
pub type zgemm_jit_kernel_t = ::std::option::Option<
    unsafe extern "C" fn(
        arg1: *mut ::std::os::raw::c_void,
        arg2: *mut MKL_Complex16,
        arg3: *mut MKL_Complex16,
        arg4: *mut MKL_Complex16,
    ),
>;
pub const mkl_jit_status_t_MKL_JIT_SUCCESS: mkl_jit_status_t = 0;
pub const mkl_jit_status_t_MKL_NO_JIT: mkl_jit_status_t = 1;
pub const mkl_jit_status_t_MKL_JIT_ERROR: mkl_jit_status_t = 2;
pub type mkl_jit_status_t = ::std::os::raw::c_uint;
pub type _MKL_DSS_HANDLE_t = *mut ::std::os::raw::c_void;
pub type _CHARACTER_t = ::std::os::raw::c_char;
pub type _CHARACTER_STR_t = ::std::os::raw::c_char;
pub type _LONG_t = ::std::os::raw::c_long;
pub type _REAL_t = f32;
pub type _DOUBLE_PRECISION_t = f64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _DOUBLE_COMPLEX_t {
    pub r: f64,
    pub i: f64,
}
#[test]
fn bindgen_test_layout__DOUBLE_COMPLEX_t() {
    const UNINIT: ::std::mem::MaybeUninit<_DOUBLE_COMPLEX_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<_DOUBLE_COMPLEX_t>(),
        16usize,
        concat!("Size of: ", stringify!(_DOUBLE_COMPLEX_t))
    );
    assert_eq!(
        ::std::mem::align_of::<_DOUBLE_COMPLEX_t>(),
        8usize,
        concat!("Alignment of ", stringify!(_DOUBLE_COMPLEX_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).r) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(_DOUBLE_COMPLEX_t),
            "::",
            stringify!(r)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).i) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(_DOUBLE_COMPLEX_t),
            "::",
            stringify!(i)
        )
    );
}
extern "C" {
    pub fn dss_create_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_define_structure_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_int,
        arg4: *const ::std::os::raw::c_int,
        arg5: *const ::std::os::raw::c_int,
        arg6: *const ::std::os::raw::c_int,
        arg7: *const ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_reorder_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_factor_real_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_void,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_factor_complex_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_void,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_solve_real_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_void,
        arg4: *const ::std::os::raw::c_int,
        arg5: *mut ::std::os::raw::c_void,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_solve_complex_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_void,
        arg4: *const ::std::os::raw::c_int,
        arg5: *mut ::std::os::raw::c_void,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_statistics_(
        arg1: *mut _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const _CHARACTER_STR_t,
        arg4: *mut _DOUBLE_PRECISION_t,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn dss_delete_(
        arg1: *const _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn pardiso(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_int,
        arg4: *const ::std::os::raw::c_int,
        arg5: *const ::std::os::raw::c_int,
        arg6: *const ::std::os::raw::c_int,
        arg7: *const ::std::os::raw::c_void,
        arg8: *const ::std::os::raw::c_int,
        arg9: *const ::std::os::raw::c_int,
        arg10: *mut ::std::os::raw::c_int,
        arg11: *const ::std::os::raw::c_int,
        arg12: *mut ::std::os::raw::c_int,
        arg13: *const ::std::os::raw::c_int,
        arg14: *mut ::std::os::raw::c_void,
        arg15: *mut ::std::os::raw::c_void,
        arg16: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *const ::std::os::raw::c_int,
        arg4: *const ::std::os::raw::c_int,
        arg5: *const ::std::os::raw::c_int,
        arg6: *const ::std::os::raw::c_int,
        arg7: *const ::std::os::raw::c_void,
        arg8: *const ::std::os::raw::c_int,
        arg9: *const ::std::os::raw::c_int,
        arg10: *mut ::std::os::raw::c_int,
        arg11: *const ::std::os::raw::c_int,
        arg12: *mut ::std::os::raw::c_int,
        arg13: *const ::std::os::raw::c_int,
        arg14: *mut ::std::os::raw::c_void,
        arg15: *mut ::std::os::raw::c_void,
        arg16: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardisoinit(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISOINIT(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_int,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardiso_handle_store(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_HANDLE_STORE(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardiso_handle_restore(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_HANDLE_RESTORE(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardiso_handle_delete(
        arg1: *const ::std::os::raw::c_char,
        arg2: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_HANDLE_DELETE(
        arg1: *const ::std::os::raw::c_char,
        arg2: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardiso_64(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_longlong,
        arg3: *const ::std::os::raw::c_longlong,
        arg4: *const ::std::os::raw::c_longlong,
        arg5: *const ::std::os::raw::c_longlong,
        arg6: *const ::std::os::raw::c_longlong,
        arg7: *const ::std::os::raw::c_void,
        arg8: *const ::std::os::raw::c_longlong,
        arg9: *const ::std::os::raw::c_longlong,
        arg10: *mut ::std::os::raw::c_longlong,
        arg11: *const ::std::os::raw::c_longlong,
        arg12: *mut ::std::os::raw::c_longlong,
        arg13: *const ::std::os::raw::c_longlong,
        arg14: *mut ::std::os::raw::c_void,
        arg15: *mut ::std::os::raw::c_void,
        arg16: *mut ::std::os::raw::c_longlong,
    );
}
extern "C" {
    pub fn PARDISO_64(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_longlong,
        arg3: *const ::std::os::raw::c_longlong,
        arg4: *const ::std::os::raw::c_longlong,
        arg5: *const ::std::os::raw::c_longlong,
        arg6: *const ::std::os::raw::c_longlong,
        arg7: *const ::std::os::raw::c_void,
        arg8: *const ::std::os::raw::c_longlong,
        arg9: *const ::std::os::raw::c_longlong,
        arg10: *mut ::std::os::raw::c_longlong,
        arg11: *const ::std::os::raw::c_longlong,
        arg12: *mut ::std::os::raw::c_longlong,
        arg13: *const ::std::os::raw::c_longlong,
        arg14: *mut ::std::os::raw::c_void,
        arg15: *mut ::std::os::raw::c_void,
        arg16: *mut ::std::os::raw::c_longlong,
    );
}
extern "C" {
    pub fn pardiso_handle_store_64(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_HANDLE_STORE_64(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardiso_handle_restore_64(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_HANDLE_RESTORE_64(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *const ::std::os::raw::c_char,
        arg3: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardiso_handle_delete_64(
        arg1: *const ::std::os::raw::c_char,
        arg2: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_HANDLE_DELETE_64(
        arg1: *const ::std::os::raw::c_char,
        arg2: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn MKL_PARDISO_PIVOT(
        aii: *const f64,
        bii: *mut f64,
        eps: *const f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn MKL_PARDISO_PIVOT_(
        aii: *const f64,
        bii: *mut f64,
        eps: *const f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn mkl_pardiso_pivot(
        aii: *const f64,
        bii: *mut f64,
        eps: *const f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn mkl_pardiso_pivot_(
        aii: *const f64,
        bii: *mut f64,
        eps: *const f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn pardiso_getdiag(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *mut ::std::os::raw::c_void,
        arg3: *mut ::std::os::raw::c_void,
        arg4: *const ::std::os::raw::c_int,
        arg5: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_GETDIAG(
        arg1: _MKL_DSS_HANDLE_t,
        arg2: *mut ::std::os::raw::c_void,
        arg3: *mut ::std::os::raw::c_void,
        arg4: *const ::std::os::raw::c_int,
        arg5: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn pardiso_export(
        pt: *mut ::std::os::raw::c_void,
        values: *mut ::std::os::raw::c_void,
        ia: *mut ::std::os::raw::c_int,
        ja: *mut ::std::os::raw::c_int,
        step: *const ::std::os::raw::c_int,
        iparm: *const ::std::os::raw::c_int,
        error: *mut ::std::os::raw::c_int,
    );
}
extern "C" {
    pub fn PARDISO_EXPORT(
        pt: *mut ::std::os::raw::c_void,
        values: *mut ::std::os::raw::c_void,
        ia: *mut ::std::os::raw::c_int,
        ja: *mut ::std::os::raw::c_int,
        step: *const ::std::os::raw::c_int,
        iparm: *const ::std::os::raw::c_int,
        error: *mut ::std::os::raw::c_int,
    );
}
