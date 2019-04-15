var N=null,E="",T="t",U="u",searchIndex={};
var R=["core_arch","vector_signed_short","vector_float","vector_unsigned_char","vector_signed_int","Vector permute.","result","try_from","try_into","borrow","borrow_mut","type_id","typeid","formatter","vector_signed_char","vector_bool_char","vector_unsigned_short","vector_bool_short","vector_unsigned_int","vector_bool_int","vector_signed_long","vector_unsigned_long","vector_bool_long","vector_double","std_detect"];
searchIndex[R[24]]={"doc":"Run-time feature detection for the Rust standard library.","i":[[14,"is_x86_feature_detected",R[24],"Prevents compilation if `is_x86_feature_detected` is used…",N,N],[14,"is_arm_feature_detected",E,"Prevents compilation if `is_arm_feature_detected` is used…",N,N],[14,"is_aarch64_feature_detected",E,"Prevents compilation if `is_aarch64_feature_detected` is…",N,N],[14,"is_powerpc_feature_detected",E,"Prevents compilation if `is_powerpc_feature_detected` is…",N,N],[14,"is_mips_feature_detected",E,"Prevents compilation if `is_mips_feature_detected` is used…",N,N],[14,"is_mips64_feature_detected",E,"Prevents compilation if `is_mips64_feature_detected` is…",N,N],[14,"is_powerpc64_feature_detected",E,"Checks if `powerpc64` feature is enabled.",N,N]],"p":[]};
searchIndex[R[0]]={"doc":"SIMD and vendor intrinsics module.","i":[[0,"powerpc64",R[0],"Platform-specific intrinsics for the `PowerPC64` platform.",N,N],[3,R[14],"core_arch::powerpc64","PowerPC-specific 128-bit wide vector of sixteen packed `i8`",N,N],[3,R[3],E,"PowerPC-specific 128-bit wide vector of sixteen packed `u8`",N,N],[3,R[15],E,"PowerPC-specific 128-bit wide vector mask of sixteen…",N,N],[3,R[1],E,"PowerPC-specific 128-bit wide vector of eight packed `i16`",N,N],[3,R[16],E,"PowerPC-specific 128-bit wide vector of eight packed `u16`",N,N],[3,R[17],E,"PowerPC-specific 128-bit wide vector mask of eight packed…",N,N],[3,R[4],E,"PowerPC-specific 128-bit wide vector of four packed `i32`",N,N],[3,R[18],E,"PowerPC-specific 128-bit wide vector of four packed `u32`",N,N],[3,R[19],E,"PowerPC-specific 128-bit wide vector mask of four packed…",N,N],[3,R[2],E,"PowerPC-specific 128-bit wide vector of four packed `f32`",N,N],[3,R[20],E,"PowerPC-specific 128-bit wide vector of two packed `i64`",N,N],[3,R[21],E,"PowerPC-specific 128-bit wide vector of two packed `u64`",N,N],[3,R[22],E,"PowerPC-specific 128-bit wide vector mask of two elements",N,N],[3,R[23],E,"PowerPC-specific 128-bit wide vector of two packed `f64`",N,N],[5,"vec_add",E,"Vector add.",N,[[[U],[T]]]],[5,"vec_madds",E,"Vector Multiply Add Saturated",N,[[[R[1]]],[R[1]]]],[5,"vec_mladd",E,"Vector Multiply Low and Add Unsigned Half Word",N,[[[U],[T]]]],[5,"vec_mradds",E,"Vector Multiply Round and Add Saturated",N,[[[R[1]]],[R[1]]]],[5,"vec_msum",E,"Vector Multiply Sum",N,[[[U],[T],["b"]],[U]]],[5,"vec_msums",E,"Vector Multiply Sum Saturated",N,[[[U],[T]],[U]]],[5,"vec_madd",E,"Vector Multiply Add",N,[[[R[2]]],[R[2]]]],[5,"vec_nmsub",E,"Vector Negative Multiply Subtract",N,[[[R[2]]],[R[2]]]],[5,"vec_sum4s",E,"Vector Sum Across Partial (1/4) Saturated",N,[[[U],[T]],[U]]],[5,"vec_perm",E,R[5],N,[[[T],[R[3]]],[T]]],[5,"vec_sum2s",E,"Vector Sum Across Partial (1/2) Saturated",N,[[[R[4]]],[R[4]]]],[5,"vec_mule",E,"Vector Multiply Even",N,[[[T]],[U]]],[5,"vec_mulo",E,"Vector Multiply Odd",N,[[[T]],[U]]],[5,"vec_xxpermdi",E,R[5],N,[[["u8"],[T]],[T]]],[5,"trap",E,"Generates the trap instruction `TRAP`",N,[[]]],[11,"from",E,E,0,[[[T]],[T]]],[11,R[7],E,E,0,[[[U]],[R[6]]]],[11,R[8],E,E,0,[[],[R[6]]]],[11,"into",E,E,0,[[],[U]]],[11,R[9],E,E,0,[[["self"]],[T]]],[11,R[10],E,E,0,[[["self"]],[T]]],[11,R[11],E,E,0,[[["self"]],[R[12]]]],[11,"from",E,E,1,[[[T]],[T]]],[11,R[7],E,E,1,[[[U]],[R[6]]]],[11,R[8],E,E,1,[[],[R[6]]]],[11,"into",E,E,1,[[],[U]]],[11,R[9],E,E,1,[[["self"]],[T]]],[11,R[10],E,E,1,[[["self"]],[T]]],[11,R[11],E,E,1,[[["self"]],[R[12]]]],[11,"from",E,E,2,[[[T]],[T]]],[11,R[7],E,E,2,[[[U]],[R[6]]]],[11,R[8],E,E,2,[[],[R[6]]]],[11,"into",E,E,2,[[],[U]]],[11,R[9],E,E,2,[[["self"]],[T]]],[11,R[10],E,E,2,[[["self"]],[T]]],[11,R[11],E,E,2,[[["self"]],[R[12]]]],[11,"from",E,E,3,[[[T]],[T]]],[11,R[7],E,E,3,[[[U]],[R[6]]]],[11,R[8],E,E,3,[[],[R[6]]]],[11,"into",E,E,3,[[],[U]]],[11,R[9],E,E,3,[[["self"]],[T]]],[11,R[10],E,E,3,[[["self"]],[T]]],[11,R[11],E,E,3,[[["self"]],[R[12]]]],[11,"from",E,E,4,[[[T]],[T]]],[11,R[7],E,E,4,[[[U]],[R[6]]]],[11,R[8],E,E,4,[[],[R[6]]]],[11,"into",E,E,4,[[],[U]]],[11,R[9],E,E,4,[[["self"]],[T]]],[11,R[10],E,E,4,[[["self"]],[T]]],[11,R[11],E,E,4,[[["self"]],[R[12]]]],[11,"from",E,E,5,[[[T]],[T]]],[11,R[7],E,E,5,[[[U]],[R[6]]]],[11,R[8],E,E,5,[[],[R[6]]]],[11,"into",E,E,5,[[],[U]]],[11,R[9],E,E,5,[[["self"]],[T]]],[11,R[10],E,E,5,[[["self"]],[T]]],[11,R[11],E,E,5,[[["self"]],[R[12]]]],[11,"from",E,E,6,[[[T]],[T]]],[11,R[7],E,E,6,[[[U]],[R[6]]]],[11,R[8],E,E,6,[[],[R[6]]]],[11,"into",E,E,6,[[],[U]]],[11,R[9],E,E,6,[[["self"]],[T]]],[11,R[10],E,E,6,[[["self"]],[T]]],[11,R[11],E,E,6,[[["self"]],[R[12]]]],[11,"from",E,E,7,[[[T]],[T]]],[11,R[7],E,E,7,[[[U]],[R[6]]]],[11,R[8],E,E,7,[[],[R[6]]]],[11,"into",E,E,7,[[],[U]]],[11,R[9],E,E,7,[[["self"]],[T]]],[11,R[10],E,E,7,[[["self"]],[T]]],[11,R[11],E,E,7,[[["self"]],[R[12]]]],[11,"from",E,E,8,[[[T]],[T]]],[11,R[7],E,E,8,[[[U]],[R[6]]]],[11,R[8],E,E,8,[[],[R[6]]]],[11,"into",E,E,8,[[],[U]]],[11,R[9],E,E,8,[[["self"]],[T]]],[11,R[10],E,E,8,[[["self"]],[T]]],[11,R[11],E,E,8,[[["self"]],[R[12]]]],[11,"from",E,E,9,[[[T]],[T]]],[11,R[7],E,E,9,[[[U]],[R[6]]]],[11,R[8],E,E,9,[[],[R[6]]]],[11,"into",E,E,9,[[],[U]]],[11,R[9],E,E,9,[[["self"]],[T]]],[11,R[10],E,E,9,[[["self"]],[T]]],[11,R[11],E,E,9,[[["self"]],[R[12]]]],[11,"from",E,E,10,[[[T]],[T]]],[11,R[7],E,E,10,[[[U]],[R[6]]]],[11,R[8],E,E,10,[[],[R[6]]]],[11,"into",E,E,10,[[],[U]]],[11,R[9],E,E,10,[[["self"]],[T]]],[11,R[10],E,E,10,[[["self"]],[T]]],[11,R[11],E,E,10,[[["self"]],[R[12]]]],[11,"from",E,E,11,[[[T]],[T]]],[11,R[7],E,E,11,[[[U]],[R[6]]]],[11,R[8],E,E,11,[[],[R[6]]]],[11,"into",E,E,11,[[],[U]]],[11,R[9],E,E,11,[[["self"]],[T]]],[11,R[10],E,E,11,[[["self"]],[T]]],[11,R[11],E,E,11,[[["self"]],[R[12]]]],[11,"from",E,E,12,[[[T]],[T]]],[11,R[7],E,E,12,[[[U]],[R[6]]]],[11,R[8],E,E,12,[[],[R[6]]]],[11,"into",E,E,12,[[],[U]]],[11,R[9],E,E,12,[[["self"]],[T]]],[11,R[10],E,E,12,[[["self"]],[T]]],[11,R[11],E,E,12,[[["self"]],[R[12]]]],[11,"from",E,E,13,[[[T]],[T]]],[11,R[7],E,E,13,[[[U]],[R[6]]]],[11,R[8],E,E,13,[[],[R[6]]]],[11,"into",E,E,13,[[],[U]]],[11,R[9],E,E,13,[[["self"]],[T]]],[11,R[10],E,E,13,[[["self"]],[T]]],[11,R[11],E,E,13,[[["self"]],[R[12]]]],[11,"fmt",E,E,0,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,1,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,2,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,3,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,4,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,5,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,6,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,7,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,8,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,9,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,10,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,11,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,12,[[["self"],[R[13]]],[R[6]]]],[11,"fmt",E,E,13,[[["self"],[R[13]]],[R[6]]]],[11,"clone",E,E,0,[[["self"]],[R[14]]]],[11,"clone",E,E,1,[[["self"]],[R[3]]]],[11,"clone",E,E,2,[[["self"]],[R[15]]]],[11,"clone",E,E,3,[[["self"]],[R[1]]]],[11,"clone",E,E,4,[[["self"]],[R[16]]]],[11,"clone",E,E,5,[[["self"]],[R[17]]]],[11,"clone",E,E,6,[[["self"]],[R[4]]]],[11,"clone",E,E,7,[[["self"]],[R[18]]]],[11,"clone",E,E,8,[[["self"]],[R[19]]]],[11,"clone",E,E,9,[[["self"]],[R[2]]]],[11,"clone",E,E,10,[[["self"]],[R[20]]]],[11,"clone",E,E,11,[[["self"]],[R[21]]]],[11,"clone",E,E,12,[[["self"]],[R[22]]]],[11,"clone",E,E,13,[[["self"]],[R[23]]]]],"p":[[3,R[14]],[3,R[3]],[3,R[15]],[3,R[1]],[3,R[16]],[3,R[17]],[3,R[4]],[3,R[18]],[3,R[19]],[3,R[2]],[3,R[20]],[3,R[21]],[3,R[22]],[3,R[23]]]};
initSearch(searchIndex);addSearchOptions(searchIndex);