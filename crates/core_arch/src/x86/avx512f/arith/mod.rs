//! Intrinsics for Arithmetic Operations
//!
//! This module contains the following intrinsics.
//! - Intrinsics for Addition Operations
//! - Intrinsics for Determining Minimum and Maximum Values
//! - Intrinsics for FP Fused Multiply-Add (FMA) Operations
//! - Intrinsics for Multiplication Operations
//! - Intrinsics for Subtraction Operations
//! - Intrinsics for Short Vector Math Library (SVML) Operations
//! - Intrinsics for Other Mathematics Operations

mod add;
pub use self::add::*;

mod min;
pub use self::min::*;
