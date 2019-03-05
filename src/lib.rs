// `clippy` is a code linting tool for improving code quality by catching
// common mistakes or strange code patterns. If the `clippy` feature is
// provided, it is enabled and all compiler warnings are prohibited.
#![cfg_attr(feature = "clippy", deny(warnings))]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![cfg_attr(feature = "clippy", allow(inline_always))]
#![cfg_attr(feature = "clippy", allow(too_many_arguments))]
#![cfg_attr(feature = "clippy", allow(unreadable_literal))]
#![cfg_attr(feature = "clippy", allow(many_single_char_names))]
#![cfg_attr(feature = "clippy", allow(new_without_default_derive))]
#![cfg_attr(feature = "clippy", allow(string_lit_as_bytes))]
// Force public structures to implement Debug
#![deny(missing_debug_implementations)]

extern crate blake2_rfc;
extern crate byteorder;
extern crate rand;

#[cfg(test)]
pub mod tests;

use std::io::{self, Read, Write};

pub mod bls12_381;

mod wnaf;
pub use self::wnaf::Wnaf;

use std::fmt;
use std::error::Error;

/// An "engine" is a collection of types (fields, elliptic curve groups, etc.)
/// with well-defined relationships. In particular, the G1/G2 curve groups are
/// of prime order `r`, and are equipped with a bilinear pairing function.
pub trait Engine: Sized + 'static + Clone {
    /// This is the scalar field of the G1/G2 groups.
    type Fr: PrimeField + SqrtField;

    /// The projective representation of an element in G1.
    type G1: CurveProjective<
            Engine = Self,
            Base = Self::Fq,
            Scalar = Self::Fr,
            Affine = Self::G1Affine,
        >
        + From<Self::G1Affine>;

    /// The affine representation of an element in G1.
    type G1Affine: CurveAffine<
            Engine = Self,
            Base = Self::Fq,
            Scalar = Self::Fr,
            Projective = Self::G1,
            Pair = Self::G2Affine,
            PairingResult = Self::Fqk,
        >
        + From<Self::G1>;

    /// The projective representation of an element in G2.
    type G2: CurveProjective<
            Engine = Self,
            Base = Self::Fqe,
            Scalar = Self::Fr,
            Affine = Self::G2Affine,
        >
        + From<Self::G2Affine>;

    /// The affine representation of an element in G2.
    type G2Affine: CurveAffine<
            Engine = Self,
            Base = Self::Fqe,
            Scalar = Self::Fr,
            Projective = Self::G2,
            Pair = Self::G1Affine,
            PairingResult = Self::Fqk,
        >
        + From<Self::G2>;

    /// The base field that hosts G1.
    type Fq: PrimeField + SqrtField;

    /// The extension field that hosts G2.
    type Fqe: SqrtField;

    /// The extension field that hosts the target group of the pairing.
    type Fqk: Field;

    /// Perform a miller loop with some number of (G1, G2) pairs.
    fn miller_loop<'a, I>(i: I) -> Self::Fqk
    where
        I: IntoIterator<
            Item = &'a (
                &'a <Self::G1Affine as CurveAffine>::Prepared,
                &'a <Self::G2Affine as CurveAffine>::Prepared,
            ),
        >;

    /// Perform final exponentiation of the result of a miller loop.
    fn final_exponentiation(&Self::Fqk) -> Option<Self::Fqk>;

    /// Performs a complete pairing operation `(p, q)`.
    fn pairing<G1, G2>(p: G1, q: G2) -> Self::Fqk
    where
        G1: Into<Self::G1Affine>,
        G2: Into<Self::G2Affine>,
    {
        Self::final_exponentiation(&Self::miller_loop(
            [(&(p.into().prepare()), &(q.into().prepare()))].into_iter(),
        )).unwrap()
    }
}

/// Projective representation of an elliptic curve point guaranteed to be
/// in the correct prime order subgroup.
pub trait CurveProjective:
    PartialEq
    + Eq
    + Sized
    + Copy
    + Clone
    + Send
    + Sync
    + fmt::Debug
    + fmt::Display
    + rand::Rand
    + 'static
{
    type Engine: Engine<Fr = Self::Scalar>;
    type Scalar: PrimeField + SqrtField;
    type Base: SqrtField;
    type Affine: CurveAffine<Projective = Self, Scalar = Self::Scalar>;

    /// Returns the additive identity.
    fn zero() -> Self;

    /// Returns a fixed generator of unknown exponent.
    fn one() -> Self;

    /// Determines if this point is the point at infinity.
    fn is_zero(&self) -> bool;

    /// Normalizes a slice of projective elements so that
    /// conversion to affine is cheap.
    fn batch_normalization(v: &mut [Self]);

    /// Checks if the point is already "normalized" so that
    /// cheap affine conversion is possible.
    fn is_normalized(&self) -> bool;

    /// Doubles this element.
    fn double(&mut self);

    /// Adds another element to this element.
    fn add_assign(&mut self, other: &Self);

    /// Subtracts another element from this element.
    fn sub_assign(&mut self, other: &Self) {
        let mut tmp = *other;
        tmp.negate();
        self.add_assign(&tmp);
    }

    /// Adds an affine element to this element.
    fn add_assign_mixed(&mut self, other: &Self::Affine);

    /// Negates this element.
    fn negate(&mut self);

    /// Performs scalar multiplication of this element.
    fn mul_assign<S: Into<<Self::Scalar as PrimeField>::Repr>>(&mut self, other: S);

    /// Converts this element into its affine representation.
    fn into_affine(&self) -> Self::Affine;

    /// Recommends a wNAF window table size given a scalar. Always returns a number
    /// between 2 and 22, inclusive.
    fn recommended_wnaf_for_scalar(scalar: <Self::Scalar as PrimeField>::Repr) -> usize;

    /// Recommends a wNAF window size given the number of scalars you intend to multiply
    /// a base by. Always returns a number between 2 and 22, inclusive.
    fn recommended_wnaf_for_num_scalars(num_scalars: usize) -> usize;

    /// Given a message, hash into a random element of the prime-order subgroup.
    fn hash(&[u8]) -> Self;
}

/// Affine representation of an elliptic curve point guaranteed to be
/// in the correct prime order subgroup.
pub trait CurveAffine:
    Copy + Clone + Sized + Send + Sync + fmt::Debug + fmt::Display + PartialEq + Eq + 'static
{
    type Engine: Engine<Fr = Self::Scalar>;
    type Scalar: PrimeField + SqrtField;
    type Base: SqrtField;
    type Projective: CurveProjective<Affine = Self, Scalar = Self::Scalar>;
    type Prepared: Clone + Send + Sync + 'static;
    type Uncompressed: EncodedPoint<Affine = Self>;
    type Compressed: EncodedPoint<Affine = Self>;
    type Pair: CurveAffine<Pair = Self>;
    type PairingResult: Field;

    /// Returns the additive identity.
    fn zero() -> Self;

    /// Returns a fixed generator of unknown exponent.
    fn one() -> Self;

    /// Determines if this point represents the point at infinity; the
    /// additive identity.
    fn is_zero(&self) -> bool;

    /// Negates this element.
    fn negate(&mut self);

    /// Performs scalar multiplication of this element with mixed addition.
    fn mul<S: Into<<Self::Scalar as PrimeField>::Repr>>(&self, other: S) -> Self::Projective;

    /// Prepares this element for pairing purposes.
    fn prepare(&self) -> Self::Prepared;

    /// Perform a pairing
    fn pairing_with(&self, other: &Self::Pair) -> Self::PairingResult;

    /// Converts this element into its affine representation.
    fn into_projective(&self) -> Self::Projective;

    /// Converts this element into its compressed encoding, so long as it's not
    /// the point at infinity.
    fn into_compressed(&self) -> Self::Compressed {
        <Self::Compressed as EncodedPoint>::from_affine(*self)
    }

    /// Converts this element into its uncompressed encoding, so long as it's not
    /// the point at infinity.
    fn into_uncompressed(&self) -> Self::Uncompressed {
        <Self::Uncompressed as EncodedPoint>::from_affine(*self)
    } 
}

/// An encoded elliptic curve point, which should essentially wrap a `[u8; N]`.
pub trait EncodedPoint:
    Sized + Send + Sync + AsRef<[u8]> + AsMut<[u8]> + Clone + Copy + 'static
{
    type Affine: CurveAffine;

    /// Creates an empty representation.
    fn empty() -> Self;

    /// Returns the number of bytes consumed by this representation.
    fn size() -> usize;

    /// Converts an `EncodedPoint` into a `CurveAffine` element,
    /// if the encoding represents a valid element.
    fn into_affine(&self) -> Result<Self::Affine, GroupDecodingError>;

    /// Converts an `EncodedPoint` into a `CurveAffine` element,
    /// without guaranteeing that the encoding represents a valid
    /// element. This is useful when the caller knows the encoding is
    /// valid already.
    ///
    /// If the encoding is invalid, this can break API invariants,
    /// so caution is strongly encouraged.
    fn into_affine_unchecked(&self) -> Result<Self::Affine, GroupDecodingError>;

    /// Creates an `EncodedPoint` from an affine point, as long as the
    /// point is not the point at infinity.
    fn from_affine(affine: Self::Affine) -> Self;
}

/// This trait represents an element of a field.
pub trait Field:
    Sized + Eq + Copy + Clone + Send + Sync + fmt::Debug + fmt::Display + 'static + rand::Rand
{
    /// Returns the zero element of the field, the additive identity.
    fn zero() -> Self;

    /// Returns the one element of the field, the multiplicative identity.
    fn one() -> Self;

    /// Returns true iff this element is zero.
    fn is_zero(&self) -> bool;

    /// Squares this element.
    fn square(&mut self);

    /// Doubles this element.
    fn double(&mut self);

    /// Negates this element.
    fn negate(&mut self);

    /// Adds another element to this element.
    fn add_assign(&mut self, other: &Self);

    /// Subtracts another element from this element.
    fn sub_assign(&mut self, other: &Self);

    /// Multiplies another element by this element.
    fn mul_assign(&mut self, other: &Self);

    /// Computes the multiplicative inverse of this element, if nonzero.
    fn inverse(&self) -> Option<Self>;

    /// Exponentiates this element by a power of the base prime modulus via
    /// the Frobenius automorphism.
    fn frobenius_map(&mut self, power: usize);

    /// Exponentiates this element by a number represented with `u64` limbs,
    /// least significant digit first.
    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();

        let mut found_one = false;

        for i in BitIterator::new(exp) {
            if found_one {
                res.square();
            } else {
                found_one = i;
            }

            if i {
                res.mul_assign(self);
            }
        }

        res
    }
}

/// This trait represents an element of a field that has a square root operation described for it.
pub trait SqrtField: Field {
    /// Returns the Legendre symbol of the field element.
    fn legendre(&self) -> LegendreSymbol;

    /// Returns the square root of the field element, if it is
    /// quadratic residue.
    fn sqrt(&self) -> Option<Self>;
}

/// This trait represents a wrapper around a biginteger which can encode any element of a particular
/// prime field. It is a smart wrapper around a sequence of `u64` limbs, least-significant digit
/// first.
pub trait PrimeFieldRepr:
    Sized
    + Copy
    + Clone
    + Eq
    + Ord
    + Send
    + Sync
    + Default
    + fmt::Debug
    + fmt::Display
    + 'static
    + rand::Rand
    + AsRef<[u64]>
    + AsMut<[u64]>
    + From<u64>
{
    /// Subtract another represetation from this one.
    fn sub_noborrow(&mut self, other: &Self);

    /// Add another representation to this one.
    fn add_nocarry(&mut self, other: &Self);

    /// Compute the number of bits needed to encode this number. Always a
    /// multiple of 64.
    fn num_bits(&self) -> u32;

    /// Returns true iff this number is zero.
    fn is_zero(&self) -> bool;

    /// Returns true iff this number is odd.
    fn is_odd(&self) -> bool;

    /// Returns true iff this number is even.
    fn is_even(&self) -> bool;

    /// Performs a rightwise bitshift of this number, effectively dividing
    /// it by 2.
    fn div2(&mut self);

    /// Performs a rightwise bitshift of this number by some amount.
    fn shr(&mut self, amt: u32);

    /// Performs a leftwise bitshift of this number, effectively multiplying
    /// it by 2. Overflow is ignored.
    fn mul2(&mut self);

    /// Performs a leftwise bitshift of this number by some amount.
    fn shl(&mut self, amt: u32);

    /// Writes this `PrimeFieldRepr` as a big endian integer.
    fn write_be<W: Write>(&self, mut writer: W) -> io::Result<()> {
        use byteorder::{BigEndian, WriteBytesExt};

        for digit in self.as_ref().iter().rev() {
            writer.write_u64::<BigEndian>(*digit)?;
        }

        Ok(())
    }

    /// Reads a big endian integer into this representation.
    fn read_be<R: Read>(&mut self, mut reader: R) -> io::Result<()> {
        use byteorder::{BigEndian, ReadBytesExt};

        for digit in self.as_mut().iter_mut().rev() {
            *digit = reader.read_u64::<BigEndian>()?;
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub enum LegendreSymbol {
    Zero = 0,
    QuadraticResidue = 1,
    QuadraticNonResidue = -1,
}

/// An error that may occur when trying to interpret a `PrimeFieldRepr` as a
/// `PrimeField` element.
#[derive(Debug)]
pub enum PrimeFieldDecodingError {
    /// The encoded value is not in the field
    NotInField(String),
}

impl Error for PrimeFieldDecodingError {
    fn description(&self) -> &str {
        match *self {
            PrimeFieldDecodingError::NotInField(..) => "not an element of the field",
        }
    }
}

impl fmt::Display for PrimeFieldDecodingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            PrimeFieldDecodingError::NotInField(ref repr) => {
                write!(f, "{} is not an element of the field", repr)
            }
        }
    }
}

/// An error that may occur when trying to decode an `EncodedPoint`.
#[derive(Debug)]
pub enum GroupDecodingError {
    /// The coordinate(s) do not lie on the curve.
    NotOnCurve,
    /// The element is not part of the r-order subgroup.
    NotInSubgroup,
    /// One of the coordinates could not be decoded
    CoordinateDecodingError(&'static str, PrimeFieldDecodingError),
    /// The compression mode of the encoded element was not as expected
    UnexpectedCompressionMode,
    /// The encoding contained bits that should not have been set
    UnexpectedInformation,
}

impl Error for GroupDecodingError {
    fn description(&self) -> &str {
        match *self {
            GroupDecodingError::NotOnCurve => "coordinate(s) do not lie on the curve",
            GroupDecodingError::NotInSubgroup => "the element is not part of an r-order subgroup",
            GroupDecodingError::CoordinateDecodingError(..) => "coordinate(s) could not be decoded",
            GroupDecodingError::UnexpectedCompressionMode => {
                "encoding has unexpected compression mode"
            }
            GroupDecodingError::UnexpectedInformation => "encoding has unexpected information",
        }
    }
}

impl fmt::Display for GroupDecodingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            GroupDecodingError::CoordinateDecodingError(description, ref err) => {
                write!(f, "{} decoding error: {}", description, err)
            }
            _ => write!(f, "{}", self.description()),
        }
    }
}

/// This represents an element of a prime field.
pub trait PrimeField: Field {
    /// The prime field can be converted back and forth into this biginteger
    /// representation.
    type Repr: PrimeFieldRepr + From<Self>;

    /// Interpret a string of numbers as a (congruent) prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    fn from_str(s: &str) -> Option<Self> {
        if s.is_empty() {
            return None;
        }

        if s == "0" {
            return Some(Self::zero());
        }

        let mut res = Self::zero();

        let ten = Self::from_repr(Self::Repr::from(10)).unwrap();

        let mut first_digit = true;

        for c in s.chars() {
            match c.to_digit(10) {
                Some(c) => {
                    if first_digit {
                        if c == 0 {
                            return None;
                        }

                        first_digit = false;
                    }

                    res.mul_assign(&ten);
                    res.add_assign(&Self::from_repr(Self::Repr::from(u64::from(c))).unwrap());
                }
                None => {
                    return None;
                }
            }
        }

        Some(res)
    }

    /// Convert this prime field element into a biginteger representation.
    fn from_repr(Self::Repr) -> Result<Self, PrimeFieldDecodingError>;

    /// Convert a biginteger representation into a prime field element, if
    /// the number is an element of the field.
    fn into_repr(&self) -> Self::Repr;

    /// Returns the field characteristic; the modulus.
    fn char() -> Self::Repr;

    /// How many bits are needed to represent an element of this field.
    const NUM_BITS: u32;

    /// How many bits of information can be reliably stored in the field element.
    const CAPACITY: u32;

    /// Returns the multiplicative generator of `char()` - 1 order. This element
    /// must also be quadratic nonresidue.
    fn multiplicative_generator() -> Self;

    /// 2^s * t = `char()` - 1 with t odd.
    const S: u32;

    /// Returns the 2^s root of unity computed by exponentiating the `multiplicative_generator()`
    /// by t.
    fn root_of_unity() -> Self;
}

#[derive(Debug)]
pub struct BitIterator<E> {
    t: E,
    n: usize,
}

impl<E: AsRef<[u64]>> BitIterator<E> {
    pub fn new(t: E) -> Self {
        let n = t.as_ref().len() * 64;

        BitIterator { t, n }
    }
}

impl<E: AsRef<[u64]>> Iterator for BitIterator<E> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.n == 0 {
            None
        } else {
            self.n -= 1;
            let part = self.n / 64;
            let bit = self.n - (64 * part);

            Some(self.t.as_ref()[part] & (1 << bit) > 0)
        }
    }
}

#[test]
fn test_bit_iterator() {
    let mut a = BitIterator::new([0xa953d79b83f6ab59, 0x6dea2059e200bd39]);
    let expected = "01101101111010100010000001011001111000100000000010111101001110011010100101010011110101111001101110000011111101101010101101011001";

    for e in expected.chars() {
        assert!(a.next().unwrap() == (e == '1'));
    }

    assert!(a.next().is_none());

    let expected = "1010010101111110101010000101101011101000011101110101001000011001100100100011011010001011011011010001011011101100110100111011010010110001000011110100110001100110011101101000101100011100100100100100001010011101010111110011101011000011101000111011011101011001";

    let mut a = BitIterator::new([
        0x429d5f3ac3a3b759,
        0xb10f4c66768b1c92,
        0x92368b6d16ecd3b4,
        0xa57ea85ae8775219,
    ]);

    for e in expected.chars() {
        assert!(a.next().unwrap() == (e == '1'));
    }

    assert!(a.next().is_none());
}

#[cfg(not(feature = "expose-arith"))]
use self::arith_impl::*;

#[cfg(feature = "expose-arith")]
pub use self::arith_impl::*;

#[cfg(feature = "u128-support")]
mod arith_impl {
    /// Calculate a - b - borrow, returning the result and modifying
    /// the borrow value.
    #[inline(always)]
    pub fn sbb(a: u64, b: u64, borrow: &mut u64) -> u64 {
        let tmp = (1u128 << 64) + u128::from(a) - u128::from(b) - u128::from(*borrow);

        *borrow = if tmp >> 64 == 0 { 1 } else { 0 };

        tmp as u64
    }

    /// Calculate a + b + carry, returning the sum and modifying the
    /// carry value.
    #[inline(always)]
    pub fn adc(a: u64, b: u64, carry: &mut u64) -> u64 {
        let tmp = u128::from(a) + u128::from(b) + u128::from(*carry);

        *carry = (tmp >> 64) as u64;

        tmp as u64
    }

    /// Calculate a + (b * c) + carry, returning the least significant digit
    /// and setting carry to the most significant digit.
    #[inline(always)]
    pub fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
        let tmp = (u128::from(a)) + u128::from(b) * u128::from(c) + u128::from(*carry);

        *carry = (tmp >> 64) as u64;

        tmp as u64
    }
}

#[cfg(not(feature = "u128-support"))]
mod arith_impl {
    #[inline(always)]
    fn split_u64(i: u64) -> (u64, u64) {
        (i >> 32, i & 0xFFFFFFFF)
    }

    #[inline(always)]
    fn combine_u64(hi: u64, lo: u64) -> u64 {
        (hi << 32) | lo
    }

    /// Calculate a - b - borrow, returning the result and modifying
    /// the borrow value.
    #[inline(always)]
    pub fn sbb(a: u64, b: u64, borrow: &mut u64) -> u64 {
        let (a_hi, a_lo) = split_u64(a);
        let (b_hi, b_lo) = split_u64(b);
        let (b, r0) = split_u64((1 << 32) + a_lo - b_lo - *borrow);
        let (b, r1) = split_u64((1 << 32) + a_hi - b_hi - ((b == 0) as u64));

        *borrow = (b == 0) as u64;

        combine_u64(r1, r0)
    }

    /// Calculate a + b + carry, returning the sum and modifying the
    /// carry value.
    #[inline(always)]
    pub fn adc(a: u64, b: u64, carry: &mut u64) -> u64 {
        let (a_hi, a_lo) = split_u64(a);
        let (b_hi, b_lo) = split_u64(b);
        let (carry_hi, carry_lo) = split_u64(*carry);

        let (t, r0) = split_u64(a_lo + b_lo + carry_lo);
        let (t, r1) = split_u64(t + a_hi + b_hi + carry_hi);

        *carry = t;

        combine_u64(r1, r0)
    }

    /// Calculate a + (b * c) + carry, returning the least significant digit
    /// and setting carry to the most significant digit.
    #[inline(always)]
    pub fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
        /*
                                [  b_hi  |  b_lo  ]
                                [  c_hi  |  c_lo  ] *
        -------------------------------------------
                                [  b_lo  *  c_lo  ] <-- w
                       [  b_hi  *  c_lo  ]          <-- x
                       [  b_lo  *  c_hi  ]          <-- y
             [   b_hi  *  c_lo  ]                   <-- z
                                [  a_hi  |  a_lo  ]
                                [  C_hi  |  C_lo  ]
        */

        let (a_hi, a_lo) = split_u64(a);
        let (b_hi, b_lo) = split_u64(b);
        let (c_hi, c_lo) = split_u64(c);
        let (carry_hi, carry_lo) = split_u64(*carry);

        let (w_hi, w_lo) = split_u64(b_lo * c_lo);
        let (x_hi, x_lo) = split_u64(b_hi * c_lo);
        let (y_hi, y_lo) = split_u64(b_lo * c_hi);
        let (z_hi, z_lo) = split_u64(b_hi * c_hi);

        let (t, r0) = split_u64(w_lo + a_lo + carry_lo);
        let (t, r1) = split_u64(t + w_hi + x_lo + y_lo + a_hi + carry_hi);
        let (t, r2) = split_u64(t + x_hi + y_hi + z_lo);
        let (_, r3) = split_u64(t + z_hi);

        *carry = combine_u64(r3, r2);

        combine_u64(r1, r0)
    }
}




#[repr(C)] #[derive(Debug)] pub struct ArrayStruct<T> { d: *mut T, len: usize }

const FIELD_SIZE: usize = 48;
const G1_SIZE: usize = 2*FIELD_SIZE;
const G1_SIZE_COMPRESSED: usize = G1_SIZE/2;
const G2_SIZE: usize = 2*G1_SIZE;
const G2_SIZE_COMPRESSED: usize = G2_SIZE/2;
const GT_SIZE: usize = 12*FIELD_SIZE;
const SCALAR_SIZE: usize = 256/(8*8);

use bls12_381::*;    
use std::panic;

fn g1_size(compressed: bool) -> usize { if compressed { G1_SIZE_COMPRESSED } else { G1_SIZE } }
fn g2_size(compressed: bool) -> usize { if compressed { G2_SIZE_COMPRESSED } else { G2_SIZE } } 

fn copy_array(from: &[u8], to: &mut [u8]) {
    assert!(from.len() == to.len());
    for x in 0 .. from.len() { to[x] = from[x] }
}

fn g1_from_raw(d: &[u8]) -> G1Affine { 
    match if d.len() == g1_size(true) { 
        let mut a: [u8 ; G1_SIZE_COMPRESSED] = [0 ; G1_SIZE_COMPRESSED]; copy_array(d, &mut a);
        G1Compressed(a).into_affine() 
    } else { 
        let mut a: [u8 ; G1_SIZE] = [0 ; G1_SIZE]; copy_array(d, &mut a);
        G1Uncompressed(a).into_affine() 
    } 
    { Ok(a) => a, Err(why) => panic!("{:?}\nd.len() = {:X}", why, d.len()) }
}

fn g2_from_raw(d: &[u8]) -> G2Affine {
    match if d.len() == g2_size(true) {
        let mut a: [u8 ; G2_SIZE_COMPRESSED] = [0 ; G2_SIZE_COMPRESSED]; copy_array(d, &mut a); 
        G2Compressed(a).into_affine()
    } else {
        let mut a: [u8 ; G2_SIZE] = [0 ; G2_SIZE]; copy_array(d, &mut a);
        G2Uncompressed(a).into_affine()
    }
    { Ok(a) => a, Err(why) => panic!("{:?}\nd.len() = {:X}", why, d.len()) }
}
/*
fn print_raw8(d: &[u8], prefix: &str) { print!("\n{}:{}(", d.len(), prefix); 
    for x in 0 .. d.len() { print!("{:X}", d[x]) };
    print!(") ");
    io::stdout().flush().ok().expect("Could not flush stdout");
}
fn print_raw64(d: &[u64], prefix: &str) { print!("\n{}:{}(", d.len(), prefix); 
    for x in 0 .. d.len() { print!("{:X}", d[x]) };
    print!(") ");
    io::stdout().flush().ok().expect("Could not flush stdout");
}
*/
fn g1_to_raw(g: G1Affine, result: ArrayStruct<u8>) {
    assert!(g.is_on_curve());
    assert!(g.is_in_correct_subgroup_assuming_on_curve());
    let mut d = mut_array(result);
    if d.len() == g1_size(true) { copy_array(&G1Compressed::from_affine(g).0, &mut d); } 
    else { copy_array(&G1Uncompressed::from_affine(g).0, &mut d); }
}

fn g2_to_raw(g: G2Affine, result: ArrayStruct<u8>) {
    assert!(g.is_on_curve());
    assert!(g.is_in_correct_subgroup_assuming_on_curve());
    let mut d = mut_array(result);
    if d.len() == g2_size(true) { copy_array(&G2Compressed::from_affine(g).0, &mut d); } 
    else { copy_array(&G2Uncompressed::from_affine(g).0, &mut d); }
}

fn gt(a: ArrayStruct<u64>) -> Fq12 { 
    assert!(a.len == GT_SIZE/8);
    let d = mut_array(a);
    assert!(d.len() == GT_SIZE/8);
    let mut g: Fq12 = Fq12::zero();
    g.stream(d, 0, false);
    return g
}

fn gt_to_raw(mut g: Fq12, result: ArrayStruct<u64>) -> () { 
    assert!(result.len == GT_SIZE/8);
    let d = mut_array(result);
    assert!(d.len() == GT_SIZE/8);
    g.stream(d, 0, true);
}

fn scalar_from_raw(s: &[u64]) -> bls12_381::Fr { 
    assert!(s.len() == SCALAR_SIZE);
    bls12_381::Fr{0: bls12_381::FrRepr([s[0], s[1], s[2], s[3]])} 
}

fn mut_array<'a, T>(a_: ArrayStruct<T>) -> &'a mut [T] { 
    unsafe { std::slice::from_raw_parts_mut(a_.d, a_.len) } 
}
fn const_array<'a, T>(a_: ArrayStruct<T>, x: usize) -> &'a [T] { assert!(a_.len == x);
    unsafe { std::slice::from_raw_parts(a_.d, a_.len) } 
}

fn g1affine(a: ArrayStruct<u8>) -> G1Affine { g1_from_raw(const_array(a, g1_size(true))) }
fn g2affine(a: ArrayStruct<u8>) -> G2Affine { g2_from_raw(const_array(a, g2_size(true))) }
//fn gt(a: ArrayStruct<u64>) -> bls12_381::Fq12 { gt_from_raw(a) }
fn scalar(a: ArrayStruct<u64>) -> bls12_381::Fr { scalar_from_raw(const_array(a, SCALAR_SIZE)) }

#[no_mangle] pub extern "C" fn g1_add(a: ArrayStruct<u8>, b: ArrayStruct<u8>, r: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    let mut x: G1 = g1affine(a).into_projective();
    let y: G1Affine = g1affine(b);
    x.add_assign_mixed(&y);
    g1_to_raw(x.into_affine(), r);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g2_add(a: ArrayStruct<u8>, b: ArrayStruct<u8>, r: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    let mut x: G2 = g2affine(a).into_projective();
    let y: G2Affine = g2affine(b);
    x.add_assign_mixed(&y);
    g2_to_raw(x.into_affine(), r);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g1_get_one(g: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    g1_to_raw(G1Affine::get_generator(), g);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g2_get_one(g: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    g2_to_raw(G2Affine::get_generator(), g);
}).is_ok(); }

#[no_mangle] pub extern "C" fn gt_get_one(g: ArrayStruct<u64>) -> bool { return panic::catch_unwind(|| {
    gt_to_raw(Fq12::one(), g);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g1_get_zero(g: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    g1_to_raw(G1Affine::zero(), g);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g2_get_zero(g: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    g2_to_raw(G2Affine::zero(), g);
}).is_ok(); }

#[no_mangle] pub extern "C" fn gt_get_zero(g: ArrayStruct<u64>) -> bool { return panic::catch_unwind(|| {
    gt_to_raw(Fq12::zero(), g);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g1_mul(g: ArrayStruct<u8>, p: ArrayStruct<u64>, result: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    g1_to_raw(g1affine(g).mul(scalar(p)).into_affine(), result);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g2_mul(g: ArrayStruct<u8>, p: ArrayStruct<u64>, result: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    g2_to_raw(g2affine(g).mul(scalar(p)).into_affine(), result);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g1_neg(g: ArrayStruct<u8>, result: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| { 
    let mut z = g1affine(g);
    z.negate();
    g1_to_raw(z, result);
}).is_ok(); }

#[no_mangle] pub extern "C" fn g2_neg(g: ArrayStruct<u8>, result: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {
    let mut z = g2affine(g);
    z.negate();
    g2_to_raw(z, result);
}).is_ok(); }

#[no_mangle] pub extern "C" fn gt_mul(a: ArrayStruct<u64>, b: ArrayStruct<u64>, result: ArrayStruct<u64>) -> bool { return panic::catch_unwind(|| {
    let mut z = gt(a);
    z.mul_assign(&gt(b));
    gt_to_raw(z, result);
}).is_ok(); }

#[no_mangle] pub extern "C" fn gt_inverse(g: ArrayStruct<u64>, result: ArrayStruct<u64>) -> bool { return panic::catch_unwind(|| {
    gt_to_raw(match gt(g).inverse() { Some(v) => v, None => Fq12::zero() }, result); 
}).is_ok(); }

#[no_mangle] pub extern "C" fn pairing(g1_: ArrayStruct<u8>, g2_: ArrayStruct<u8>, gt: ArrayStruct<u64>) -> bool { return panic::catch_unwind(|| {
    gt_to_raw(bls12_381::Bls12::pairing(g1affine(g1_), g2affine(g2_)), gt);
}).is_ok(); }
  
#[no_mangle] pub extern "C" fn hash_to_g1(data: ArrayStruct<u8>, result: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| {   
    assert!(data.len == 32); 
    let h = G1::hash(mut_array(data)).into_affine();
    assert!(!h.is_zero());
    assert!(h.is_on_curve());
    assert!(h.is_in_correct_subgroup_assuming_on_curve());
    g1_to_raw(h, result);
}).is_ok(); }

#[no_mangle] pub extern "C" fn hash_to_g2(data: ArrayStruct<u8>, result: ArrayStruct<u8>) -> bool { return panic::catch_unwind(|| { 
    assert!(data.len == 32); 
    let h = G2::hash(mut_array(data)).into_affine();
    assert!(!h.is_zero());
    assert!(h.is_on_curve());
    assert!(h.is_in_correct_subgroup_assuming_on_curve());
    g2_to_raw(h, result);
}).is_ok(); }
