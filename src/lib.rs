#![allow(clippy::missing_safety_doc)]
#![feature(maybe_uninit_extra)]
#![feature(min_const_generics)]
#![feature(vec_into_raw_parts)]

use core::mem::MaybeUninit;
use core::ops;
use rand::Rng;
use std::io::{Result as IoResult, Write};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    __m256i, _mm256_load_si256, _mm256_or_si256, _mm256_setzero_si256, _mm256_store_si256,
    _mm256_testc_si256,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::mem::transmute;

use core::fmt;

#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Error {
    MissingHeader,
    InvalidHeader,
    InvalidParameter,
    TooLarge,
    Truncated,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Error::MissingHeader => "Missing header block",
            Error::InvalidHeader => "Invalid header block",
            Error::InvalidParameter => "Invalid filter parameter",
            Error::TooLarge => "Filter is too large",
            Error::Truncated => "Filter data is truncated",
        };
        formatter.write_str(&message)
    }
}

#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct Block {
    data: [u32; Block::WORDS],
}

impl Block {
    const WORDS: usize = 16;
    const BYTES: usize = 64;
    const BITS: usize = 512;

    pub fn random<R: Rng>(ones: u32, rng: &mut R) -> Block {
        debug_assert!(ones <= Block::BITS as u32);

        let mut one_bits: [MaybeUninit<u16>; Block::BITS] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for (bit, ptr) in one_bits[0..(ones as usize)].iter_mut().enumerate() {
            ptr.write(bit as u16);
        }

        let mut i = (ones as usize) + 1;
        let mut w = (rng.gen::<f64>().ln() / (ones as f64)).exp();

        while i < Block::BITS {
            i += (rng.gen::<f64>().ln() / (1.0 - w).ln()).floor() as usize;

            if i < Block::BITS {
                let j = rng.gen_range(0..(ones as usize));
                one_bits[j].write(i as u16);
                w *= (rng.gen::<f64>().ln() / (ones as f64)).exp();
            }
        }

        let mut data = [0u32; Block::WORDS];

        for ptr in one_bits[0..(ones as usize)].iter_mut() {
            let index = unsafe { ptr.read() };
            data[(index as usize) / 32] |= 1 << ((index as usize) % 32);
        }

        Block { data }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn load_mm256i(&self, index: usize) -> __m256i {
        debug_assert!(index < 2);
        _mm256_load_si256(self.data.as_ptr().add(index * 8) as *const __m256i)
    }

    fn merge(&mut self, block: &Block) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            let x = _mm256_or_si256(self.load_mm256i(0), block.load_mm256i(0));
            let y = _mm256_or_si256(self.load_mm256i(1), block.load_mm256i(1));

            _mm256_store_si256(self.data.as_mut_ptr() as *mut __m256i, x);
            _mm256_store_si256(self.data.as_mut_ptr().add(8) as *mut __m256i, y);
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        for i in 0..Block::WORDS {
            self.data[i] |= block.data[i];
        }
    }

    fn merge_many<'a, I: Iterator<Item = &'a Block>>(blocks: I) -> Block {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            let mut x = _mm256_setzero_si256();
            let mut y = _mm256_setzero_si256();

            for block in blocks {
                x = _mm256_or_si256(x, block.load_mm256i(0));
                y = _mm256_or_si256(y, block.load_mm256i(1));
            }

            let mut data: [MaybeUninit<u32>; Block::WORDS] = MaybeUninit::uninit().assume_init();

            _mm256_store_si256(data.as_mut_ptr() as *mut __m256i, x);
            _mm256_store_si256(data.as_mut_ptr().add(8) as *mut __m256i, x);

            Block {
                data: transmute(data),
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut blocks = blocks;
            if let Some(first) = blocks.next() {
                let mut merged = first.clone();
                for block in blocks {
                    for i in 0..Block::WORDS {
                        merged.data[i] |= block.data[i];
                    }
                }
                merged
            } else {
                Block::default()
            }
        }
    }

    fn contains(&self, block: &Block) -> bool {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        for i in 0..2 {
            if unsafe { _mm256_testc_si256(self.load_mm256i(i), block.load_mm256i(i)) } == 0 {
                return false;
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        for i in 0..Block::WORDS {
            if ((!self.data[i]) & block.data[i]) != 0 {
                return false;
            }
        }

        true
    }
}

mod private_filter {
    use crate::{extract_index_from_hash, Block, Error};
    use core::slice;
    use rand::Rng;

    const MAGIC_NUMBER: u32 = 0xf100f001;

    pub trait Filter<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> {
        fn blocks(&self) -> u32;
        fn table_entries(&self) -> u32;

        fn buffer(&self) -> *const Block;
        fn buffer_mut(&mut self) -> *mut Block;

        // Key assumption: buffer_len overflows neither u32 nor platform usize
        // (enforced with buffer_len_assert_bounds in the constructors of individual Filter implementations)
        fn buffer_len(&self) -> u32 {
            1 + self.blocks() + self.table_entries()
        }

        fn buffer_len_assert_bounds(blocks: u32, table_entries: u32) -> Result<u32, Error> {
            if blocks == 0 || table_entries == 0 {
                return Err(Error::InvalidParameter);
            }

            let len = 1u32
                .checked_add(blocks)
                .and_then(|x| x.checked_add(table_entries))
                .map_or(Err(Error::TooLarge), Ok)?;

            #[cfg(target_pointer_width = "16")]
            if len > usize::MAX {
                return Err(Error::TooLarge);
            }

            Ok(len)
        }

        unsafe fn header_mut(&mut self) -> *mut Block {
            &mut *self.buffer_mut()
        }

        unsafe fn table_entry(&self, index: u32) -> *const Block {
            debug_assert!(index < self.table_entries());
            &*self.buffer().add(1 + index as usize)
        }

        unsafe fn table_entry_mut(&mut self, index: u32) -> *mut Block {
            debug_assert!(index < self.table_entries());
            &mut *self.buffer_mut().add(1 + index as usize)
        }

        unsafe fn block(&self, index: u32) -> *const Block {
            debug_assert!(index < self.blocks());
            &*self
                .buffer()
                .add((1 + self.table_entries() + index) as usize)
        }

        unsafe fn block_mut(&mut self, index: u32) -> *mut Block {
            debug_assert!(index < self.blocks());
            &mut *self
                .buffer_mut()
                .add((1 + self.table_entries() + index) as usize)
        }

        fn initialize_new<R: Rng>(&mut self, ones: u32, rng: &mut R) -> Result<(), Error> {
            if ones == 0 || ones > Block::BITS as u32 {
                return Err(Error::InvalidParameter);
            }

            unsafe {
                self.header_mut().write(self.compute_header_block());
            }

            for i in 0..self.table_entries() {
                unsafe {
                    self.table_entry_mut(i).write(Block::random(ones, rng));
                }
            }

            for i in 0..self.blocks() {
                unsafe {
                    self.block_mut(i).write(Block::default());
                }
            }

            Ok(())
        }

        fn compute_header_block(&self) -> Block {
            Block {
                data: [
                    MAGIC_NUMBER.to_le(),
                    BLOCKS_PER_ELEM.to_le(),
                    TABLE_ENTRIES_PER_BLOCK.to_le(),
                    self.blocks().to_le(),
                    self.table_entries().to_le(),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        }

        fn parse_header_block(block: &Block) -> Result<(u32, u32), Error> {
            if u32::from_le(block.data[0]) != MAGIC_NUMBER {
                return Err(Error::InvalidHeader);
            }

            if u32::from_le(block.data[1]) != BLOCKS_PER_ELEM {
                return Err(Error::InvalidParameter);
            }

            if u32::from_le(block.data[2]) != TABLE_ENTRIES_PER_BLOCK {
                return Err(Error::InvalidParameter);
            }

            let blocks = u32::from_le(block.data[3]);
            let table_entries = u32::from_le(block.data[4]);
            Ok((blocks, table_entries))
        }

        fn query_block_for_hash(&self, hash: u64) -> (Block, u64) {
            let mut hash = hash;

            let to_merge = (0..TABLE_ENTRIES_PER_BLOCK).map(|_| {
                let (index, hash2) = extract_index_from_hash(hash, self.table_entries());
                hash = hash2;
                unsafe { &*self.table_entry(index) }
            });

            let block = Block::merge_many(to_merge);
            (block, hash)
        }

        fn to_bytes(&self) -> &[u8] {
            let bytes = Block::BYTES * self.buffer_len() as usize;
            unsafe { slice::from_raw_parts(self.buffer() as *const u8, bytes) }
        }
    }
}

use private_filter::Filter as PrivateFilter;

trait Filter<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> {
    fn hash_bits(&self) -> u32;

    fn insert_hash(&mut self, hash: u64);
    fn contains_hash(&self, hash: u64) -> bool;

    fn insert<A: Hashable>(&mut self, value: A);
    fn contains<A: Hashable>(&self, value: A) -> bool;

    fn serialize<W: Write>(&self, output: &mut W) -> IoResult<()>;
}

impl<
        F: PrivateFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>,
        const BLOCKS_PER_ELEM: u32,
        const TABLE_ENTRIES_PER_BLOCK: u32,
    > Filter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK> for F
{
    fn hash_bits(&self) -> u32 {
        (bits(self.table_entries()) * TABLE_ENTRIES_PER_BLOCK + bits(self.blocks()))
            * BLOCKS_PER_ELEM
    }

    fn insert_hash(&mut self, hash: u64) {
        let mut hash = hash;

        for _ in 0..BLOCKS_PER_ELEM {
            let (query_block, hash2) = self.query_block_for_hash(hash);
            let (index, hash3) = extract_index_from_hash(hash2, self.blocks());
            unsafe {
                (*self.block_mut(index)).merge(&query_block);
            }
            hash = hash3;
        }
    }

    fn contains_hash(&self, hash: u64) -> bool {
        let mut hash = hash;

        for _ in 0..BLOCKS_PER_ELEM {
            let (block, hash2) = self.query_block_for_hash(hash);
            let (index, hash3) = extract_index_from_hash(hash2, self.blocks());
            if !unsafe { (*self.block(index)).contains(&block) } {
                return false;
            }
            hash = hash3;
        }

        true
    }

    fn insert<A: Hashable>(&mut self, value: A) {
        self.insert_hash(value.hash());
    }

    fn contains<A: Hashable>(&self, value: A) -> bool {
        self.contains_hash(value.hash())
    }

    fn serialize<W: Write>(&self, output: &mut W) -> IoResult<()> {
        output.write_all(self.to_bytes())?;
        Ok(())
    }
}

fn extract_index_from_hash(hash: u64, bound: u32) -> (u32, u64) {
    let bits = bits(bound);
    let unscaled = hash & ((1 << bits) - 1);
    let scaled = (unscaled as f64) * (bound as f64) / ((1 << bits) as f64);
    (scaled as u32, hash >> bits)
}

fn bits(n: u32) -> u32 {
    32 - n.leading_zeros()
}

pub trait Hashable {
    fn hash(&self) -> u64;
}

pub struct RawFilter<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> {
    buffer: *mut Block,
    blocks: u32,
    table_entries: u32,
}

impl<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32>
    RawFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
{
    pub fn new<R: Rng>(
        buffer: *mut Block,
        blocks: u32,
        table_entries: u32,
        ones: u32,
        rng: &mut R,
    ) -> Self {
        Self::checked_new(buffer, blocks, table_entries, ones, rng).unwrap_or_else(|error| {
            panic!("{}", error);
        })
    }

    pub fn checked_new<R: Rng>(
        buffer: *mut Block,
        blocks: u32,
        table_entries: u32,
        ones: u32,
        rng: &mut R,
    ) -> Result<Self, Error> {
        let mut filter = RawFilter {
            buffer,
            blocks,
            table_entries,
        };
        filter.initialize_new(ones, rng)?;
        Ok(filter)
    }

    pub unsafe fn from_ptr(buffer: *mut Block, len: usize) -> Result<Self, Error> {
        if len == 0 {
            return Err(Error::MissingHeader);
        }

        let (blocks, table_entries) = Self::parse_header_block(&*buffer)?;
        let buffer_len = Self::buffer_len_assert_bounds(blocks, table_entries)?;

        if len < buffer_len as usize {
            return Err(Error::Truncated);
        }

        Ok(RawFilter {
            buffer,
            blocks,
            table_entries,
        })
    }
}

impl<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32>
    PrivateFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
    for RawFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
{
    fn buffer(&self) -> *const Block {
        self.buffer
    }

    fn buffer_mut(&mut self) -> *mut Block {
        self.buffer
    }

    fn blocks(&self) -> u32 {
        self.blocks
    }

    fn table_entries(&self) -> u32 {
        self.table_entries
    }
}

pub struct VecFilter<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> {
    buffer: * mut Block,
    blocks: u32,
    table_entries: u32,
}

impl<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32>
    VecFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
{
    pub fn new<R: Rng>(blocks: u32, table_entries: u32, ones: u32, rng: &mut R) -> Self {
        Self::checked_new(blocks, table_entries, ones, rng).unwrap_or_else(|error| {
            panic!("{}", error);
        })
    }

    pub fn checked_new<R: Rng>(
        blocks: u32,
        table_entries: u32,
        ones: u32,
        rng: &mut R,
    ) -> Result<Self, Error> {
        let len = Self::buffer_len_assert_bounds(blocks, table_entries)?;
        let (buffer, _, _) = Vec::<Block>::with_capacity(len as usize).into_raw_parts();
        let mut filter = VecFilter { buffer, blocks, table_entries };
        filter.initialize_new(ones, rng)?;
        Ok(filter)
    }
}

impl<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> ops::Drop
    for VecFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
{
    fn drop(&mut self) {
        let len = self.buffer_len() as usize;
        unsafe {
            Vec::from_raw_parts(self.buffer_mut(), len, len);
        }
    }
}

impl<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32>
    PrivateFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
    for VecFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
{
    fn buffer(&self) -> *const Block {
        self.buffer
    }

    fn buffer_mut(&mut self) -> *mut Block {
        self.buffer
    }

    fn blocks(&self) -> u32 {
        self.blocks
    }

    fn table_entries(&self) -> u32 {
        self.table_entries
    }
}
