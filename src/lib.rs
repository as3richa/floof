#![allow(clippy::missing_safety_doc)]
#![feature(min_const_generics)]
#![feature(maybe_uninit_extra)]

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    __m256i, _mm256_load_si256, _mm256_or_si256, _mm256_store_si256, _mm256_testc_si256,
};

use rand::Rng;

use core::mem::MaybeUninit;
use std::io::{Result as IoResult, Write};

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
        assert!(ones <= Block::BITS as u32);

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

        let mut data: [u32; Block::WORDS] = [0; Block::WORDS];

        for ptr in one_bits[0..(ones as usize)].iter_mut() {
            let index = unsafe { ptr.read() };
            data[(index as usize) / 32] |= 1 << ((index as usize) % 32);
        }

        Block { data }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn load_mm256i(&self, index: usize) -> __m256i {
        debug_assert!(index < 2);
        _mm256_load_si256(&self.data[index * 8] as *const u32 as *const __m256i)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe fn store_mm256i(&mut self, index: usize, value: __m256i) {
        debug_assert!(index < 2);
        _mm256_store_si256(&mut self.data[index * 8] as *mut u32 as *mut __m256i, value);
    }

    fn merge(&mut self, block: &Block) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        for i in 0..2 {
            unsafe {
                let merged = _mm256_or_si256(self.load_mm256i(i), block.load_mm256i(i));
                self.store_mm256i(i, merged);
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        for i in 0..Block::WORDS {
            self.data[i] |= block.data[i];
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
    use crate::{extract_index_from_hash, Block};

    use core::slice;

    pub trait Filter<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> {
        fn blocks(&self) -> u32;
        fn table_entries(&self) -> u32;

        fn buffer(&self) -> *const Block;
        fn buffer_mut(&mut self) -> *mut Block;

        fn buffer_len(&self) -> u32 {
            1 + self.blocks() + self.table_entries()
        }

        unsafe fn block(&self, index: u32) -> &Block {
            debug_assert!(index < self.blocks());
            &*self
                .buffer()
                .add((1 + self.table_entries() + index) as usize)
        }

        unsafe fn block_mut(&mut self, index: u32) -> &mut Block {
            debug_assert!(index < self.blocks());
            &mut *self
                .buffer_mut()
                .add((1 + self.table_entries() + index) as usize)
        }

        unsafe fn table_entry(&self, index: u32) -> &Block {
            debug_assert!(index < self.table_entries());
            &*self.buffer().add(1 + index as usize)
        }

        unsafe fn init_header_block(&mut self) {
            *self.buffer_mut() = Block {
                data: [
                    0xf100f001u32.to_le(),
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
            };
        }

        fn query_block_for_hash(&self, hash: u64) -> (Block, u64) {
            let (index0, hash) = extract_index_from_hash(hash, self.table_entries());
            let mut block = unsafe { self.table_entry(index0) }.clone();
            let mut hash = hash;

            for _ in 1..TABLE_ENTRIES_PER_BLOCK {
                let (index, hash2) = extract_index_from_hash(hash, self.table_entries());
                block.merge(unsafe { self.table_entry(index) });
                hash = hash2;
            }

            (block, hash)
        }

        fn to_bytes(&self) -> &[u8] {
            let bytes = Block::BYTES * (1 + self.table_entries() + self.blocks()) as usize;
            unsafe { slice::from_raw_parts(self.buffer() as *const u8, bytes) }
        }
    }
}

trait Filter<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> {
    fn hash_bits(&self) -> u32;

    fn insert_hash(&mut self, hash: u64);
    fn contains_hash(&self, hash: u64) -> bool;

    fn insert<A: Hashable>(&mut self, value: A);
    fn contains<A: Hashable>(&self, value: A) -> bool;

    fn serialize<W: Write>(&self, output: &mut W) -> IoResult<()>;
}

impl<
        F: private_filter::Filter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>,
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
            let (block, hash2) = self.query_block_for_hash(hash);
            let (index, hash3) = extract_index_from_hash(hash2, self.blocks());
            unsafe {
                self.block_mut(index).merge(&block);
            }
            hash = hash3;
        }
    }

    fn contains_hash(&self, hash: u64) -> bool {
        let mut hash = hash;

        for _ in 0..BLOCKS_PER_ELEM {
            let (block, hash2) = self.query_block_for_hash(hash);
            let (index, hash3) = extract_index_from_hash(hash2, self.blocks());
            if !unsafe { self.block(index).contains(&block) } {
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

fn extract_index_from_hash(hash: u64, n: u32) -> (u32, u64) {
    let bits = bits(n);
    let unscaled = hash & ((1 << bits) - 1);
    let scaled = (unscaled as f64) * (n as f64) / ((1 << bits) as f64);
    (scaled as u32, hash >> bits)
}

fn bits(n: u32) -> u32 {
    32 - n.leading_zeros()
}

pub trait Hashable {
    fn hash(&self) -> u64;
}

pub struct VecFilter<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32> {
    buffer: Vec<Block>,
    table_entries: u32,
}

impl<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32>
    VecFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
{
    pub fn new<R: Rng>(
        blocks: u32,
        table_entries: u32,
        ones: u32,
        rng: &mut R,
    ) -> VecFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK> {
        let mut buffer = Vec::<Block>::with_capacity((  blocks + table_entries) as usize);

        for i in 0..table_entries {
            unsafe {
                std::ptr::write(
                    buffer.as_mut_ptr().add(i as usize),
                    Block::random(ones, rng),
                );
            }
        }

        for i in 0..blocks {
            unsafe {
                std::ptr::write(
                    buffer.as_mut_ptr().add((table_entries + i) as usize),
                    Block::default(),
                );
            }
        }

        unsafe {
            buffer.set_len((blocks + table_entries) as usize);
        }

        VecFilter {
            buffer,
            table_entries,
        }
    }
}

impl<const BLOCKS_PER_ELEM: u32, const TABLE_ENTRIES_PER_BLOCK: u32>
    private_filter::Filter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
    for VecFilter<BLOCKS_PER_ELEM, TABLE_ENTRIES_PER_BLOCK>
{
    fn buffer(&self) -> *const Block {
        self.buffer.as_ptr()
    }

    fn buffer_mut(&mut self) -> *mut Block {
        self.buffer.as_mut_ptr()
    }

    fn blocks(&self) -> u32 {
        (self.buffer.len() - (self.table_entries() as usize)) as u32
    }

    fn table_entries(&self) -> u32 {
        self.table_entries
    }
}
