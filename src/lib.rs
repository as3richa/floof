#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{__m256i, _mm256_or_si256, _mm256_testc_si256};

use core::mem::size_of;

#[derive(Clone)]
#[repr(align(64))]
pub struct Block {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    data: [__m256i; 2],

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    data: [u32; 16],
}

impl Block {
    fn merge(&mut self, block: &Block) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        for i in 0..2 {
            self.data[i] = unsafe { _mm256_or_si256(self.data[i], block.data[i]) };
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        for i in 0..16 {
            self.data[i] |= block.data[i];
        }
    }

    fn contains(&self, block: &Block) -> bool {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            for i in 0..2 {
                if unsafe { _mm256_testc_si256(self.data[i], block.data[i]) } == 0 {
                    return false;
                }
            }

            return true;
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for i in 0..16 {
                if ((!self.data[i]) & block.data[i]) != 0 {
                    return false;
                }
            }

            return true;
        }
    }
}

pub trait InternalFilter {
    fn blocks(&self) -> usize;
    fn blocks_per_elem(&self) -> u32;
    unsafe fn block(&self, i: usize) -> &Block;
    unsafe fn block_mut(&mut self, index: usize) -> &mut Block;

    fn table_entries(&self) -> usize;
    fn table_entries_per_block(&self) -> u32;
    unsafe fn table_entry(&self, index: usize) -> &Block;

    fn blocks_bits(&self) -> u32 {
        bits(self.blocks())
    }

    fn table_entries_bits(&self) -> u32 {
        bits(self.table_entries())
    }

    fn hash_bits(&self) -> u32 {
        (self.table_entries_bits() * self.table_entries_per_block() + self.blocks_bits())
            * self.blocks_per_elem()
    }
}

pub trait Filter {
    fn insert64(&mut self, hash: u64);
    fn insert32(&mut self, hash: u32);

    fn contains64(&self, hash: u64) -> bool;
    fn contains32(&self, hash: u32) -> bool;
}

macro_rules! compute_block {
    ($self: ident, $hash: expr, $extract: ident) => {{
        let (bits0, hash0) = $extract($hash, $self.table_entries());
        let mut block = unsafe { $self.table_entry(bits0) }.clone();
        let mut hash = hash0;

        for _ in 1..$self.table_entries_per_block() {
            let (bits, hash2) = $extract(hash, $self.table_entries());
            block.merge(unsafe { $self.table_entry(bits) });
            hash = hash2;
        }

        (block, hash)
    }};
}

macro_rules! insert {
    ($self: ident, $hash: expr, $extract: ident) => {{
        let mut hash = $hash;
        for _ in 0..$self.blocks_per_elem() {
            let (block, hash2) = compute_block!($self, hash, $extract);
            let (index, hash3) = $extract(hash2, $self.blocks());
            unsafe {
                $self.block_mut(index).merge(&block);
            }
            hash = hash3;
        }
    }};
}

macro_rules! contains {
    ($self: ident, $hash: expr, $extract: ident) => {{
        let mut hash = $hash;
        for _ in 0..$self.blocks_per_elem() {
            let (block, hash2) = compute_block!($self, hash, $extract);
            let (index, hash3) = $extract(hash2, $self.blocks());
            if !unsafe { $self.block(index).contains(&block) } {
                return false;
            }
            hash = hash3;
        }
        return true;
    }};
}

impl<T: InternalFilter> Filter for T {
    fn insert64(&mut self, hash: u64) {
        debug_assert!(self.hash_bits() <= 64);
        insert!(self, hash, extract64);
    }

    fn insert32(&mut self, hash: u32) {
        debug_assert!(self.hash_bits() <= 32);
        insert!(self, hash, extract32);
    }

    fn contains64(&self, hash: u64) -> bool {
        debug_assert!(self.hash_bits() <= 64);
        contains!(self, hash, extract64);
    }

    fn contains32(&self, hash: u32) -> bool {
        debug_assert!(self.hash_bits() <= 32);
        contains!(self, hash, extract32);
    }
}

fn extract64(hash: u64, n: usize) -> (usize, u64) {
    let bits = bits(n);
    let unscaled = hash & ((1 << bits) - 1);
    let scaled = (unscaled as f64) * (n as f64) / ((1 << bits) as f64);
    (scaled as usize, hash >> bits)
}

fn extract32(hash: u32, n: usize) -> (usize, u32) {
    let bits = bits(n);
    let unscaled = hash & ((1 << bits) - 1);
    let scaled = (unscaled as f64) * (n as f64) / ((1 << bits) as f64);
    (scaled as usize, hash >> bits)
}

fn bits(n: usize) -> u32 {
    ((8 * size_of::<usize>()) as u32) - n.leading_zeros()
}

pub trait Hashable64<A> {
    fn hash(value: A) -> u64;
}

pub trait Hashable32<A> {
    fn hash(value: A) -> u32;
}

pub trait TypedFilter<A> {
    fn insert(value: &A);
    fn contains(value: &A);
}
