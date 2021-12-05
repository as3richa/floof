#![feature(maybe_uninit_extra)]

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{__m256i, _mm256_or_si256, _mm256_setzero_si256, _mm256_testc_si256};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::mem::transmute;

use rand::Rng;

use core::mem::{size_of, MaybeUninit};
use core::ptr::drop_in_place;

#[derive(Clone)]
#[repr(align(64))]
pub struct Block {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    data: [__m256i; 2],

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    data: [u32; 16],
}

impl Block {
    pub fn new() -> Block {
        Block {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            data: [unsafe { _mm256_setzero_si256() }; 2],

            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            data: [0; 16],
        }
    }

    pub fn random<R: Rng>(ones: u8, rng: &mut R) -> Block {
        let mut one_bits: [MaybeUninit<u16>; 256] = unsafe { MaybeUninit::uninit().assume_init() };

        for (bit, ptr) in one_bits[0..(ones as usize)].iter_mut().enumerate() {
            ptr.write(bit as u16);
        }

        let mut i = (ones as usize) + 1;
        let mut w = (rng.gen::<f64>().ln()/(ones as f64)).exp();

        while i < 512 {
            i += (rng.gen::<f64>().ln()/(1.0 - w).ln()).floor() as usize;
            if i < 512 {
                one_bits[rng.gen_range(0..(ones as usize))] = MaybeUninit::new(i as u16);
                w*= (rng.gen::<f64>().ln() / (ones as f64)).exp();
            }
        }

        let mut data: [u32; 16] = [0; 16];

        for ptr in one_bits[0..(ones as usize)].iter_mut() {
            let index = unsafe { ptr.read() };
            data[(index as usize) / 32] |= 1 << ((index as usize) % 32);

            unsafe {
                drop_in_place(ptr.as_mut_ptr());
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            Block {
                data: unsafe { transmute::<_, [__m256i; 2]>(data) },
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            Block { data: data }
        }
    }

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
}

pub trait Filter {
    fn hash_bits(&self) -> u32;
    fn insert_hash(&mut self, hash: u64);
    fn contains_hash(&self, hash: u64) -> bool;
}

impl<T: InternalFilter> Filter for T {
    fn hash_bits(&self) -> u32 {
        (bits(self.table_entries()) * self.table_entries_per_block() + bits(self.blocks()))
            * self.blocks_per_elem()
    }

    fn insert_hash(&mut self, mut hash: u64) {
        for _ in 0..self.blocks_per_elem() {
            let (block, hash2) = compute_block(self, hash);
            let (index, hash3) = extract_from_hash(hash2, self.blocks());
            unsafe {
                self.block_mut(index).merge(&block);
            }
            hash = hash3;
        }
    }

    fn contains_hash(&self, mut hash: u64) -> bool {
        for _ in 0..self.blocks_per_elem() {
            let (block, hash2) = compute_block(self, hash);
            let (index, hash3) = extract_from_hash(hash2, self.blocks());
            if !unsafe { self.block(index).contains(&block) } {
                return false;
            }
            hash = hash3;
        }
        return true;
    }
}

fn compute_block<T: InternalFilter>(filter: &T, mut hash: u64) -> (Block, u64) {
    let (index0, hash0) = extract_from_hash(hash, filter.table_entries());
    let mut block = unsafe { filter.table_entry(index0) }.clone();
    hash = hash0;

    for _ in 1..filter.table_entries_per_block() {
        let (index, hash2) = extract_from_hash(hash, filter.table_entries());
        block.merge(unsafe { filter.table_entry(index) });
        hash = hash2;
    }

    (block, hash)
}

fn extract_from_hash(hash: u64, n: usize) -> (usize, u64) {
    let bits = bits(n);
    let unscaled = hash & ((1 << bits) - 1);
    let scaled = (unscaled as f64) * (n as f64) / ((1 << bits) as f64);
    (scaled as usize, hash >> bits)
}

fn bits(n: usize) -> u32 {
    ((8 * size_of::<usize>()) as u32) - n.leading_zeros()
}

pub trait Hashable {
    fn hash(&self) -> u64;
}

pub trait TypedFilter<A> {
    fn insert(&mut self, value: A);

    fn contains(&self, value: A) -> bool;
}

impl<F: Filter, A: Hashable> TypedFilter<A> for F {
    fn insert(&mut self, value: A) {
        self.insert_hash(value.hash())
    }

    fn contains(&self, value: A) -> bool {
        self.contains_hash(value.hash())
    }
}

pub struct BlockedFilter {
    all_blocks: Vec<Block>,
    table_entries: usize,
}

impl BlockedFilter {
    pub fn new(blocks: usize, table_entries: usize, bits_per_entry: u32) -> BlockedFilter {
        BlockedFilter {
            all_blocks: vec![Block::new(); blocks + table_entries],
            table_entries: 0,
        }
    }
}

impl InternalFilter for BlockedFilter {
    fn blocks(&self) -> usize {
        self.all_blocks.len() - self.table_entries()
    }

    fn blocks_per_elem(&self) -> u32 {
        1
    }

    unsafe fn block(&self, index: usize) -> &Block {
        self.all_blocks.get_unchecked(self.table_entries + index)
    }

    unsafe fn block_mut(&mut self, index: usize) -> &mut Block {
        self.all_blocks
            .get_unchecked_mut(self.table_entries + index)
    }

    fn table_entries(&self) -> usize {
        self.table_entries
    }

    fn table_entries_per_block(&self) -> u32 {
        1
    }

    unsafe fn table_entry(&self, index: usize) -> &Block {
        self.all_blocks.get_unchecked(index)
    }
}
