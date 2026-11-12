// Paged KV-cache bookkeeping (the CPU half; the GPU reads the page table).
//
// The cache is a pool of fixed-size pages, each holding `page_tokens` token
// slots for every layer. A sequence owns an ordered list of pages; logical
// position p lives in page table[p / page_tokens], slot p % page_tokens.
// Sequences grow one page at a time and return pages to the free list when
// they finish or are truncated, so many sequences of unknown length share
// one pool without fragmentation or copying (the idea behind vLLM's
// PagedAttention).
#pragma once
#include <cstdint>
#include <vector>

class KVPager {
public:
    // `table` points at max_seqs * pages_per_seq uint32 entries (GPU-visible memory).
    KVPager(int max_seqs, int n_pages, int page_tokens, uint32_t* table);

    int create();                      // returns a sequence slot, or -1 if none free
    void release(int seq);             // free the slot and all its pages
    int length(int seq) const { return len_[seq]; }
    // Reserve room for n more tokens; returns the position of the first, or -1
    // if the pool is exhausted (nothing is changed in that case).
    int append(int seq, int n);
    void truncate(int seq, int new_len);  // drop tokens (and whole pages) from the tail

    int free_pages() const { return (int)free_.size(); }
    int n_pages() const { return n_pages_; }
    int page_tokens() const { return page_tokens_; }
    int pages_per_seq() const { return pages_per_seq_; }
    int max_seqs() const { return max_seqs_; }
    int pages_of(int seq) const { return (len_[seq] + page_tokens_ - 1) / page_tokens_; }

private:
    int max_seqs_, n_pages_, page_tokens_, pages_per_seq_;
    uint32_t* table_;
    std::vector<int> len_;
    std::vector<uint8_t> live_;
    std::vector<uint32_t> free_;
};
