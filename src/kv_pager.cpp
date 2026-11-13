#include "kv_pager.h"

KVPager::KVPager(int max_seqs, int n_pages, int page_tokens, uint32_t* table)
    : max_seqs_(max_seqs), n_pages_(n_pages), page_tokens_(page_tokens), pages_per_seq_(n_pages), table_(table),
      len_(max_seqs, 0), live_(max_seqs, 0) {
    // Hand out low page numbers first.
    for (int p = n_pages - 1; p >= 0; p--) free_.push_back((uint32_t)p);
}

int KVPager::create() {
    for (int s = 0; s < max_seqs_; s++)
        if (!live_[s]) {
            live_[s] = 1;
            len_[s] = 0;
            return s;
        }
    return -1;
}

void KVPager::release(int seq) {
    truncate(seq, 0);
    live_[seq] = 0;
}

int KVPager::append(int seq, int n) {
    const int have = pages_of(seq);
    const int need = (len_[seq] + n + page_tokens_ - 1) / page_tokens_;
    if (need - have > (int)free_.size() || need > pages_per_seq_) return -1;
    for (int p = have; p < need; p++) {
        table_[(size_t)seq * pages_per_seq_ + p] = free_.back();
        free_.pop_back();
    }
    const int pos = len_[seq];
    len_[seq] += n;
    return pos;
}

void KVPager::truncate(int seq, int new_len) {
    if (new_len >= len_[seq]) return;
    const int have = pages_of(seq);
    len_[seq] = new_len;
    for (int p = pages_of(seq); p < have; p++) free_.push_back(table_[(size_t)seq * pages_per_seq_ + p]);
}
