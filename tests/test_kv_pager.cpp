// Unit tests for the paged KV-cache allocator (no GPU, no model needed).
#include <cstdio>
#include <set>
#include <vector>

#include "../src/kv_pager.h"

static int failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            failures++;                                                \
        }                                                              \
    } while (0)

int main() {
    const int kSeqs = 4, kPages = 8, kPageTokens = 16;
    std::vector<uint32_t> table(kSeqs * kPages, 0xFFFFFFFFu);
    KVPager p(kSeqs, kPages, kPageTokens, table.data());

    // Positions are handed out consecutively; pages are allocated lazily.
    const int a = p.create();
    CHECK(a == 0 && p.length(a) == 0 && p.free_pages() == kPages);
    CHECK(p.append(a, 1) == 0);
    CHECK(p.free_pages() == kPages - 1);
    CHECK(p.append(a, 15) == 1);
    CHECK(p.free_pages() == kPages - 1);  // still inside the first page
    CHECK(p.append(a, 1) == 16);
    CHECK(p.free_pages() == kPages - 2);

    // Two sequences never share a page.
    const int b = p.create();
    CHECK(p.append(b, 40) == 0);          // 3 pages
    std::set<uint32_t> used;
    for (int i = 0; i < p.pages_of(a); i++) used.insert(table[a * kPages + i]);
    for (int i = 0; i < p.pages_of(b); i++) used.insert(table[b * kPages + i]);
    CHECK((int)used.size() == p.pages_of(a) + p.pages_of(b));
    CHECK(p.free_pages() == kPages - 5);

    // Exhaustion is reported without side effects.
    CHECK(p.append(a, 4 * kPageTokens) == -1);
    CHECK(p.length(a) == 17 && p.free_pages() == kPages - 5);

    // Truncation returns whole pages only (speculative-decoding rollback).
    p.truncate(a, 16);
    CHECK(p.length(a) == 16 && p.free_pages() == kPages - 4);
    p.truncate(a, 5);
    CHECK(p.free_pages() == kPages - 4);  // partially used page is kept
    CHECK(p.append(a, 1) == 5);

    // Releasing a sequence recycles its pages and its slot.
    p.release(b);
    CHECK(p.free_pages() == kPages - 1);
    const int c = p.create();
    CHECK(c == b);
    CHECK(p.append(c, 7 * kPageTokens) == 0);
    CHECK(p.free_pages() == 0);

    // Slots are limited too.
    p.create();
    p.create();
    CHECK(p.create() == -1);

    printf("kv_pager: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
