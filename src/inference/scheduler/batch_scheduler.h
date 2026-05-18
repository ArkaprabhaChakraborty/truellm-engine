#pragma once
// ---------------------------------------------------------------------------
// batch_scheduler.h — Continuous batching scheduler for llama.cpp
//
// Architecture: modelled directly on
//   third_party/llama.cpp/examples/parallel/parallel.cpp
//
// One background thread drives a single llama_decode() call per step that
// contains tokens from BOTH new (prefill) and ongoing (decode) sequences.
// HTTP handler threads submit GenerateRequests and block on std::future.
//
// Slot lifecycle:
//   free → pending_prompt set (admit) → prefill batch → decode loop → complete
//
// Public API is intentionally thin:
//   submit()    — enqueue a request, returns a future
//   is_running  — liveness check
// ---------------------------------------------------------------------------

#include "engine_interface.h"  // GenerateRequest, GenerateResult

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_context;
struct llama_vocab;
struct llama_sampler;
struct llama_batch;

namespace truellm {

class ContinuousBatchScheduler {
public:
    // ctx / vocab are NOT owned — they must outlive this object.
    ContinuousBatchScheduler(llama_context*      ctx,
                             const llama_vocab*  vocab,
                             const SamplingConfig& default_smp,
                             int max_seqs,
                             int max_queue_size);
    ~ContinuousBatchScheduler();

    // Thread-safe — may be called from any HTTP handler thread.
    // Returns immediately; the future blocks the caller until the result is ready.
    std::future<GenerateResult> submit(GenerateRequest req);

    bool is_running() const { return running_.load(std::memory_order_relaxed); }

private:
    // -----------------------------------------------------------------------
    // Slot — one concurrent sequence (mirrors 'struct client' in parallel.cpp)
    // -----------------------------------------------------------------------
    struct Slot {
        int32_t id     = -1;   // slot index (0..max_seqs-1)
        int32_t seq_id = -1;   // llama KV-cache sequence id (-1 = free)

        // --- State ---
        // pending_prompt holds the prompt tokens until they are prefilled.
        // Once the prefill batch is built, pending_prompt is cleared and
        // prefilled is set to true.
        std::vector<int32_t> pending_prompt;
        bool                 prefilled = false;

        // Position in the KV cache (= number of tokens decoded so far).
        // During prefill: advances by len(pending_prompt).
        // During decode:  advances by 1 per step.
        int32_t n_past = 0;

        // The token that was sampled last step — fed back as input next step.
        // int32_t rather than llama_token to avoid pulling llama.h into this header.
        int32_t sampled = 0;

        // Index of this slot's token inside the *current* batch chunk.
        // Set just before llama_decode, read during sampling, then reset to -1.
        int32_t i_batch = -1;

        // Generation metadata
        int32_t n_prompt  = 0;
        int32_t n_decoded = 0;
        int32_t max_new   = 512;

        // Stop criteria
        std::vector<int32_t>     stop_token_ids;
        std::vector<std::string> stop_strings;

        // Output accumulation
        std::string          accumulated;
        std::vector<int32_t> gen_token_ids;

        // Per-request sampler (owned, freed in complete_slot / destructor)
        llama_sampler* sampler = nullptr;

        // Streaming callback (optional, thread-safe via scheduler thread only)
        std::function<void(int32_t, const std::string&)> on_token;

        // Result delivery
        std::promise<GenerateResult> promise;
        int64_t t_start_us = 0;  // microseconds since epoch

        // ── §4.2 — Session KV-prefix cache ─────────────────────────────────
        // When a request supplies session_id, we retain the slot's KV cache
        // after completion so a follow-up request sharing the same
        // session_id can skip prefill on the matched prefix.  Warm slots
        // are kept until either (a) the matching session returns, or (b)
        // a fresh request needs the slot and there are no truly-free
        // alternatives (LRU eviction).
        bool                  warm                = false;
        std::string           cached_session_id;
        std::vector<int32_t>  cached_tokens;        // full prompt + generated
        int64_t               cached_at_us        = 0;

        // True iff the slot is genuinely free (no active request, no warm
        // cache).  Preserved for the prior call-sites that don't care to
        // distinguish warm-but-evictable from outright-free.
        bool is_free()  const { return seq_id == -1; }
        bool is_warm()  const { return seq_id != -1 && warm; }
        bool is_busy()  const { return seq_id != -1 && !warm; }
    };

    // -----------------------------------------------------------------------
    // Queue entry
    // -----------------------------------------------------------------------
    struct PendingRequest {
        GenerateRequest              req;
        std::promise<GenerateResult> promise;
    };

    // ---- non-owned refs ----
    llama_context*      ctx_;
    const llama_vocab*  vocab_;
    const SamplingConfig& default_smp_;
    int                 max_seqs_;
    int                 max_queue_;

    // ---- slot pool ----
    // Each slot owns a fixed seq_id = slot.id (0..max_seqs-1).
    // This keeps all seq_ids within [0, n_seq_max) so llama_decode never rejects them.
    // llama_memory_seq_rm clears the KV entries between requests; the id itself is reused.
    std::vector<Slot>  slots_;

    // ---- reusable batch buffer (size = context length) ----
    llama_batch* batch_owned_ = nullptr;

    // ---- pending-request queue ----
    std::deque<PendingRequest> queue_;
    std::mutex                 queue_mu_;
    std::condition_variable    queue_cv_;

    // ---- background worker ----
    std::thread       worker_;
    std::atomic<bool> running_{false};

    // ---- private methods ----
    void run();
    void try_admit_pending();
    void complete_slot(Slot& slot,
                       ErrorCode ec = ErrorCode::Ok,
                       const std::string& msg = "",
                       const std::string& finish_reason = "stop");
    llama_sampler* make_sampler(const GenerateRequest& req) const;
    bool           matches_stop(const Slot& slot) const;

    // Inline equivalents of common_batch_clear / common_batch_add
    static void batch_clear(llama_batch& b);
    static void batch_add  (llama_batch& b,
                            int32_t      tok,   // llama_token = int32_t
                            int32_t      pos,
                            int32_t      seq_id,
                            bool         need_logits);
};

} // namespace truellm
