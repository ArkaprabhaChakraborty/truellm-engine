// ---------------------------------------------------------------------------
// batch_scheduler.cpp — Continuous batching scheduler
//
// Algorithm directly from llama.cpp/examples/parallel/parallel.cpp:
//
//  • Each step: ONE llama_decode() call with tokens from ALL sequences.
//    Prefill tokens and single decode tokens share the same batch.
//  • After decode: sample one token per slot using its i_batch index.
//  • KV cache: per-slot seq_id, freed with llama_memory_seq_rm on completion.
//  • Per-slot sampler: built from request params (temp/top_p/top_k overrides).
// ---------------------------------------------------------------------------

#include "batch_scheduler.h"

#include <llama.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>

namespace truellm {

// ---------------------------------------------------------------------------
// batch_clear / batch_add — equivalents of common_batch_clear / common_batch_add
// (keeping LLAMA_BUILD_COMMON disabled avoids pulling in ~30 extra files)
// ---------------------------------------------------------------------------
void ContinuousBatchScheduler::batch_clear(llama_batch& b)
{
    b.n_tokens = 0;
}

void ContinuousBatchScheduler::batch_add(llama_batch& b,
                                          int32_t      tok,
                                          int32_t      pos,
                                          int32_t      seq_id,
                                          bool         need_logits)
{
    b.token    [b.n_tokens]    = tok;
    b.pos      [b.n_tokens]    = pos;
    b.n_seq_id [b.n_tokens]    = 1;
    b.seq_id   [b.n_tokens][0] = seq_id;
    b.logits   [b.n_tokens]    = need_logits ? 1 : 0;
    b.n_tokens++;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
ContinuousBatchScheduler::ContinuousBatchScheduler(
        llama_context*      ctx,
        const llama_vocab*  vocab,
        const SamplingConfig& default_smp,
        int max_seqs,
        int max_queue_size)
    : ctx_(ctx)
    , vocab_(vocab)
    , default_smp_(default_smp)
    , max_seqs_(max_seqs)
    , max_queue_(max_queue_size)
{
    slots_.resize(static_cast<std::size_t>(max_seqs_));
    for (int i = 0; i < max_seqs_; ++i) slots_[i].id = i;

    // Allocate a batch large enough for the full context window.
    // This handles large prompts and the decode tokens all in one pass.
    // llama_batch_init(n_tokens_alloc, embd=0, n_seq_max=1):
    //   embd=0    → use token ids, not embedding vectors
    //   n_seq_max=1 → each token belongs to exactly one sequence
    const int n_ctx = static_cast<int>(llama_n_ctx(ctx_));
    auto* bp = new llama_batch(llama_batch_init(n_ctx, 0, 1));
    batch_owned_ = bp;

    running_.store(true);
    worker_ = std::thread([this]{ run(); });

    spdlog::info("[Scheduler] Started  max_seqs={} max_queue={} ctx_size={}",
                 max_seqs_, max_queue_, n_ctx);
}

ContinuousBatchScheduler::~ContinuousBatchScheduler()
{
    running_.store(false, std::memory_order_relaxed);
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();

    for (auto& slot : slots_) {
        if (slot.sampler) { llama_sampler_free(slot.sampler); slot.sampler = nullptr; }
    }
    if (batch_owned_) {
        llama_batch_free(*batch_owned_);
        delete batch_owned_;
        batch_owned_ = nullptr;
    }
    spdlog::debug("[Scheduler] Stopped");
}

// ---------------------------------------------------------------------------
// submit — called from HTTP handler threads (thread-safe)
// ---------------------------------------------------------------------------
std::future<GenerateResult> ContinuousBatchScheduler::submit(GenerateRequest req)
{
    std::promise<GenerateResult> prom;
    auto fut = prom.get_future();
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        if (static_cast<int>(queue_.size()) >= max_queue_) {
            GenerateResult err;
            err.error         = ErrorCode::Unavailable;
            err.error_message = "Request queue is full";
            prom.set_value(std::move(err));
            return fut;
        }
        queue_.push_back({ std::move(req), std::move(prom) });
    }
    queue_cv_.notify_one();
    return fut;
}

// ---------------------------------------------------------------------------
// make_sampler — per-request sampler chain
// Mirrors GgmlContext::build_sampler() but with per-request param overrides.
// ---------------------------------------------------------------------------
llama_sampler* ContinuousBatchScheduler::make_sampler(const GenerateRequest& req) const
{
    llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(cp);

    const int   top_k = (req.top_k   > 0)    ? req.top_k    : default_smp_.top_k;
    const float top_p = (req.top_p   > 0.f)   ? req.top_p    : default_smp_.top_p;
    const float min_p = default_smp_.min_p;
    const float temp  = (req.temperature >= 0.f) ? req.temperature : default_smp_.temperature;
    const float rep   = (req.repeat_penalty > 0.f) ? req.repeat_penalty : default_smp_.repeat_penalty;

    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain,
        llama_sampler_init_penalties(64, rep,
                                     default_smp_.frequency_penalty,
                                     default_smp_.presence_penalty));

    const uint32_t seed = (default_smp_.seed < 0)
        ? static_cast<uint32_t>(
            std::chrono::steady_clock::now().time_since_epoch().count() & 0xFFFFFFFFull)
        : static_cast<uint32_t>(default_smp_.seed);
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    return chain;
}

// ---------------------------------------------------------------------------
// matches_stop — check if accumulated text ends with any stop string
// ---------------------------------------------------------------------------
bool ContinuousBatchScheduler::matches_stop(const Slot& slot) const
{
    for (const auto& s : slot.stop_strings) {
        if (!s.empty() && slot.accumulated.size() >= s.size()) {
            if (slot.accumulated.compare(
                    slot.accumulated.size() - s.size(), s.size(), s) == 0)
                return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// complete_slot — fulfil promise and release KV cache slot
// ---------------------------------------------------------------------------
void ContinuousBatchScheduler::complete_slot(Slot& slot, ErrorCode ec, const std::string& msg)
{
    const int64_t t_end_us = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    GenerateResult res;
    res.text             = std::move(slot.accumulated);
    res.tokens           = std::move(slot.gen_token_ids);
    res.prompt_tokens    = slot.n_prompt;
    res.generated_tokens = slot.n_decoded;
    res.time_ms          = static_cast<float>(t_end_us - slot.t_start_us) / 1000.f;
    res.error            = ec;
    res.error_message    = msg;

    slot.promise.set_value(std::move(res));

    // Free KV cache for this sequence
    llama_memory_seq_rm(llama_get_memory(ctx_), slot.seq_id, -1, -1);

    if (slot.sampler) { llama_sampler_free(slot.sampler); slot.sampler = nullptr; }

    spdlog::debug("[Scheduler] Slot {} seq_id={} done  prompt={} gen={} err={}",
                  slot.id, slot.seq_id, slot.n_prompt, slot.n_decoded,
                  static_cast<int>(ec));

    // Reset slot to free
    slot.seq_id = -1;
    slot.prefilled = false;
    slot.pending_prompt.clear();
    slot.accumulated.clear();
    slot.gen_token_ids.clear();
    slot.stop_strings.clear();
    slot.stop_token_ids.clear();
    slot.on_token = nullptr;
    slot.n_past    = 0;
    slot.n_prompt  = 0;
    slot.n_decoded = 0;
    slot.i_batch   = -1;
}

// ---------------------------------------------------------------------------
// try_admit_pending — pull waiting requests into free slots (called from worker)
// ---------------------------------------------------------------------------
void ContinuousBatchScheduler::try_admit_pending()
{
    std::lock_guard<std::mutex> lk(queue_mu_);
    for (auto& slot : slots_) {
        if (queue_.empty()) break;
        if (!slot.is_free()) continue;

        auto& entry = queue_.front();

        if (entry.req.tokens.empty()) {
            GenerateResult err;
            err.error         = ErrorCode::InvalidArgument;
            err.error_message = "Empty token list";
            entry.promise.set_value(std::move(err));
            queue_.pop_front();
            continue;
        }

        // Reject prompts that exceed the KV-cache context window.  Admitting
        // an oversized prompt causes llama_decode() to silently truncate or
        // crash.  Leave one token of headroom for at least one generated token.
        const int n_ctx = static_cast<int>(llama_n_ctx(ctx_));
        const int n_prompt = static_cast<int>(entry.req.tokens.size());
        if (n_prompt >= n_ctx) {
            spdlog::warn("[Scheduler] Rejecting request: prompt_toks={} >= ctx_size={} "
                         "(no room for generation)", n_prompt, n_ctx);
            GenerateResult err;
            err.error         = ErrorCode::InvalidArgument;
            err.error_message = "Prompt length (" + std::to_string(n_prompt) +
                                " tokens) exceeds context window (" +
                                std::to_string(n_ctx) + " tokens)";
            err.prompt_tokens = static_cast<int32_t>(n_prompt);
            entry.promise.set_value(std::move(err));
            queue_.pop_front();
            continue;
        }

        // Assign slot — seq_id is fixed as slot.id so it stays within [0, n_seq_max)
        slot.seq_id         = slot.id;
        slot.prefilled      = false;
        slot.n_past         = 0;
        slot.n_prompt       = static_cast<int32_t>(entry.req.tokens.size());
        slot.n_decoded      = 0;
        slot.i_batch        = -1;
        slot.max_new        = (entry.req.max_tokens > 0) ? entry.req.max_tokens
                                                         : default_smp_.max_tokens;
        slot.stop_token_ids = entry.req.stop_tokens;
        slot.stop_strings   = default_smp_.stop;
        slot.accumulated.clear();
        slot.gen_token_ids.clear();
        slot.on_token       = entry.req.on_token;
        slot.promise        = std::move(entry.promise);
        slot.t_start_us     = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

        // Store prompt tokens in pending_prompt — consumed during prefill step
        slot.pending_prompt.assign(entry.req.tokens.begin(), entry.req.tokens.end());

        if (slot.sampler) { llama_sampler_free(slot.sampler); }
        slot.sampler = make_sampler(entry.req);

        spdlog::debug("[Scheduler] Admit slot={} seq_id={} prompt_toks={}",
                      slot.id, slot.seq_id, slot.n_prompt);
        queue_.pop_front();
    }
}

// ---------------------------------------------------------------------------
// run — background thread
//
// Per-step logic (mirrors parallel.cpp main loop):
//   1. Admit new requests into free slots.
//   2. Build batch:
//      - Slots with pending_prompt  → add all prompt tokens (prefill).
//      - Slots that are prefilled   → add single sampled token (decode).
//   3. llama_decode() (chunked by n_batch for large batches).
//   4. For each slot: sample using i_batch, update state, check stop.
// ---------------------------------------------------------------------------
void ContinuousBatchScheduler::run()
{
    spdlog::debug("[Scheduler] Worker thread started");

    llama_batch& batch = *batch_owned_;
    const int32_t n_batch_max = static_cast<int32_t>(llama_n_batch(ctx_));
    auto* mem = llama_get_memory(ctx_);

    while (running_.load(std::memory_order_relaxed)) {

        // ---- 1. Admit new requests ----
        try_admit_pending();

        // ---- Count active slots ----
        int n_active = 0;
        for (const auto& s : slots_) if (!s.is_free()) ++n_active;

        if (n_active == 0) {
            // Sleep until new work arrives or shutdown is signalled
            std::unique_lock<std::mutex> lk(queue_mu_);
            queue_cv_.wait_for(lk, std::chrono::milliseconds(5),
                [this]{ return !queue_.empty() || !running_; });
            continue;
        }

        // ---- 2. Build batch ----
        batch_clear(batch);

        for (auto& slot : slots_) {
            if (slot.is_free()) continue;

            if (!slot.prefilled) {
                // --- Prefill: add all prompt tokens ---
                const int32_t np = static_cast<int32_t>(slot.pending_prompt.size());
                for (int32_t i = 0; i < np; ++i) {
                    batch_add(batch, slot.pending_prompt[i],
                              slot.n_past + i,
                              slot.seq_id,
                              /*need_logits=*/(i == np - 1));
                }
                slot.i_batch = batch.n_tokens - 1; // index of the last (logit) token
                slot.n_past += np;                  // advance: next token goes here
                // pending_prompt cleared after successful decode+sample
            } else {
                // --- Decode: add single sampled token ---
                slot.i_batch = batch.n_tokens;
                batch_add(batch, slot.sampled, slot.n_past, slot.seq_id, /*need_logits=*/true);
                slot.n_past++;                      // advance for next step
            }
        }

        if (batch.n_tokens == 0) continue;

        // ---- 3. Decode (chunked, mirrors parallel.cpp's chunk loop) ----
        int32_t n_batch = n_batch_max;
        for (int32_t chunk_start = 0; chunk_start < batch.n_tokens; ) {

            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - chunk_start);

            // Slice view into the pre-allocated batch arrays (zero-copy)
            llama_batch chunk_view = {
                n_tokens,
                batch.token    + chunk_start,
                nullptr,
                batch.pos      + chunk_start,
                batch.n_seq_id + chunk_start,
                batch.seq_id   + chunk_start,
                batch.logits   + chunk_start,
            };

            const int ret = llama_decode(ctx_, chunk_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // Unrecoverable failure.
                    // Fail every slot whose i_batch is >= chunk_start: this
                    // covers both slots in the failing chunk AND slots in
                    // subsequent unprocessed chunks (which will never run
                    // because we goto next_step).  Leaving later-chunk slots
                    // alive would corrupt their n_past — it was already
                    // incremented during the BUILD step but their
                    // pending_prompt was never consumed, so the next BUILD
                    // would re-add the same prompt at the wrong position.
                    spdlog::error("[Scheduler] llama_decode failed ret={}", ret);
                    for (auto& slot : slots_) {
                        if (!slot.is_free() && slot.i_batch >= chunk_start) {
                            complete_slot(slot, ErrorCode::InternalError,
                                          "llama_decode failed");
                        }
                    }
                    goto next_step; // break both loops
                }
                // Retry with half the batch size (KV pressure)
                spdlog::warn("[Scheduler] llama_decode ret={}, retrying with n_batch={}",
                             ret, n_batch / 2);
                n_batch /= 2;
                continue;
            }

            // Restore batch size on success
            n_batch = n_batch_max;

            // ---- 4. Sample one token per slot in this chunk ----
            for (auto& slot : slots_) {
                if (slot.is_free()) continue;
                if (slot.i_batch < chunk_start ||
                    slot.i_batch >= chunk_start + n_tokens) continue;

                const int32_t idx_in_chunk = slot.i_batch - chunk_start;

                const llama_token id = llama_sampler_sample(slot.sampler, ctx_, idx_in_chunk);
                llama_sampler_accept(slot.sampler, id);

                // EOG check FIRST — must not count the end-of-generation token
                // in n_decoded or include it in gen_token_ids / accumulated.
                // (Mirrors the ggml_engine.cpp decode loop where EOG breaks
                //  before any state is updated.)
                const bool eog = llama_vocab_is_eog(vocab_, id);

                // Decode token to text
                char piece_buf[256];
                const int piece_len = llama_token_to_piece(vocab_, id, piece_buf,
                                                            sizeof(piece_buf), 0, false);
                std::string piece;
                if (piece_len > 0)
                    piece.assign(piece_buf, static_cast<std::size_t>(piece_len));

                // For a just-prefilled slot: clear pending_prompt, mark prefilled
                if (!slot.prefilled) {
                    slot.pending_prompt.clear();
                    slot.prefilled = true;
                }

                slot.sampled  = id;
                slot.i_batch  = -1;

                if (eog) {
                    spdlog::trace("[Scheduler] Slot {} EOG", slot.id);
                    complete_slot(slot);
                    continue;  // next slot — do not accumulate or stream EOG
                }

                // Update slot state (non-EOG tokens only)
                slot.n_decoded++;
                slot.accumulated += piece;
                slot.gen_token_ids.push_back(static_cast<int32_t>(id));

                // Streaming callback
                if (slot.on_token) slot.on_token(static_cast<int32_t>(id), piece);

                // Check non-EOG termination conditions
                const bool stop_str   = matches_stop(slot);
                const bool max_hit    = (slot.n_decoded >= slot.max_new);
                bool       stop_by_id = false;
                for (int32_t st : slot.stop_token_ids) {
                    if (static_cast<llama_token>(st) == id) { stop_by_id = true; break; }
                }

                if (stop_str || max_hit || stop_by_id) {
                    spdlog::trace("[Scheduler] Slot {} completing  "
                                  "stop_str={} max={} stop_id={}",
                                  slot.id, stop_str, max_hit, stop_by_id);
                    complete_slot(slot);
                }
            }

            chunk_start += n_tokens;
        }
        next_step:;
    }

    // Drain remaining active slots and queue on shutdown
    for (auto& slot : slots_) {
        if (!slot.is_free())
            complete_slot(slot, ErrorCode::Unavailable, "Scheduler shutting down");
    }
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        while (!queue_.empty()) {
            GenerateResult err;
            err.error         = ErrorCode::Unavailable;
            err.error_message = "Scheduler shutting down";
            queue_.front().promise.set_value(std::move(err));
            queue_.pop_front();
        }
    }

    spdlog::debug("[Scheduler] Worker thread exited");
}

} // namespace truellm
