// ---------------------------------------------------------------------------
// preprocessor_chain.cpp — Phase D: ordered preprocessor dispatch
// ---------------------------------------------------------------------------

#include "preprocessor_chain.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <shared_mutex>
#include <vector>

namespace truellm {

// ── Internal storage ──────────────────────────────────────────────────────────
struct PreprocessorChain::Impl {
    struct Entry {
        truellm_preprocessor_t* prep;
        std::string             plugin_id;
        int32_t                 priority;
    };
    mutable std::shared_mutex   mutex;
    std::vector<Entry>          entries;
};

PreprocessorChain::PreprocessorChain()
    : impl_(std::make_unique<Impl>())
{}

PreprocessorChain::~PreprocessorChain() = default;

// ---------------------------------------------------------------------------
void PreprocessorChain::register_preprocessor(truellm_preprocessor_t* prep)
{
    if (!prep || !prep->plugin_id) return;

    std::unique_lock lk(impl_->mutex);

    Impl::Entry e;
    e.prep      = prep;
    e.plugin_id = prep->plugin_id;
    e.priority  = prep->priority;
    impl_->entries.push_back(std::move(e));

    // Keep sorted by priority (ascending — lower = earlier in chain).
    std::stable_sort(impl_->entries.begin(), impl_->entries.end(),
        [](const Impl::Entry& a, const Impl::Entry& b) {
            return a.priority < b.priority;
        });

    spdlog::info("[PreprocessorChain] registered preprocessor '{}' priority={}",
                 prep->plugin_id, prep->priority);
}

// ---------------------------------------------------------------------------
void PreprocessorChain::unregister_plugin(const std::string& plugin_id)
{
    std::unique_lock lk(impl_->mutex);
    auto it = std::remove_if(impl_->entries.begin(), impl_->entries.end(),
        [&](const Impl::Entry& e) { return e.plugin_id == plugin_id; });
    impl_->entries.erase(it, impl_->entries.end());
    spdlog::info("[PreprocessorChain] unregistered preprocessors for '{}'", plugin_id);
}

// ---------------------------------------------------------------------------
bool PreprocessorChain::has_preprocessors() const
{
    std::shared_lock lk(impl_->mutex);
    return !impl_->entries.empty();
}

// ---------------------------------------------------------------------------
PreprocessResult PreprocessorChain::run(const std::string& messages_json,
                                         truellm_context_t* ctx)
{
    // Take a snapshot of the current entry list to avoid holding the lock
    // during potentially-slow plugin calls.
    std::vector<Impl::Entry> snapshot;
    {
        std::shared_lock lk(impl_->mutex);
        snapshot = impl_->entries;
    }

    std::string current_messages = messages_json;

    for (const auto& entry : snapshot) {
        if (!entry.prep->preprocess) continue;

        truellm_preproc_result_t raw =
            entry.prep->preprocess(entry.prep->state, ctx, current_messages.c_str());

        const truellm_preproc_decision_t decision = raw.decision;

        // Capture strings before freeing plugin memory.
        std::string mod_msgs = raw.modified_messages_json
            ? raw.modified_messages_json : "";
        std::string resp_text = raw.response_text
            ? raw.response_text : "";

        if (raw.free_fn) raw.free_fn(raw.free_arg);

        if (!mod_msgs.empty()) {
            current_messages = std::move(mod_msgs);
        }

        if (decision == TRUELLM_PREPROC_RESPOND) {
            spdlog::debug("[PreprocessorChain] '{}' responded directly",
                          entry.plugin_id);
            return {TRUELLM_PREPROC_RESPOND, current_messages, std::move(resp_text)};
        }
        if (decision == TRUELLM_PREPROC_ABORT) {
            spdlog::debug("[PreprocessorChain] '{}' aborted request",
                          entry.plugin_id);
            return {TRUELLM_PREPROC_ABORT, {}, {}};
        }
        // TRUELLM_PREPROC_PASS — continue to next entry.
    }

    // All preprocessors passed.
    PreprocessResult out;
    out.decision               = TRUELLM_PREPROC_PASS;
    out.modified_messages_json = (current_messages != messages_json)
                                 ? std::move(current_messages) : "";
    return out;
}

} // namespace truellm
