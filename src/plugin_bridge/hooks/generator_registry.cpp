// ---------------------------------------------------------------------------
// generator_registry.cpp — Phase D: priority-ordered generator dispatch
// ---------------------------------------------------------------------------

#include "generator_registry.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <shared_mutex>
#include <vector>

namespace truellm {

// ── Internal storage ──────────────────────────────────────────────────────────
struct GeneratorRegistry::Impl {
    struct Entry {
        truellm_generator_t* gen;
        std::string          plugin_id;
        int32_t              priority;
    };
    mutable std::shared_mutex mutex;
    std::vector<Entry>        entries;
};

GeneratorRegistry::GeneratorRegistry()
    : impl_(std::make_unique<Impl>())
{}

GeneratorRegistry::~GeneratorRegistry() = default;

// ---------------------------------------------------------------------------
void GeneratorRegistry::register_generator(truellm_generator_t* gen)
{
    if (!gen || !gen->plugin_id) return;

    std::unique_lock lk(impl_->mutex);

    Impl::Entry e;
    e.gen       = gen;
    e.plugin_id = gen->plugin_id;
    e.priority  = gen->priority;
    impl_->entries.push_back(std::move(e));

    std::stable_sort(impl_->entries.begin(), impl_->entries.end(),
        [](const Impl::Entry& a, const Impl::Entry& b) {
            return a.priority < b.priority;
        });

    spdlog::info("[GeneratorRegistry] registered generator '{}' priority={}",
                 gen->plugin_id, gen->priority);
}

// ---------------------------------------------------------------------------
void GeneratorRegistry::unregister_plugin(const std::string& plugin_id)
{
    std::unique_lock lk(impl_->mutex);
    auto it = std::remove_if(impl_->entries.begin(), impl_->entries.end(),
        [&](const Impl::Entry& e) { return e.plugin_id == plugin_id; });
    impl_->entries.erase(it, impl_->entries.end());
}

// ---------------------------------------------------------------------------
bool GeneratorRegistry::has_generators() const
{
    std::shared_lock lk(impl_->mutex);
    return !impl_->entries.empty();
}

// ---------------------------------------------------------------------------
bool GeneratorRegistry::claims(const std::string& messages_json,
                                truellm_context_t* ctx)
{
    std::vector<Impl::Entry> snapshot;
    {
        std::shared_lock lk(impl_->mutex);
        snapshot = impl_->entries;
    }

    for (const auto& entry : snapshot) {
        if (!entry.gen->claims) continue;
        if (entry.gen->claims(entry.gen->state, ctx, messages_json.c_str())) {
            spdlog::debug("[GeneratorRegistry] generator '{}' claims request",
                          entry.plugin_id);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// dispatch — claim-and-run in one pass.
//
// Walks entries in priority order and asks each generator whether it claims
// the request.  The first generator that claims the request also produces
// the response — no other generators are consulted.
//
// The output token callback is forwarded verbatim to generate(), so streaming
// generators can emit pieces directly to the HTTP sink.  For the non-streaming
// path the full text is assembled from the tool_result_t payload.
// ---------------------------------------------------------------------------
GeneratorDispatchOutcome GeneratorRegistry::dispatch(
    const std::string&   messages_json,
    truellm_context_t*   ctx,
    void (*on_token)(void*, const char*, int),
    void*                cb_data)
{
    GeneratorDispatchOutcome out;

    std::vector<Impl::Entry> snapshot;
    {
        std::shared_lock lk(impl_->mutex);
        snapshot = impl_->entries;
    }

    for (const auto& entry : snapshot) {
        if (!entry.gen->claims || !entry.gen->generate) continue;
        if (!entry.gen->claims(entry.gen->state, ctx, messages_json.c_str()))
            continue;

        out.claimed   = true;
        out.plugin_id = entry.plugin_id;

        spdlog::info("[GeneratorRegistry] dispatching to generator '{}' "
                     "priority={}", entry.plugin_id, entry.priority);

        truellm_tool_result_t tr = entry.gen->generate(
            entry.gen->state,
            ctx,
            messages_json.c_str(),
            on_token,
            cb_data);

        out.error = tr.error;
        if (tr.error == TRUELLM_OK && tr.payload) {
            out.ok   = true;
            out.text = tr.payload;
        } else {
            out.ok = false;
            if (tr.payload) out.error_message = tr.payload;
            else            out.error_message = "generator returned no payload";
            spdlog::warn("[GeneratorRegistry] generator '{}' failed "
                         "error={} msg='{}'", entry.plugin_id,
                         static_cast<int>(tr.error), out.error_message);
        }

        // Plugin owns the payload buffer; release it now.
        if (tr.free_fn) tr.free_fn(tr.free_arg);
        return out;
    }

    return out;  // .claimed == false
}

} // namespace truellm
