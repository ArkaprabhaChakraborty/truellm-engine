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

} // namespace truellm
