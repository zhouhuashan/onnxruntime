#ifndef CORE_FRAMEWORK_INFERENCE_ENGINE_H
#define CORE_FRAMEWORK_INFERENCE_ENGINE_H

#include <vector>
#include "core\framework\execution_provider.h"
#include "core\framework\ml_value.h"
#include "core\graph\status.h"


using namespace Lotus::Common;

namespace Lotus
{

    // TODO: Executor design.
    class Executor
    {

    };

    // TODO: Cache manager design.
    class CacheManager
    {
        // TODO: hold <node*, executionstate> cache.
    };

    // Per model, handling multiple requests.
    class InferenceEngine {
    public:
        // Load an ONNX model and initialize.
        Status Load(const std::string& p_model_name);

        // Both feeds and fetches are owned by client code, and can't be changed
        // by client code during Run().
        Status Run(const std::vector<MLValue>& p_feeds, /*out*/ std::vector<MLValue>* p_fetches);

        // TODO: how to set provider preferences?
        // The list of execution providers in preference order.
        // Status SetProviderPreference(const std::vector<IExecutionProvider>& p_providers);

    private:

        // The model served by this inference engine instance.
        std::shared_ptr<Model> m_model;

        std::vector<IExecutionProvider> m_executionProviders;

        // Question: Only one Executor needed?
        std::vector<Executor> m_executors;

        CacheManager m_cacheManager;
    };
}

#endif  // CORE_FRAMEWORK_INFERENCE_ENGINE_H