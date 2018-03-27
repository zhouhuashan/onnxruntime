#ifndef CORE_PROVIDERS_CPU_NN_POOL_H
#define CORE_PROVIDERS_CPU_NN_POOL_H

#include "core/providers/cpu/nn/pool_base.h"

namespace Lotus {

class AveragePool;

class MaxPool;

template<typename T, typename PoolType>
class Pool final : public PoolBase {
public:        
    Pool(OpKernelInfo info) : PoolBase(info) {}

    virtual ~Pool() {}

    Status compute(OpKernelContext* context) const override;

};

}
#endif //!CORE_PROVIDERS_CPU_NN_POOL_H 