#ifndef CORE_FRAMEWORK_EXECUTION_FRAME_H
#define CORE_FRAMEWORK_EXECUTION_FRAME_H

#include <mutex>
#include <vector>
#include "core/framework/ml_value.h"
#include "core/graph/graph.h"
#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"

namespace Lotus
{
    namespace Test
    {
        class TestUtils;
    }

    class OpKernel;

    class ExecutionFrame {
    public:
        typedef MLValue* NodeArgValue;
        typedef std::vector<NodeArgValue> ArgTable;
        // For arena management design, we could have two options:
        // 1. For each device, arena is global in the entire process,
        //    like all the infer session shared the same cpu arena.
        //    The benefit is it gives us protential to share memory 
        //    between different session/request, but will pay the cost
        //    for locking in concurrency.
        // 2. Each executor will host its own arena, after memory planning
        //    we could allocate more efficient with no locking. But the
        //    peak working set memory usage will equal to arenas used by
        //    all concurrent requests. And we might need Arena to have 
        //    different strategy for different graph to address it.
        // No matter which approach we chose, we definitly need to hold
        // Arena in execution frame, the question is should arena owned
        // by execution frame or not. That's why we make this typedef here.
        // Right now the milestone1 implementation goes with option 1.
        // So I make it naked ptr here. Once we finished option 2 and got
        // better result, we can replace this part with something like unique_ptr.
        typedef IArenaAllocator* ArenaPtr;

        ExecutionFrame() {
            InitArenas();
        }

        ~ExecutionFrame() {

        }

        // Index to the first argument of the given node.
        int get_first_arg_index(LotusIR::NODEINDEX node_index) {
            return node_infos_[node_index].start_index;
        }

        template<typename T>
        const T* get_input(int index) const {
            auto value = node_values_[index];
            return reinterpret_cast<T*>(value->pData);
        }

        template<typename T>
        T* get_output(int index) {
            auto value = node_values_[index];
            return reinterpret_cast<T*>(value->pData);
        }

        ArenaPtr GetArena(AllocatorInfo& info)
        {
            for (auto arena : arenas_)
            {
                if (arena->Info() == info)
                    return arena;
            }
            return nullptr;
        }

    private:
        friend class OpKernelContext;

        //The TestUtils need hack this class to provide input/output 
        // tensors since the class is not fully implemented yet.
        friend class Lotus::Test::TestUtils;

        struct NodeInfo {
            // The kernel for this node.
            OpKernel* kernel = nullptr;

            // node_values_[start_index] is the first argument of this node.
            int start_index = 0;
        };

        Tensor* get_or_create_tensor(int tensor_index, const TensorShape& shape) {
            auto value = node_values_[tensor_index];
            if (nullptr != value->pData) {
                // The tensor has already been allocated.
                // TODO: Check the size of the allocated tensor with given shape,
                // if they match each other, then return, else throw error.
                // TODO: type also needs to be checked and then use static_cast.
                return reinterpret_cast<Tensor*>(value->pData);
            }
            else {
                // It's not allocated, then allocate it with given shape and return.
                (shape);
                return nullptr;
            }
        }
        
        void InitArenas()
        {
            // For milestone 1, we only have CPU arena.
            // If later we want executor to host its own arena
            // Need to update this part.
            auto alloc_mgr = AllocatorManager::Instance();
            LOTUS_ENFORCE(alloc_mgr);
            arenas_.push_back(&alloc_mgr->GetArena(CPU));
        }

        std::mutex mu_;
        Status status_;

        // The values for the inputs and outputs of the nodes.
        ArgTable node_values_;

        // All the values for the entire graph.
        vector<MLValue> all_values_;

        // The start index into node_values_ for all the nodes.
        vector<NodeInfo> node_infos_;

        // i-th kernel is still waiting for pending_counts_[i] inputs.
        vector<int> pending_counts_;
        
        // The arenas used for current execution
        // Like mentioned in comments above, we could have two approach:
        // Arena owned by global allocator manager, or arena owned by 
        // Execution frame. Currently we are implment by global arena approach
        // So here is a list of raw pointer and execution frame don't need
        // release them. If we switch to another approach later, we should
        // define ArenaPtr as unique_ptr here.
        vector<ArenaPtr> arenas_;
    };
}

#endif  // CORE_FRAMEWORK_EXECUTION_FRAME_H
