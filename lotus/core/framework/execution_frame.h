#ifndef CORE_FRAMEWORK_EXECUTION_FRAME_H
#define CORE_FRAMEWORK_EXECUTION_FRAME_H

#include <mutex>
#include <vector>
#include "core/framework/ml_value.h"
#include "core/graph/graph.h"
#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
//#include "core/framework/session_state.h"

namespace Lotus
{
    namespace Test
    {
        class TestUtils;
    }

    class SessionState;

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

        ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                       const std::vector<std::string>& output_names,
                       const SessionState& session_state);

        ~ExecutionFrame() {
        }

        // Create tensor at index mlvalue, and allocate buffer for it.
        // This tensor will own this buffer.
        // This method is not thread safe!
        Status AllocateTensorWithSelfOwnBuffer(const int index,
            const MLDataType element_type,
            const AllocatorInfo& location,
            const TensorShape& shape);

        // Create tensor at index mlvalue, with pre-allocate buffer
        // This tensor does not own the buffer.
        // The executor / planner need to be careful about the 
        // lifetime of the buffer. Tensor itself won't manage it.
        // This method is not thread safe!
        Status AllocateTensorWithPreAllocateBuffer(const int offset,
            void* pBuffer,
            const MLDataType element_type,
            const AllocatorInfo& location,
            const TensorShape& shape);

        // Index to the first argument of the given node.
        int get_first_arg_index(LotusIR::NODEINDEX index) {
            LOTUS_ENFORCE(index >= 0 && index < node_offsets_.size());
            return node_offsets_[index];
        }

        template<typename T>
        const T* get_value(int index) const {
            LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
            return &node_values_[index]->Get<T>();
        }

        template<typename T>
        T* get_mutable_value(int index) {
            LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
            return node_values_[index]->GetMutable<T>();
        }

        ArenaPtr GetArena(const AllocatorInfo& info)
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
        
        // This method is not thread safe!
        void Release(const int offset);

        void Init(const LotusIR::Graph* graph, 
            const std::unordered_map<string, MLValue>& feeds,
            const std::vector<string>& outputs);

        void SetupNodeArg(LotusIR::NodeArg* arg);
        
        // This method is not thread safe!
        Tensor* get_or_create_tensor(int index, const TensorShape& shape) {
            LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
            auto value = node_values_[index];
            if (value->IsAllocated()) {
                // The tensor has already been allocated.
                // TODO: Check the size of the allocated tensor with given shape,
                // if they match each other, then return, else throw error.
                // TODO: type also needs to be checked and then use static_cast.
                Tensor* tensor = value->GetMutable<Tensor>();
                LOTUS_ENFORCE(tensor->shape() == shape);
                return tensor;
            }
            else {
                // It's not allocated, then allocate it with given shape and return.
                // TODO: at this point, we should already know the location and dtype
                // for the tensor, the graph should be able to tell us. But now graph
                // don't have it. So here hack to default as CPU and float.
                auto location = AllocatorManager::Instance()->GetArena(CPU).Info();
                auto dtype = DataTypeImpl::GetType<float>();
                LOTUS_ENFORCE(AllocateTensorWithSelfOwnBuffer(
                    index, 
                    dtype, 
                    location, 
                    shape).IsOK());
                return value->GetMutable<Tensor>();
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

        // All the intermedia values for the entire graph.
        // Input and Output values are passed in by executors
        vector<MLValue> all_values_;

        // The start index into node_values_ for all the nodes.
        std::vector<int> node_offsets_;

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

        std::unordered_map<string, int> value_name_to_index_;

        const SessionState& session_state_;
    };
}

#endif  // CORE_FRAMEWORK_EXECUTION_FRAME_H
