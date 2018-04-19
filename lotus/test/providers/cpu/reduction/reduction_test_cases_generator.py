import os
import numpy as np
import cntk as C

def TestReduction(op, data, axes, keepdims):
    if op == "ReduceL1":
        return C.reduce_l1(data, axis = axes, keepdims = keepdims == 1).eval()
    elif op == "ReduceL2":
        return C.reduce_l2(data, axis = axes, keepdims = keepdims == 1).eval()
    elif op == "ReduceLogSum":
        res = np.sum(np.log(data), axis = axes, keepdims = keepdims == 1)
        return res
    elif op == "ReduceLogSumExp":
        model = C.reduce_log_sum_exp(data, axis = axes)
        if (keepdims != 1):
            model = C.squeeze(model, axes = axes)
        return model.eval()
    elif op == "ReduceMax":
        res = np.max(data, axis = axes, keepdims = keepdims == 1)
        return res
    elif op == "ReduceMean":
        res = np.mean(data, axis = axes, keepdims = keepdims)
        return res
    elif op == "ReduceMin":
        res = np.min(data, axis = axes, keepdims = keepdims)
        return res
    elif op == "ReduceProd":
        model = C.reduce_prod(data, axis = axes)
        if (keepdims != 1):
            model = C.squeeze(model, axes = axes)
        return model.eval()
    elif op == "ReduceSum":
        res = np.sum(data, axis = axes, keepdims = keepdims)
        return res
    elif op == "ReduceSumSquare":
        res = np.sum(np.square(data), axis = axes, keepdims = keepdims)
        return res
    elif op == "ArgMax":
        res = np.argmax(data, axis = axes[0])
        if keepdims:
            res = np.expand_dims(res, axes[0])
        return res
    elif op == "ArgMin":
        res = np.argmin(data, axis = axes[0])
        if keepdims:
            res = np.expand_dims(res, axes[0])
        return res

def PrintResult(op, axes, keepdims, res):
    print("  {\"%s\"," % op)
    print("OpAttributesResult(")
    print("    // ReductionAttribute")
    print("      {")
    print (" // axes_")
    print ("{",  end='')
    print(*axes, sep=", ",  end='')
    print ("},")
    print (" // keep_dims_")
    print (keepdims, ",")
    print ("},")

    print (" // expected dims")
    print ("{",  end='')
    print(*res.shape, sep=", ",  end='')
    print ("},")

    print (" // expected values")
    print ("{",  end='')
    for i in range(0, res.size):
        print("%5.6ff," % res.item(i))

    print ("})},")

if __name__ == "__main__":
    from itertools import product
    input_shape = [2,3,2,2,3]
    np.random.seed(0)
    input_data = np.random.uniform(size=input_shape)
    axes_options = [(2,3), (2, 1, 4), (0, 2, 3), (0,), (2,), (4,)]
    keepdims_options = [0, 1]
    ops = ["ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean", 
           "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "ArgMax", "ArgMin"]
    print ("// Please don't manually edit this file. Generated from reduction_test_cases_generator.py")
    print ("ReductionTestCases testcases = {")
    print ("// input_data")
    print ("{")
    for i in range(0, input_data.size):
        print("%5.6ff," % input_data.item(i),)
    print ("},")
    print ("// input_dims")
    print ("{", end='')
    print(*input_shape, sep=", ", end='')
    print ("},")

    print("  // map_op_attribute_expected")
    print ("{")

    for config in product(axes_options, keepdims_options, ops):
        axes, keepdims, op = config
        
        #ArgMax and ArgMin only take single axis
        skip = False;
        if op == "ArgMax" or op == "ArgMin":
            skip = len(axes) > 1

        if not skip:
            res = TestReduction(op, input_data, axes, keepdims)
            PrintResult(op, axes, keepdims, res)

    print ("}")
    print ("};")
