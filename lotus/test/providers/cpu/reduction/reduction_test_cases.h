#pragma once

namespace Lotus {
namespace Test {
struct ReductionAttribute {
  std::vector<int64_t> axes_;
  int64_t keep_dims_;
};

typedef std::tuple<ReductionAttribute, std::vector<int64_t>, std::vector<float>> OpAttributesResult;
typedef std::multimap<std::string, OpAttributesResult> OpAttributesResultMap;
struct ReductionTestCases {
  std::vector<float> input_data_;
  std::vector<int64_t> input_dims_;

  OpAttributesResultMap map_op_attribute_expected_;
};

/*
    test data is generated with following python code:

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
from itertools import product
input_shape = [2,3,2,2,3]
input_data = np.random.uniform(size=input_shape)
axes_options = [(2,3), (2, 1, 4), (0, 2, 3)]
keepdims_options = [0, 1]
print ("// input_data_")
print ("{")
for i in range(0, input_data.size):
print("%5.6ff," % input_data.item(i),)
print ("},")
print ("// input_dims_")
print ("{", end='')
print(*input_shape, sep=", ", end='')
print ("},")

print("  // map_op_attribute_expected_")
print ("{")

for config in product(axes_options, keepdims_options):
    axes, keepdims = config

    op = "ReduceL1";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceL2";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceLogSum";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceLogSumExp";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceMax";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceMean";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceMin";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceProd";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceSum";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

    op = "ReduceSumSquare";
    res = TestReduction(op, input_data, axes, keepdims)
    PrintResult(op, axes, keepdims, res)

print ("}")
*/

ReductionTestCases testcases = {
    // input_data_
    {
        0.386928f,
        0.366349f,
        0.156029f,
        0.803152f,
        0.669347f,
        0.786927f,
        0.116783f,
        0.992597f,
        0.407939f,
        0.319136f,
        0.437474f,
        0.402205f,
        0.626443f,
        0.912320f,
        0.583840f,
        0.272436f,
        0.548837f,
        0.999187f,
        0.246557f,
        0.902238f,
        0.235322f,
        0.950525f,
        0.155981f,
        0.554578f,
        0.285247f,
        0.837742f,
        0.505560f,
        0.913605f,
        0.058592f,
        0.996266f,
        0.995292f,
        0.253380f,
        0.297805f,
        0.533719f,
        0.201560f,
        0.773094f,
        0.245394f,
        0.340602f,
        0.926403f,
        0.898356f,
        0.760306f,
        0.882147f,
        0.070999f,
        0.386120f,
        0.098060f,
        0.301741f,
        0.570136f,
        0.818627f,
        0.169706f,
        0.419190f,
        0.329421f,
        0.146100f,
        0.863944f,
        0.109973f,
        0.370687f,
        0.659951f,
        0.337975f,
        0.193819f,
        0.667611f,
        0.555733f,
        0.585693f,
        0.317747f,
        0.264899f,
        0.470643f,
        0.749396f,
        0.859603f,
        0.213388f,
        0.637605f,
        0.721399f,
        0.607934f,
        0.127394f,
        0.964857f,
    },
    // input_dims_
    {2, 3, 2, 2, 3},
    // map_op_attribute_expected_
    {
        {"ReduceL1",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 1.625999f,
                 2.465767f,
                 1.753100f,
                 2.095961f,
                 2.519376f,
                 2.372927f,
                 2.727862f,
                 1.351275f,
                 2.572726f,
                 1.516490f,
                 2.057164f,
                 2.725236f,
                 0.880312f,
                 2.610696f,
                 1.333102f,
                 1.877659f,
                 1.832142f,
                 2.810759f,
             })},
        {"ReduceL2",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 0.954071f,
                 1.326224f,
                 0.985790f,
                 1.196219f,
                 1.404248f,
                 1.304675f,
                 1.480372f,
                 0.900041f,
                 1.390863f,
                 0.981504f,
                 1.080842f,
                 1.521897f,
                 0.474468f,
                 1.342892f,
                 0.737346f,
                 0.989778f,
                 1.041790f,
                 1.503481f,
             })},
        {"ReduceLogSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 -4.458301f,
                 -2.239791f,
                 -3.904762f,
                 -3.218951f,
                 -2.652614f,
                 -2.575289f,
                 -1.977364f,
                 -5.988724f,
                 -2.154499f,
                 -5.355354f,
                 -2.864563f,
                 -2.724150f,
                 -6.330378f,
                 -1.835318f,
                 -4.990192f,
                 -3.330945f,
                 -3.945494f,
                 -1.842028f,
             })},
        {"ReduceLogSumExp",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 1.825328f,
                 2.033717f,
                 1.850709f,
                 1.953139f,
                 2.061182f,
                 2.016968f,
                 2.108291f,
                 1.772096f,
                 2.064255f,
                 1.817992f,
                 1.914640f,
                 2.117689f,
                 1.610412f,
                 2.051235f,
                 1.731963f,
                 1.867448f,
                 1.874500f,
                 2.121932f,
             })},
        {"ReduceMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 0.803152f,
                 0.992597f,
                 0.786927f,
                 0.950525f,
                 0.912320f,
                 0.999187f,
                 0.995292f,
                 0.837742f,
                 0.996266f,
                 0.898356f,
                 0.760306f,
                 0.926403f,
                 0.370687f,
                 0.863944f,
                 0.555733f,
                 0.607934f,
                 0.749396f,
                 0.964857f,
             })},
        {"ReduceMean",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 0.406500f,
                 0.616442f,
                 0.438275f,
                 0.523990f,
                 0.629844f,
                 0.593232f,
                 0.681966f,
                 0.337819f,
                 0.643181f,
                 0.379123f,
                 0.514291f,
                 0.681309f,
                 0.220078f,
                 0.652674f,
                 0.333276f,
                 0.469415f,
                 0.458036f,
                 0.702690f,
             })},
        {"ReduceMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 0.116783f,
                 0.366349f,
                 0.156029f,
                 0.246557f,
                 0.155981f,
                 0.235322f,
                 0.285247f,
                 0.058592f,
                 0.297805f,
                 0.070999f,
                 0.340602f,
                 0.098060f,
                 0.146100f,
                 0.419190f,
                 0.109973f,
                 0.213388f,
                 0.127394f,
                 0.264899f,
             })},
        {"ReduceProd",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 0.011582f,
                 0.106481f,
                 0.020146f,
                 0.039997f,
                 0.070467f,
                 0.076132f,
                 0.138434f,
                 0.002507f,
                 0.115961f,
                 0.004723f,
                 0.057008f,
                 0.065602f,
                 0.001781f,
                 0.159563f,
                 0.006804f,
                 0.035759f,
                 0.019342f,
                 0.158496f,
             })},
        {"ReduceSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 1.625999f,
                 2.465767f,
                 1.753100f,
                 2.095961f,
                 2.519376f,
                 2.372928f,
                 2.727862f,
                 1.351275f,
                 2.572726f,
                 1.516490f,
                 2.057164f,
                 2.725236f,
                 0.880312f,
                 2.610696f,
                 1.333102f,
                 1.877659f,
                 1.832143f,
                 2.810759f,
             })},
        {"ReduceSumSquare",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 3},
             // expected values
             {
                 0.910252f,
                 1.758869f,
                 0.971783f,
                 1.430940f,
                 1.971914f,
                 1.702178f,
                 2.191501f,
                 0.810073f,
                 1.934500f,
                 0.963350f,
                 1.168219f,
                 2.316172f,
                 0.225120f,
                 1.803359f,
                 0.543679f,
                 0.979660f,
                 1.085328f,
                 2.260456f,
             })},
        {"ReduceL1",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 1.625999f,
                 2.465767f,
                 1.753100f,
                 2.095961f,
                 2.519376f,
                 2.372927f,
                 2.727862f,
                 1.351275f,
                 2.572726f,
                 1.516490f,
                 2.057164f,
                 2.725236f,
                 0.880312f,
                 2.610696f,
                 1.333102f,
                 1.877659f,
                 1.832142f,
                 2.810759f,
             })},
        {"ReduceL2",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 0.954071f,
                 1.326224f,
                 0.985790f,
                 1.196219f,
                 1.404248f,
                 1.304675f,
                 1.480372f,
                 0.900041f,
                 1.390863f,
                 0.981504f,
                 1.080842f,
                 1.521897f,
                 0.474468f,
                 1.342892f,
                 0.737346f,
                 0.989778f,
                 1.041790f,
                 1.503481f,
             })},
        {"ReduceLogSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 -4.458301f,
                 -2.239791f,
                 -3.904762f,
                 -3.218951f,
                 -2.652614f,
                 -2.575289f,
                 -1.977364f,
                 -5.988724f,
                 -2.154499f,
                 -5.355354f,
                 -2.864563f,
                 -2.724150f,
                 -6.330378f,
                 -1.835318f,
                 -4.990192f,
                 -3.330945f,
                 -3.945494f,
                 -1.842028f,
             })},
        {"ReduceLogSumExp",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 1.825328f,
                 2.033717f,
                 1.850709f,
                 1.953139f,
                 2.061182f,
                 2.016968f,
                 2.108291f,
                 1.772096f,
                 2.064255f,
                 1.817992f,
                 1.914640f,
                 2.117689f,
                 1.610412f,
                 2.051235f,
                 1.731963f,
                 1.867448f,
                 1.874500f,
                 2.121932f,
             })},
        {"ReduceMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 0.803152f,
                 0.992597f,
                 0.786927f,
                 0.950525f,
                 0.912320f,
                 0.999187f,
                 0.995292f,
                 0.837742f,
                 0.996266f,
                 0.898356f,
                 0.760306f,
                 0.926403f,
                 0.370687f,
                 0.863944f,
                 0.555733f,
                 0.607934f,
                 0.749396f,
                 0.964857f,
             })},
        {"ReduceMean",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 0.406500f,
                 0.616442f,
                 0.438275f,
                 0.523990f,
                 0.629844f,
                 0.593232f,
                 0.681966f,
                 0.337819f,
                 0.643181f,
                 0.379123f,
                 0.514291f,
                 0.681309f,
                 0.220078f,
                 0.652674f,
                 0.333276f,
                 0.469415f,
                 0.458036f,
                 0.702690f,
             })},
        {"ReduceMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 0.116783f,
                 0.366349f,
                 0.156029f,
                 0.246557f,
                 0.155981f,
                 0.235322f,
                 0.285247f,
                 0.058592f,
                 0.297805f,
                 0.070999f,
                 0.340602f,
                 0.098060f,
                 0.146100f,
                 0.419190f,
                 0.109973f,
                 0.213388f,
                 0.127394f,
                 0.264899f,
             })},
        {"ReduceProd",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 0.011582f,
                 0.106481f,
                 0.020146f,
                 0.039997f,
                 0.070467f,
                 0.076132f,
                 0.138434f,
                 0.002507f,
                 0.115961f,
                 0.004723f,
                 0.057008f,
                 0.065602f,
                 0.001781f,
                 0.159563f,
                 0.006804f,
                 0.035759f,
                 0.019342f,
                 0.158496f,
             })},
        {"ReduceSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 1.625999f,
                 2.465767f,
                 1.753100f,
                 2.095961f,
                 2.519376f,
                 2.372928f,
                 2.727862f,
                 1.351275f,
                 2.572726f,
                 1.516490f,
                 2.057164f,
                 2.725236f,
                 0.880312f,
                 2.610696f,
                 1.333102f,
                 1.877659f,
                 1.832143f,
                 2.810759f,
             })},
        {"ReduceSumSquare",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 1, 3},
             // expected values
             {
                 0.910252f,
                 1.758869f,
                 0.971783f,
                 1.430940f,
                 1.971914f,
                 1.702178f,
                 2.191501f,
                 0.810073f,
                 1.934500f,
                 0.963350f,
                 1.168219f,
                 2.316172f,
                 0.225120f,
                 1.803359f,
                 0.543679f,
                 0.979660f,
                 1.085328f,
                 2.260456f,
             })},
        {"ReduceL1",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 9.108372f,
                 10.376621f,
                 7.095240f,
                 10.548320f,
             })},
        {"ReduceL2",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 2.480255f,
                 2.744148f,
                 1.919653f,
                 2.767720f,
             })},
        {"ReduceLogSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 -15.612760f,
                 -13.557534f,
                 -20.054667f,
                 -13.163755f,
             })},
        {"ReduceLogSumExp",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 3.440851f,
                 3.509046f,
                 3.310516f,
                 3.515338f,
             })},
        {"ReduceMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 0.995292f,
                 0.999187f,
                 0.926403f,
                 0.964857f,
             })},
        {"ReduceMean",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 0.506021f,
                 0.576479f,
                 0.394180f,
                 0.586018f,
             })},
        {"ReduceMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 0.116783f,
                 0.058592f,
                 0.070999f,
                 0.109973f,
             })},
        {"ReduceProd",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 0.000000f,
                 0.000001f,
                 0.000000f,
                 0.000002f,
             })},
        {"ReduceSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 9.108372f,
                 10.376622f,
                 7.095240f,
                 10.548321f,
             })},
        {"ReduceSumSquare",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 2},
             // expected values
             {
                 6.151664f,
                 7.530346f,
                 3.685069f,
                 7.660272f,
             })},
        {"ReduceL1",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 9.108372f,
                 10.376621f,
                 7.095240f,
                 10.548320f,
             })},
        {"ReduceL2",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 2.480255f,
                 2.744148f,
                 1.919653f,
                 2.767720f,
             })},
        {"ReduceLogSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 -15.612760f,
                 -13.557534f,
                 -20.054667f,
                 -13.163755f,
             })},
        {"ReduceLogSumExp",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 3.440851f,
                 3.509046f,
                 3.310516f,
                 3.515338f,
             })},
        {"ReduceMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 0.995292f,
                 0.999187f,
                 0.926403f,
                 0.964857f,
             })},
        {"ReduceMean",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 0.506021f,
                 0.576479f,
                 0.394180f,
                 0.586018f,
             })},
        {"ReduceMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 0.116783f,
                 0.058592f,
                 0.070999f,
                 0.109973f,
             })},
        {"ReduceProd",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 0.000000f,
                 0.000001f,
                 0.000000f,
                 0.000002f,
             })},
        {"ReduceSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 9.108372f,
                 10.376622f,
                 7.095240f,
                 10.548321f,
             })},
        {"ReduceSumSquare",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2, 1, 4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 1, 1, 2, 1},
             // expected values
             {
                 6.151664f,
                 7.530346f,
                 3.685069f,
                 7.660272f,
             })},
        {"ReduceL1",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 3.142489f,
                 4.522931f,
                 4.478337f,
                 2.976274f,
                 5.130072f,
                 3.706030f,
                 4.605521f,
                 3.183418f,
                 5.383485f,
             })},
        {"ReduceL2",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 1.368796f,
                 1.710873f,
                 1.813272f,
                 1.286880f,
                 1.943006f,
                 1.498618f,
                 1.780775f,
                 1.376735f,
                 2.048159f,
             })},
        {"ReduceLogSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 -9.813654f,
                 -5.104353f,
                 -6.628912f,
                 -9.549329f,
                 -4.487932f,
                 -7.565481f,
                 -5.308308f,
                 -9.934219f,
                 -3.996527f,
             })},
        {"ReduceLogSumExp",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 2.514814f,
                 2.669097f,
                 2.686229f,
                 2.489534f,
                 2.749368f,
                 2.577732f,
                 2.688250f,
                 2.517755f,
                 2.786656f,
             })},
        {"ReduceMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 0.898356f,
                 0.992597f,
                 0.926403f,
                 0.950525f,
                 0.912320f,
                 0.999187f,
                 0.995292f,
                 0.837742f,
                 0.996266f,
             })},
        {"ReduceMean",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 0.392811f,
                 0.565366f,
                 0.559792f,
                 0.372034f,
                 0.641259f,
                 0.463254f,
                 0.575690f,
                 0.397927f,
                 0.672936f,
             })},
        {"ReduceMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 0.070999f,
                 0.340602f,
                 0.098060f,
                 0.146100f,
                 0.155981f,
                 0.109973f,
                 0.213388f,
                 0.058592f,
                 0.264899f,
             })},
        {"ReduceProd",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 0.000055f,
                 0.006070f,
                 0.001322f,
                 0.000071f,
                 0.011244f,
                 0.000518f,
                 0.004950f,
                 0.000048f,
                 0.018379f,
             })},
        {"ReduceSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 3.142489f,
                 4.522931f,
                 4.478337f,
                 2.976273f,
                 5.130072f,
                 3.706030f,
                 4.605521f,
                 3.183418f,
                 5.383485f,
             })},
        {"ReduceSumSquare",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 3},
             // expected values
             {
                 1.873602f,
                 2.927088f,
                 3.287954f,
                 1.656060f,
                 3.775273f,
                 2.245857f,
                 3.171161f,
                 1.895401f,
                 4.194956f,
             })},
        {"ReduceL1",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 3.142489f,
                 4.522931f,
                 4.478337f,
                 2.976274f,
                 5.130072f,
                 3.706030f,
                 4.605521f,
                 3.183418f,
                 5.383485f,
             })},
        {"ReduceL2",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 1.368796f,
                 1.710873f,
                 1.813272f,
                 1.286880f,
                 1.943006f,
                 1.498618f,
                 1.780775f,
                 1.376735f,
                 2.048159f,
             })},
        {"ReduceLogSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 -9.813654f,
                 -5.104353f,
                 -6.628912f,
                 -9.549329f,
                 -4.487932f,
                 -7.565481f,
                 -5.308308f,
                 -9.934219f,
                 -3.996527f,
             })},
        {"ReduceLogSumExp",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 2.514814f,
                 2.669097f,
                 2.686229f,
                 2.489534f,
                 2.749368f,
                 2.577732f,
                 2.688250f,
                 2.517755f,
                 2.786656f,
             })},
        {"ReduceMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 0.898356f,
                 0.992597f,
                 0.926403f,
                 0.950525f,
                 0.912320f,
                 0.999187f,
                 0.995292f,
                 0.837742f,
                 0.996266f,
             })},
        {"ReduceMean",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 0.392811f,
                 0.565366f,
                 0.559792f,
                 0.372034f,
                 0.641259f,
                 0.463254f,
                 0.575690f,
                 0.397927f,
                 0.672936f,
             })},
        {"ReduceMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 0.070999f,
                 0.340602f,
                 0.098060f,
                 0.146100f,
                 0.155981f,
                 0.109973f,
                 0.213388f,
                 0.058592f,
                 0.264899f,
             })},
        {"ReduceProd",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 0.000055f,
                 0.006070f,
                 0.001322f,
                 0.000071f,
                 0.011244f,
                 0.000518f,
                 0.004950f,
                 0.000048f,
                 0.018379f,
             })},
        {"ReduceSum",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 3.142489f,
                 4.522931f,
                 4.478337f,
                 2.976273f,
                 5.130072f,
                 3.706030f,
                 4.605521f,
                 3.183418f,
                 5.383485f,
             })},
        {"ReduceSumSquare",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0, 2, 3},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 1, 1, 3},
             // expected values
             {
                 1.873602f,
                 2.927088f,
                 3.287954f,
                 1.656060f,
                 3.775273f,
                 2.245857f,
                 3.171161f,
                 1.895401f,
                 4.194956f,
             })},
    }};

}  // namespace Test
}  // namespace Lotus
