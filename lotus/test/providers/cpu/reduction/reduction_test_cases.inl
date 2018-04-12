// Please don't manually edit this file. Generated from reduction_test_cases_generator.py
ReductionTestCases testcases = {
    // input_data
    {
        0.173873f,
        0.162377f,
        0.128382f,
        0.471156f,
        0.231175f,
        0.963752f,
        0.638650f,
        0.493665f,
        0.947692f,
        0.656611f,
        0.374315f,
        0.127104f,
        0.464905f,
        0.630366f,
        0.434652f,
        0.110208f,
        0.517520f,
        0.503517f,
        0.924441f,
        0.288796f,
        0.414257f,
        0.010688f,
        0.978599f,
        0.949431f,
        0.003015f,
        0.226244f,
        0.437479f,
        0.666499f,
        0.514862f,
        0.798796f,
        0.583368f,
        0.077543f,
        0.811930f,
        0.984628f,
        0.932310f,
        0.191445f,
        0.128417f,
        0.722960f,
        0.888399f,
        0.334182f,
        0.612728f,
        0.497867f,
        0.070958f,
        0.588102f,
        0.312739f,
        0.878979f,
        0.562831f,
        0.013986f,
        0.174777f,
        0.047096f,
        0.807133f,
        0.018700f,
        0.365126f,
        0.366105f,
        0.854438f,
        0.980941f,
        0.213769f,
        0.034512f,
        0.669924f,
        0.680533f,
        0.857328f,
        0.720448f,
        0.127831f,
        0.780965f,
        0.592376f,
        0.321963f,
        0.629134f,
        0.603569f,
        0.614648f,
        0.494387f,
        0.274134f,
        0.214345f,
    },
    // input_dims
    {2, 3, 2, 2, 3},
    // map_op_attribute_expected
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
                 1.940290f,
                 1.261533f,
                 2.166929f,
                 1.510241f,
                 2.415281f,
                 2.301857f,
                 2.237510f,
                 1.750960f,
                 2.239649f,
                 1.412537f,
                 2.486621f,
                 1.712990f,
                 1.082426f,
                 2.063087f,
                 2.067541f,
                 2.761812f,
                 2.190528f,
                 1.278787f,
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
                 1.044620f,
                 0.680901f,
                 1.363662f,
                 1.040667f,
                 1.306234f,
                 1.231048f,
                 1.324402f,
                 1.091552f,
                 1.235048f,
                 0.951739f,
                 1.249300f,
                 1.065422f,
                 0.873013f,
                 1.243615f,
                 1.137682f,
                 1.408951f,
                 1.144290f,
                 0.737385f,
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
                 -3.371059f,
                 -4.970969f,
                 -4.206143f,
                 -7.588504f,
                 -2.383830f,
                 -2.452508f,
                 -6.764148f,
                 -4.777006f,
                 -2.912873f,
                 -5.923197f,
                 -1.919866f,
                 -6.247835f,
                 -9.247235f,
                 -4.482922f,
                 -3.146837f,
                 -1.569010f,
                 -2.650527f,
                 -5.217240f,
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
                 1.889176f,
                 1.709951f,
                 2.011401f,
                 1.830912f,
                 2.021773f,
                 1.987387f,
                 2.004086f,
                 1.879204f,
                 1.978918f,
                 1.794389f,
                 2.009848f,
                 1.865272f,
                 1.721877f,
                 1.961498f,
                 1.931064f,
                 2.086406f,
                 1.947004f,
                 1.723525f,
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
                 0.656611f,
                 0.493665f,
                 0.963752f,
                 0.924441f,
                 0.978599f,
                 0.949431f,
                 0.984628f,
                 0.932310f,
                 0.811930f,
                 0.878979f,
                 0.722960f,
                 0.888399f,
                 0.854438f,
                 0.980941f,
                 0.807133f,
                 0.857328f,
                 0.720448f,
                 0.614648f,
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
                 0.485073f,
                 0.315383f,
                 0.541732f,
                 0.377560f,
                 0.603820f,
                 0.575464f,
                 0.559378f,
                 0.437740f,
                 0.559912f,
                 0.353134f,
                 0.621655f,
                 0.428248f,
                 0.270607f,
                 0.515772f,
                 0.516885f,
                 0.690453f,
                 0.547632f,
                 0.319697f,
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
                 0.173873f,
                 0.162377f,
                 0.127104f,
                 0.010688f,
                 0.288796f,
                 0.414257f,
                 0.003015f,
                 0.077543f,
                 0.191445f,
                 0.070958f,
                 0.562831f,
                 0.013986f,
                 0.018700f,
                 0.047096f,
                 0.213769f,
                 0.494387f,
                 0.274134f,
                 0.127831f,
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
                 0.034353f,
                 0.006936f,
                 0.014904f,
                 0.000506f,
                 0.092197f,
                 0.086077f,
                 0.001154f,
                 0.008421f,
                 0.054319f,
                 0.002677f,
                 0.146627f,
                 0.001935f,
                 0.000096f,
                 0.011300f,
                 0.042988f,
                 0.208251f,
                 0.070614f,
                 0.005422f,
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
                 1.940290f,
                 1.261533f,
                 2.166929f,
                 1.510241f,
                 2.415281f,
                 2.301857f,
                 2.237510f,
                 1.750960f,
                 2.239649f,
                 1.412537f,
                 2.486621f,
                 1.712990f,
                 1.082426f,
                 2.063087f,
                 2.067541f,
                 2.761812f,
                 2.190528f,
                 1.278787f,
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
                 1.091232f,
                 0.463626f,
                 1.859574f,
                 1.082987f,
                 1.706246f,
                 1.515480f,
                 1.754040f,
                 1.191485f,
                 1.525344f,
                 0.905808f,
                 1.560749f,
                 1.135125f,
                 0.762151f,
                 1.546579f,
                 1.294320f,
                 1.985143f,
                 1.309400f,
                 0.543737f,
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
                 1.940290f,
                 1.261533f,
                 2.166929f,
                 1.510241f,
                 2.415281f,
                 2.301857f,
                 2.237510f,
                 1.750960f,
                 2.239649f,
                 1.412537f,
                 2.486621f,
                 1.712990f,
                 1.082426f,
                 2.063087f,
                 2.067541f,
                 2.761812f,
                 2.190528f,
                 1.278787f,
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
                 1.044620f,
                 0.680901f,
                 1.363662f,
                 1.040667f,
                 1.306234f,
                 1.231048f,
                 1.324402f,
                 1.091552f,
                 1.235048f,
                 0.951739f,
                 1.249300f,
                 1.065422f,
                 0.873013f,
                 1.243615f,
                 1.137682f,
                 1.408951f,
                 1.144290f,
                 0.737385f,
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
                 -3.371059f,
                 -4.970969f,
                 -4.206143f,
                 -7.588504f,
                 -2.383830f,
                 -2.452508f,
                 -6.764148f,
                 -4.777006f,
                 -2.912873f,
                 -5.923197f,
                 -1.919866f,
                 -6.247835f,
                 -9.247235f,
                 -4.482922f,
                 -3.146837f,
                 -1.569010f,
                 -2.650527f,
                 -5.217240f,
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
                 1.889176f,
                 1.709951f,
                 2.011401f,
                 1.830912f,
                 2.021773f,
                 1.987387f,
                 2.004086f,
                 1.879204f,
                 1.978918f,
                 1.794389f,
                 2.009848f,
                 1.865272f,
                 1.721877f,
                 1.961498f,
                 1.931064f,
                 2.086406f,
                 1.947004f,
                 1.723525f,
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
                 0.656611f,
                 0.493665f,
                 0.963752f,
                 0.924441f,
                 0.978599f,
                 0.949431f,
                 0.984628f,
                 0.932310f,
                 0.811930f,
                 0.878979f,
                 0.722960f,
                 0.888399f,
                 0.854438f,
                 0.980941f,
                 0.807133f,
                 0.857328f,
                 0.720448f,
                 0.614648f,
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
                 0.485073f,
                 0.315383f,
                 0.541732f,
                 0.377560f,
                 0.603820f,
                 0.575464f,
                 0.559378f,
                 0.437740f,
                 0.559912f,
                 0.353134f,
                 0.621655f,
                 0.428248f,
                 0.270607f,
                 0.515772f,
                 0.516885f,
                 0.690453f,
                 0.547632f,
                 0.319697f,
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
                 0.173873f,
                 0.162377f,
                 0.127104f,
                 0.010688f,
                 0.288796f,
                 0.414257f,
                 0.003015f,
                 0.077543f,
                 0.191445f,
                 0.070958f,
                 0.562831f,
                 0.013986f,
                 0.018700f,
                 0.047096f,
                 0.213769f,
                 0.494387f,
                 0.274134f,
                 0.127831f,
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
                 0.034353f,
                 0.006936f,
                 0.014904f,
                 0.000506f,
                 0.092197f,
                 0.086077f,
                 0.001154f,
                 0.008421f,
                 0.054319f,
                 0.002677f,
                 0.146627f,
                 0.001935f,
                 0.000096f,
                 0.011300f,
                 0.042988f,
                 0.208251f,
                 0.070614f,
                 0.005422f,
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
                 1.940290f,
                 1.261533f,
                 2.166929f,
                 1.510241f,
                 2.415281f,
                 2.301857f,
                 2.237510f,
                 1.750960f,
                 2.239649f,
                 1.412537f,
                 2.486621f,
                 1.712990f,
                 1.082426f,
                 2.063087f,
                 2.067541f,
                 2.761812f,
                 2.190528f,
                 1.278787f,
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
                 1.091232f,
                 0.463626f,
                 1.859574f,
                 1.082987f,
                 1.706246f,
                 1.515480f,
                 1.754040f,
                 1.191485f,
                 1.525344f,
                 0.905808f,
                 1.560749f,
                 1.135125f,
                 0.762151f,
                 1.546579f,
                 1.294320f,
                 1.985143f,
                 1.309400f,
                 0.543737f,
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
                 7.841635f,
                 9.982615f,
                 9.342687f,
                 7.713642f,
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
                 2.186995f,
                 2.721593f,
                 2.570535f,
                 2.106030f,
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
                 -22.511561f,
                 -16.915480f,
                 -17.561486f,
                 -22.843183f,
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
                 3.364842f,
                 3.495831f,
                 3.456578f,
                 3.349801f,
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
                 0.947692f,
                 0.984628f,
                 0.980941f,
                 0.878979f,
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
                 0.435646f,
                 0.554590f,
                 0.519038f,
                 0.428536f,
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
                 0.003015f,
                 0.010688f,
                 0.047096f,
                 0.013986f,
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
                 0.000000f,
                 0.000000f,
                 0.000000f,
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
                 7.841635f,
                 9.982615f,
                 9.342686f,
                 7.713642f,
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
                 4.782946f,
                 7.407067f,
                 6.607648f,
                 4.435364f,
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
                 7.841635f,
                 9.982615f,
                 9.342687f,
                 7.713642f,
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
                 2.186995f,
                 2.721593f,
                 2.570535f,
                 2.106030f,
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
                 -22.511561f,
                 -16.915480f,
                 -17.561486f,
                 -22.843183f,
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
                 3.364842f,
                 3.495831f,
                 3.456578f,
                 3.349801f,
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
                 0.947692f,
                 0.984628f,
                 0.980941f,
                 0.878979f,
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
                 0.435646f,
                 0.554590f,
                 0.519038f,
                 0.428536f,
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
                 0.003015f,
                 0.010688f,
                 0.047096f,
                 0.013986f,
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
                 0.000000f,
                 0.000000f,
                 0.000000f,
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
                 7.841635f,
                 9.982615f,
                 9.342686f,
                 7.713642f,
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
                 4.782946f,
                 7.407067f,
                 6.607648f,
                 4.435364f,
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
                 3.352827f,
                 3.748154f,
                 3.879920f,
                 2.592668f,
                 4.478367f,
                 4.369398f,
                 4.999322f,
                 3.941487f,
                 3.518436f,
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
                 1.413167f,
                 1.422805f,
                 1.730520f,
                 1.358359f,
                 1.803559f,
                 1.676246f,
                 1.933697f,
                 1.581419f,
                 1.438430f,
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
                 -9.294256f,
                 -6.890835f,
                 -10.453978f,
                 -16.835739f,
                 -6.866752f,
                 -5.599345f,
                 -8.333159f,
                 -7.427533f,
                 -8.130113f,
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
                 2.536053f,
                 2.564247f,
                 2.634151f,
                 2.471027f,
                 2.685237f,
                 2.652769f,
                 2.739240f,
                 2.606826f,
                 2.552500f,
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
                 0.878979f,
                 0.722960f,
                 0.963752f,
                 0.924441f,
                 0.980941f,
                 0.949431f,
                 0.984628f,
                 0.932310f,
                 0.811930f,
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
                 0.419103f,
                 0.468519f,
                 0.484990f,
                 0.324083f,
                 0.559796f,
                 0.546175f,
                 0.624915f,
                 0.492686f,
                 0.439805f,
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
                 0.070958f,
                 0.162377f,
                 0.013986f,
                 0.010688f,
                 0.047096f,
                 0.213769f,
                 0.003015f,
                 0.077543f,
                 0.127831f,
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
                 0.000092f,
                 0.001017f,
                 0.000029f,
                 0.000000f,
                 0.001042f,
                 0.003700f,
                 0.000240f,
                 0.000595f,
                 0.000295f,
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
                 3.352827f,
                 3.748154f,
                 3.879919f,
                 2.592668f,
                 4.478367f,
                 4.369398f,
                 4.999322f,
                 3.941487f,
                 3.518436f,
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
                 1.997040f,
                 2.024375f,
                 2.994699f,
                 1.845138f,
                 3.252825f,
                 2.809799f,
                 3.739183f,
                 2.500885f,
                 2.069080f,
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
                 3.352827f,
                 3.748154f,
                 3.879920f,
                 2.592668f,
                 4.478367f,
                 4.369398f,
                 4.999322f,
                 3.941487f,
                 3.518436f,
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
                 1.413167f,
                 1.422805f,
                 1.730520f,
                 1.358359f,
                 1.803559f,
                 1.676246f,
                 1.933697f,
                 1.581419f,
                 1.438430f,
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
                 -9.294256f,
                 -6.890835f,
                 -10.453978f,
                 -16.835739f,
                 -6.866752f,
                 -5.599345f,
                 -8.333159f,
                 -7.427533f,
                 -8.130113f,
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
                 2.536053f,
                 2.564247f,
                 2.634151f,
                 2.471027f,
                 2.685237f,
                 2.652769f,
                 2.739240f,
                 2.606826f,
                 2.552500f,
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
                 0.878979f,
                 0.722960f,
                 0.963752f,
                 0.924441f,
                 0.980941f,
                 0.949431f,
                 0.984628f,
                 0.932310f,
                 0.811930f,
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
                 0.419103f,
                 0.468519f,
                 0.484990f,
                 0.324083f,
                 0.559796f,
                 0.546175f,
                 0.624915f,
                 0.492686f,
                 0.439805f,
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
                 0.070958f,
                 0.162377f,
                 0.013986f,
                 0.010688f,
                 0.047096f,
                 0.213769f,
                 0.003015f,
                 0.077543f,
                 0.127831f,
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
                 0.000092f,
                 0.001017f,
                 0.000029f,
                 0.000000f,
                 0.001042f,
                 0.003700f,
                 0.000240f,
                 0.000595f,
                 0.000295f,
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
                 3.352827f,
                 3.748154f,
                 3.879919f,
                 2.592668f,
                 4.478367f,
                 4.369398f,
                 4.999322f,
                 3.941487f,
                 3.518436f,
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
                 1.997040f,
                 2.024375f,
                 2.994699f,
                 1.845138f,
                 3.252825f,
                 2.809799f,
                 3.739183f,
                 2.500885f,
                 2.069080f,
             })},
        {"ArgMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 2, 2, 3},
             // expected values
             {
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
             })},
        {"ArgMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0},
                 // keep_dims_
                 0,
             },
             // expected dims
             {3, 2, 2, 3},
             // expected values
             {
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
             })},
        {"ArgMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 2, 2, 3},
             // expected values
             {
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
             })},
        {"ArgMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {0},
                 // keep_dims_
                 1,
             },
             // expected dims
             {1, 3, 2, 2, 3},
             // expected values
             {
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
             })},
        {"ArgMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 2, 3},
             // expected values
             {
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
             })},
        {"ArgMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 2, 3},
             // expected values
             {
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
             })},
        {"ArgMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 2, 3},
             // expected values
             {
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
             })},
        {"ArgMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {2},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 1, 2, 3},
             // expected values
             {
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 1.000000f,
             })},
        {"ArgMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 2, 2},
             // expected values
             {
                 0.000000f,
                 2.000000f,
                 2.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 2.000000f,
                 2.000000f,
                 2.000000f,
                 0.000000f,
                 2.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 2.000000f,
                 2.000000f,
                 1.000000f,
                 2.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
             })},
        {"ArgMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {4},
                 // keep_dims_
                 0,
             },
             // expected dims
             {2, 3, 2, 2},
             // expected values
             {
                 2.000000f,
                 1.000000f,
                 1.000000f,
                 2.000000f,
                 2.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 2.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 2.000000f,
                 1.000000f,
                 0.000000f,
                 2.000000f,
                 0.000000f,
                 2.000000f,
                 2.000000f,
                 1.000000f,
                 2.000000f,
             })},
        {"ArgMax",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 2, 2, 1},
             // expected values
             {
                 0.000000f,
                 2.000000f,
                 2.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 1.000000f,
                 2.000000f,
                 2.000000f,
                 2.000000f,
                 0.000000f,
                 2.000000f,
                 1.000000f,
                 1.000000f,
                 0.000000f,
                 2.000000f,
                 2.000000f,
                 1.000000f,
                 2.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
             })},
        {"ArgMin",
         OpAttributesResult(
             // ReductionAttribute
             {
                 // axes_
                 {4},
                 // keep_dims_
                 1,
             },
             // expected dims
             {2, 3, 2, 2, 1},
             // expected values
             {
                 2.000000f,
                 1.000000f,
                 1.000000f,
                 2.000000f,
                 2.000000f,
                 0.000000f,
                 1.000000f,
                 0.000000f,
                 0.000000f,
                 1.000000f,
                 1.000000f,
                 2.000000f,
                 0.000000f,
                 0.000000f,
                 0.000000f,
                 2.000000f,
                 1.000000f,
                 0.000000f,
                 2.000000f,
                 0.000000f,
                 2.000000f,
                 2.000000f,
                 1.000000f,
                 2.000000f,
             })},
    }};
