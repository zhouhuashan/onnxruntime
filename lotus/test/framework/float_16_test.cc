#include "core/framework/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "core/inc/op_kernel_author_helper.h"

#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include "core/graph/schema_registry.h"

namespace Lotus {
namespace Test {

// The code below is port from numpy
// Only for testing purpose now


/*
* This chooses between 'ties to even' and 'ties away from zero'.
*/
#define NPY_HALF_ROUND_TIES_TO_EVEN 1
/*
* If these are 1, the conversions try to trigger underflow,
* overflow, and invalid exceptions in the FP system when needed.
*/
//#define NPY_HALF_GENERATE_OVERFLOW 1
//#define NPY_HALF_GENERATE_UNDERFLOW 1
#define NPY_HALF_GENERATE_INVALID 1

/*
********************************************************************
*                   HALF-PRECISION ROUTINES                        *
********************************************************************
*/

uint32_t npy_halfbits_to_floatbits(uint16_t h) {
  uint16_t h_exp, h_sig;
  uint32_t f_sgn, f_exp, f_sig;
  h_exp = (h & 0x7c00u);
  f_sgn = ((uint32_t)h & 0x8000u) << 16;
  switch (h_exp) {
    case 0x0000u: /* 0 or subnormal */
      h_sig = (h & 0x03ffu);
      /* Signed zero */
      if (h_sig == 0) {
        return f_sgn;
      }
      /* Subnormal */
      h_sig <<= 1;
      while ((h_sig & 0x0400u) == 0) {
        h_sig <<= 1;
        h_exp++;
      }
      f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
      f_sig = ((uint32_t)(h_sig & 0x03ffu)) << 13;
      return f_sgn + f_exp + f_sig;
    case 0x7c00u: /* inf or NaN */
      /* All-ones exponent and a copy of the significand */
      return f_sgn + 0x7f800000u + (((uint32_t)(h & 0x03ffu)) << 13);
    default: /* normalized */
      /* Just need to adjust the exponent and shift */
      return f_sgn + (((uint32_t)(h & 0x7fffu) + 0x1c000u) << 13);
  }
}


uint16_t npy_floatbits_to_halfbits(uint32_t f)
{
	uint32_t f_exp, f_sig;
	uint16_t h_sgn, h_exp, h_sig;

	h_sgn = (uint16_t)((f & 0x80000000u) >> 16);
	f_exp = (f & 0x7f800000u);

	/* Exponent overflow/NaN converts to signed inf/NaN */
	if (f_exp >= 0x47800000u) {
		if (f_exp == 0x7f800000u) {
			/* Inf or NaN */
			f_sig = (f & 0x007fffffu);
			if (f_sig != 0) {
				/* NaN - propagate the flag in the significand... */
				uint16_t ret = (uint16_t)(0x7c00u + (f_sig >> 13));
				/* ...but make sure it stays a NaN */
				if (ret == 0x7c00u) {
					ret++;
				}
				return h_sgn + ret;
			}
			else {
				/* signed inf */
				return (uint16_t)(h_sgn + 0x7c00u);
			}
		}
		else {
			/* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
			npy_set_floatstatus_overflow();
#endif
			return (uint16_t)(h_sgn + 0x7c00u);
		}
	}

	/* Exponent underflow converts to a subnormal half or signed zero */
	if (f_exp <= 0x38000000u) {
		/*
		* Signed zeros, subnormal floats, and floats with small
		* exponents all convert to signed zero half-floats.
		*/
		if (f_exp < 0x33000000u) {
#if NPY_HALF_GENERATE_UNDERFLOW
			/* If f != 0, it underflowed to 0 */
			if ((f & 0x7fffffff) != 0) {
				npy_set_floatstatus_underflow();
			}
#endif
			return h_sgn;
		}
		/* Make the subnormal significand */
		f_exp >>= 23;
		f_sig = (0x00800000u + (f & 0x007fffffu));
#if NPY_HALF_GENERATE_UNDERFLOW
		/* If it's not exactly represented, it underflowed */
		if ((f_sig&(((npy_uint32)1 << (126 - f_exp)) - 1)) != 0) {
			npy_set_floatstatus_underflow();
		}
#endif
		f_sig >>= (113 - f_exp);
		/* Handle rounding by adding 1 to the bit beyond half precision */
#if NPY_HALF_ROUND_TIES_TO_EVEN
		/*
		* If the last bit in the half significand is 0 (already even), and
		* the remaining bit pattern is 1000...0, then we do not add one
		* to the bit after the half significand.  In all other cases, we do.
		*/
		if ((f_sig & 0x00003fffu) != 0x00001000u) {
			f_sig += 0x00001000u;
		}
#else
		f_sig += 0x00001000u;
#endif
		h_sig = (uint16_t)(f_sig >> 13);
		/*
		* If the rounding causes a bit to spill into h_exp, it will
		* increment h_exp from zero to one and h_sig will be zero.
		* This is the correct result.
		*/
		return (uint16_t)(h_sgn + h_sig);
	}

	/* Regular case with no overflow or underflow */
	h_exp = (uint16_t)((f_exp - 0x38000000u) >> 13);
	/* Handle rounding by adding 1 to the bit beyond half precision */
	f_sig = (f & 0x007fffffu);
#if NPY_HALF_ROUND_TIES_TO_EVEN
	/*
	* If the last bit in the half significand is 0 (already even), and
	* the remaining bit pattern is 1000...0, then we do not add one
	* to the bit after the half significand.  In all other cases, we do.
	*/
	if ((f_sig & 0x00003fffu) != 0x00001000u) {
		f_sig += 0x00001000u;
	}
#else
	f_sig += 0x00001000u;
#endif
	h_sig = (uint16_t)(f_sig >> 13);
	/*
	* If the rounding causes a bit to spill into h_exp, it will
	* increment h_exp by one and h_sig will be zero.  This is the
	* correct result.  h_exp may increment to 15, at greatest, in
	* which case the result overflows to a signed inf.
	*/
#if NPY_HALF_GENERATE_OVERFLOW
	h_sig += h_exp;
	if (h_sig == 0x7c00u) {
		npy_set_floatstatus_overflow();
	}
	return h_sgn + h_sig;
#else
	return h_sgn + h_exp + h_sig;
#endif
}

float npy_half_to_float(uint16_t h) {
  union {
    float ret;
	uint32_t retbits;
  } conv;

  conv.retbits = npy_halfbits_to_floatbits(h);

  return conv.ret;
}

uint16_t npy_float_to_half(float f) {
  union {
    float f;
	uint32_t fbits;
  } conv;

  conv.f = f;
  return npy_floatbits_to_halfbits(conv.fbits);
}

class MulFP16Kernel {
 public:
  MulFP16Kernel(const MLOpKernelInfo& /*info*/) {}

  MLStatus Compute(const MLOpKernelInfo& /*info*/, const MLOpKernelContext& context) const {
    const auto X = context.GetInputTensor(0);
    const auto W = context.GetInputTensor(1);

    auto X_Data = X.GetData<MLFloat16>();
    auto W_Data = W.GetData<MLFloat16>();

    auto& shape = X.GetDimensions();
    auto Y = context.GetOutputTensor(0, shape);
    auto Y_Data = Y.GetData<MLFloat16>();

    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      Y_Data[i].val = npy_float_to_half(npy_half_to_float(X_Data[i].val) * npy_half_to_float(W_Data[i].val));
    }

    return MLStatus::OK;
  }
};

//For test purpose, we register this MulFP16Kernel kernel to Mul op.
//Once the custom schema is ready, should update this.
KernelDefBuilder MulFP16KernelDef() {
  KernelDefBuilder def("Mul16");
  def.Domain(LotusIR::kOnnxDomain)
      .SinceVersion(6)
      .Provider(LotusIR::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>());
  return def;
}

MLStatus CreateMulFP16Kernel(const IMLOpKernelInfo& kernelInfo, IMLOpKernel** opKernel) {
  return MLOpKernel<MulFP16Kernel>::CreateInstance(kernelInfo, opKernel);
}

onnx::OpSchema GetMulFP16Schema() {
	onnx::OpSchema schema("Mul16", "unknown", 0);
	schema.Input(0,
		"A",
		"First operand, should share the type with the second operand.",
		"T");
	schema.Input(
		1,
		"B",
		"Second operand. With broadcasting can be of smaller size than A. ",
		"T");
	schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
	schema.TypeConstraint(
		"T",
		OpSchema::all_numeric_types(),
		"Constrain input and output types to high-precision numeric tensors.");
	schema.SinceVersion(6);
	return schema;
}

static const std::string MUL_MODEL_URI = "testdata/mul_16.pb";

void RunSession(InferenceSession& session_object,
                RunOptions& run_options,
                std::vector<int64_t>& dims_x,
                std::vector<MLFloat16>& values_x,
                std::vector<int64_t>& dims_y,
                std::vector<MLFloat16>& values_y) {
  // prepare inputs
  MLValue ml_value;
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->GetAllocator(), dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // Now run
  Common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  EXPECT_EQ(expected_shape, rtensor.Shape());
  const std::vector<MLFloat16> found(rtensor.Data<MLFloat16>(), rtensor.Data<MLFloat16>() + expected_shape.Size());
  ASSERT_EQ(found.size(), values_y.size());
  for (size_t i = 0; i < found.size(); i++)
	  ASSERT_EQ(found[i].val, values_y[i].val);
}

TEST(Float16_Tests, Mul_16_Test) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  auto mulfp16_schema = GetMulFP16Schema();
  std::vector<OpSchema> schemas = { mulfp16_schema };
  EXPECT_TRUE(session_object.RegisterCustomOpSet(schemas, LotusIR::kOnnxDomain, 6).IsOK());

  auto def = MulFP16KernelDef();
  //Register a foo kernel which is doing Add, but bind to Mul.
  EXPECT_TRUE(session_object.RegisterCustomKernel(def, CreateMulFP16Kernel).IsOK());

  EXPECT_TRUE(session_object.Load(MUL_MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {3, 2};
  std::vector<float> values_x_32 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<MLFloat16> values_x;
  for (float i : values_x_32) {
	  MLFloat16 v;
	  v.val = npy_float_to_half(i);
	  values_x.push_back(v);
  }

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  // now the expected value should be Add's result.
  std::vector<float> expected_values_y_32 = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  std::vector<MLFloat16> expected_values_y;
  for (float i : expected_values_y_32) {
	  MLFloat16 v;
	  v.val = npy_float_to_half(i);
	  expected_values_y.push_back(v);
  }

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}
}  // namespace Test
}  // namespace Lotus
