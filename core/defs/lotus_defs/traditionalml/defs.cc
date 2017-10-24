#include "core/graph/op.h"

namespace LotusIR {
    REGISTER_OPERATOR_SCHEMA(FeatureVectorizer)
        .Description(R"DOC(
            Concatenates input features into one continuous output.  
            Inputlist is a list of input feature names, inputdimensions is the size of each input feature.
            Only supports a single batch at this time.
            Inputs can be in any order, but will be written to the output in the order of inputlist.
            Any missing input is set to zero in the output array.
            Allowable inputs are scalers, sequences, tensors, and maps, where the map key is an int64.
            For maps they will be written to the output in ascending key order.
            )DOC")
        .Attr("inputlist", "list of string names of the input features, output features will appear in this order", AttrType::STRINGS)
        .Attr("inputdimensions", "the size of the inputs in the input list (useful for passing sparse dicts at runtime", AttrType::INTS)
        .Input("featureArgs", "feature name to value args, which are variadic arguments.", "T")
        .Output("featurevector", "flattened feature vectors.", "tensor(double)")
        .TypeConstraint("T",
        {
        "record(name:string, value:float)", "record(name:string, value:int64)", "record(name:string, value:int32)", "record(name:string, value:double)",
        "record(name:string, value:seq(float))", "record(name:string, value:seq(int64))", "record(name:string, value:seq(double))",
        "record(name:string, value:map(int64, float))", "record(name:string, value:map(int64, int64))",
        "record(name:string, value:tensor(float))", "record(name:string, value:tensor(int64))", "record(name:string, value:tensor(double))", "record(name:string, value:tensor(int32))"
        },
            "allowed feature value types.");
}
