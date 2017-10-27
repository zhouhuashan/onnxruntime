#include "core/graph/op.h"

namespace LotusIR {

    REGISTER_OPERATOR_SCHEMA(ArrayFeatureExtractor)
        .Input("X", "Data to be selected from", "T")
        .Output("Y", "Selected data as an array", "T")
        .Description(R"DOC(
            Select a subset of the data based on the indices chosen.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)", "tensor(string)" }, " allowed types.")
        .Attr("indices", "Index positions to extract the data from in the input X", AttrType::INTS);


    REGISTER_OPERATOR_SCHEMA(Binarizer)
        .Input("X", "Data to be binarized", "T")
        .Output("Y", "Binarized output data", "T")
        .Description(R"DOC(
            Makes values 1 or 0 based on a single threshold.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("threshold", "Values greater than this are set to 1, else set to 0", AttrType::FLOAT);


    REGISTER_OPERATOR_SCHEMA(CategoryMapper)
        .Input("X", "Input data", "T1")
        .Output("Y", "Output data, if strings are input, then output is INTS, and vice versa.", "T2")
        .Description(R"DOC(
            Convert strings to INTS and vice versa.
            Takes in a map to use for the conversion.
            The index position in the strings and ints repeated inputs
             is used to do the mapping.
            Each instantiated operator converts either ints to strings or strings to ints.
            This behavior is triggered based on which default value is set.
            If the string default value is set, it will convert ints to strings.
            If the int default value is set, it will convert strings to ints.
            )DOC")
        .TypeConstraint("T1", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("cats_strings", "strings part of the input map, must be same size as the ints", AttrType::STRINGS)
        .Attr("cats_int64s", "ints part of the input map, must be same size and the strings", AttrType::INTS)
        .Attr("default_string", "string value to use if the int is not in the map", AttrType::STRING)
        .Attr("default_int64", "int value to use if the string is not in the map", AttrType::INT);


    REGISTER_OPERATOR_SCHEMA(DictVectorizer)
        .Input("X", "The input dictionary", "T")
        .Output("Y", "The tensor", "tensor(int64)")
        .Description(R"DOC(
            Uses an index mapping to convert a dictionary to an array.
            The output array will be equal in length to the index mapping vector parameter.
            All keys in the input dictionary must be present in the index mapping vector.
            For each item in the input dictionary, insert its value in the ouput array.
            The position of the insertion is determined by the position of the item's key
            in the index mapping. Any keys not present in the input dictionary, will be
            zero in the output array.  Use either string_vocabulary or int64_vocabulary, not both.
            For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
            then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
            )DOC")
        .TypeConstraint("T", { "map(string, int64)", "map(int64, string)"}, " allowed types.")
        .Attr("string_vocabulary", "The vocabulary vector of strings", AttrType::STRINGS)
        .Attr("int64_vocabulary", "The vocabulary vector of int64s", AttrType::INTS);


    REGISTER_OPERATOR_SCHEMA(Imputer)
        .Input("X", "Data to be imputed", "T")
        .Output("Y", "Imputed output data", "T")
        .Description(R"DOC(
            Replace imputs that equal replaceValue/s  with  imputeValue/s.
            All other inputs are copied to the output unchanged.
            This op is used to replace missing values where we know what a missing value looks like.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("imputed_value_floats", "value(s) to change to, can be length 1 or length F if using int type", AttrType::FLOATS)
        .Attr("replaced_value_float", "value that needs replacing if using int type", AttrType::FLOAT)
        .Attr("imputed_value_int64s", "value(s) to change to, can be length 1 or length F if using int type", AttrType::INTS)
        .Attr("replaced_value_int64", "value that needs replacing if using int type", AttrType::INT);


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
    

    REGISTER_OPERATOR_SCHEMA(LabelEncoder)
        .Input("X", "Data to be encoded", "T1")
        .Output("Y", "Encoded output data", "T2")
        .Description(R"DOC(
            Convert class label to their integral type and vice versa.
            In both cases the operator is instantiated with the list of class strings.
            The integral value of the string is the index position in the list.
            )DOC")
        .TypeConstraint("T1", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("classes_strings", "List of class label strings to be encoded as INTS", AttrType::STRINGS)
        .Attr("default_int64", "Default value if not in class list as int64", AttrType::INT)
        .Attr("default_string", "Default value if not in class list as string", AttrType::STRING);


    REGISTER_OPERATOR_SCHEMA(LinearClassifier)
        .Input("X", "Data to be classified", "T1")
        .Output("Y", "Classification outputs (one class per example", "T2")
        .Output("Z", "Classification outputs (All classes scores per example,N,E", "tensor(float)")
        .Description(R"DOC(
            Linear classifier prediction (choose class)
            )DOC")
        .TypeConstraint("T1", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("coefficients", "weights of the model(s)", AttrType::FLOATS)
        .Attr("intercepts", "weights of the intercepts (if used)", AttrType::FLOATS)
        .Attr("post_transform", "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT", AttrType::INT)
        .Attr("multi_class", "whether to do OvR or multinomial (0=OvR and is default)", AttrType::INT)
        .Attr("classlabels_strings", "class labels if using string labels, size E", AttrType::STRINGS)
        .Attr("classlabels_int64s", "class labels if using int labels, size E", AttrType::INTS);


    REGISTER_OPERATOR_SCHEMA(LinearRegression)
        .Input("X", "Data to be regressed", "T")
        .Output("Y", "Regression outputs (one per target, per example", "tensor(float)")
        .Description(R"DOC(
            Generalized linear regression evaluation.
            If targets is set to 1 (default) then univariate regression is performed.
            If targets is set to M then M sets of coefficients must be passed in as a sequence
            and M results will be output for each input n in N.
            Coefficients are of the same length as an n, and coefficents for each target are contiguous.
           "Intercepts are optional but if provided must match the number of targets.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("coefficients", "weights of the model(s)", AttrType::FLOATS)
        .Attr("intercepts", "weights of the intercepts (if used)", AttrType::FLOATS)
        .Attr("targets", "total number of regression targets (default is 1)", AttrType::INT)
        .Attr("post_transform", "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT", AttrType::INT);


    REGISTER_OPERATOR_SCHEMA(Normalizer)
        .Input("X", "Data to be encoded","T")
        .Output("Y", "encoded output data", "tensor(float)")
        .Description(R"DOC(
            Normalize the input.  There are three normalization modes,
            which have the corresponding formulas:
            Max .. math::     max(x_i)
            L1  .. math::  z = ||x||_1 = \sum_{i=1}^{n} |x_i|
            L2  .. math::  z = ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("norm", "0=Lmax, 1=L1, 2=L2", AttrType::INT);


    REGISTER_OPERATOR_SCHEMA(OneHotEncoder)
        .Input("X", "Data to be encoded", "T")
        .Output("Y", "encoded output data", "tensor(float)")
        .Description(R"DOC(
            Replace the inputs with an array of ones and zeros, where the only
            one is the zero-based category that was passed in.  The total category count 
            will determine the length of the vector. For example if we pass a 
            tensor with a single value of 4, and a category count of 8, the 
            output will be a tensor with 0,0,0,0,1,0,0,0 .
            This operator assumes every input in X is of the same category set 
            (meaning there is only one category count).
            )DOC")
        .TypeConstraint("T", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("cats_int64s", "list of cateogries, ints", AttrType::INTS)
        .Attr("cats_strings", "list of cateogries, strings", AttrType::STRINGS)
        .Attr("zeros", "if true and category is not present, will return all zeros, if false and missing category, operator will return false", AttrType::INT);


    // Input: X, output: Y
    REGISTER_OPERATOR_SCHEMA(Scaler)
        .Input("X", "Data to be scaled", "T")
        .Output("Y", "Scaled output data", "tensor(float)")
        .Description(R"DOC(
            Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("scale", "second, multiply by this", AttrType::FLOAT)
        .Attr("offset", "first, offset by this", AttrType::FLOAT);

    REGISTER_OPERATOR_SCHEMA(SVMClassifier)
        .Input("X", "Data to be classified", "T1")
        .Output("Y", "Classification outputs (one class per example", "T2")
        .Output("Z", "Classification outputs (All classes scores per example,N,E", "tensor(float)")
        .Description(R"DOC(
            SVM classifier prediction 
            )DOC")
        .TypeConstraint("T1", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("kernel_type", "enum LINEAR, POLY, RBF, SIGMOID, defaults to linear",AttrType::INT)
        .Attr("kernel_params", "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.", AttrType::FLOATS)
        .Attr("prob_A", "", AttrType::FLOATS)
        .Attr("prob_B", "", AttrType::FLOATS)
        .Attr("vectors_per_class", "", AttrType::INTS)
        .Attr("support_vectors", "", AttrType::FLOATS)
        .Attr("coefficients", "", AttrType::FLOATS)
        .Attr("rho", "", AttrType::FLOATS)
        .Attr("post_transform", "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT", AttrType::INT)
        .Attr("classlabels_strings", "class labels if using string labels", AttrType::STRINGS)
        .Attr("classlabels_int64s", "class labels if using int labels", AttrType::INTS);


    REGISTER_OPERATOR_SCHEMA(SVMRegressor)
        .Input("X", "Input N,F", "T")
        .Output("Y", "All target scores, N,E", "tensor(float)")
        .Description(R"DOC(
            SVM classifier prediction 
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("kernel_type", "enum LINEAR, POLY, RBF, SIGMOID, defaults to linear", AttrType::INT)
        .Attr("kernel_params", "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.", AttrType::FLOATS)
        .Attr("post_transform", "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT", AttrType::INT)
        .Attr("vectors_per_class", "", AttrType::INTS)
        .Attr("support_vectors", "", AttrType::FLOATS)
        .Attr("coefficients", "", AttrType::FLOATS)
        .Attr("prob_a", "", AttrType::FLOATS)
        .Attr("prob_b", "", AttrType::FLOATS)
        .Attr("rho", "", AttrType::FLOATS)
        .Attr("n_targets", "number of targets for regressions", AttrType::INT);

    REGISTER_OPERATOR_SCHEMA(TreeClassifier)
        .Input("X", "Data to be classified", "T1")
        .Output("Y", "Classification outputs (one class per example", "T2")
        .Output("Z", "Classification outputs (All classes scores per example,N,E", "tensor(float)")
        .Description(R"DOC(
            Tree Ensemble classifier.  Returns the top class for each input in N.
            All args with nodes_ are fields of a tuple of tree nodes, and 
            it is assumed they are the same length, and an index i will decode the
            tuple across these inputs.  Each node id can appear only once 
            for each tree id."
            All fields prefixed with class_ are tuples of votes at the leaves.
            A leaf may have multiple votes, where each vote is weighted by
            the associated class_weights index.  
            It is expected that either classlabels_strings or classlabels_INTS
            will be passed and the class_ids are an index into this list.
            Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF.
            )DOC")
        .TypeConstraint("T1", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .TypeConstraint("T2", { "tensor(string)", "tensor(int64)" }, " allowed types.")
        .Attr("nodes_treeids", "tree id for this node", AttrType::INTS)
        .Attr("nodes_nodeids", "node id for this node, node ids may restart at zero for each tree (but not required).", AttrType::INTS)
        .Attr("nodes_featureids", "feature id for this node", AttrType::INTS)
        .Attr("nodes_values", "thresholds to do the splitting on for this node.", AttrType::FLOATS)
        .Attr("nodes_hitrates", "", AttrType::FLOATS)
        .Attr("nodes_modes", "enum of behavior for this node", AttrType::INTS)
        .Attr("nodes_truenodeids", "child node if expression is true", AttrType::INTS)
        .Attr("nodes_falsenodeids", "child node if expression is false", AttrType::INTS)
        .Attr("nodes_missing_value_tracks_true", "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes", AttrType::INTS)
        .Attr("class_treeids", "tree that this node is in", AttrType::INTS)
        .Attr("class_nodeids", "node id that this weight is for", AttrType::INTS)
        .Attr("class_ids", "index of the class list that this weight is for", AttrType::INTS)
        .Attr("class_weights", "the weight for the class in class_id", AttrType::FLOATS)
        .Attr("post_transform", "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT", AttrType::INT)
        .Attr("classlabels_strings", "class labels if using string labels, size E", AttrType::STRINGS)
        .Attr("classlabels_int64s", "class labels if using int labels, size E", AttrType::INTS);


    REGISTER_OPERATOR_SCHEMA(TreeRegressor)
        .Input("X", "Input N,F", "T")
        .Output("Y", "NxE floats", "tensor(float)")
        .Description(R"DOC(
            Tree Ensemble regressor.  Returns the regressed values for each input in N.
            All args with nodes_ are fields of a tuple of tree nodes, and 
            it is assumed they are the same length, and an index i will decode the
            tuple across these inputs.  Each node id can appear only once 
            for each tree id.
            All fields prefixed with target_ are tuples of votes at the leaves.
            A leaf may have multiple votes, where each vote is weighted by
            the associated target_weights index.  
            All trees must have their node ids start at 0 and increment by 1.
            Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
            )DOC")
        .TypeConstraint("T", { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" }, " allowed types.")
        .Attr("nodes_treeids", "tree id for this node", AttrType::INTS)
        .Attr("nodes_nodeids", "node id for this node, node ids may restart at zero for each tree (but not required).", AttrType::INTS)
        .Attr("nodes_featureids", "feature id for this node", AttrType::INTS)
        .Attr("nodes_values", "thresholds to do the splitting on for this node.", AttrType::FLOATS)
        .Attr("nodes_hitrates", "", AttrType::FLOATS)
        .Attr("nodes_modes", "enum of behavior for this node", AttrType::INTS)
        .Attr("nodes_truenodeids", "child node if expression is true", AttrType::INTS)
        .Attr("nodes_falsenodeids", "child node if expression is false", AttrType::INTS)
        .Attr("nodes_missing_value_tracks_true", "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes", AttrType::INTS)
        .Attr("target_treeids", "tree that this node is in", AttrType::INTS)
        .Attr("target_nodeids", "node id that this weight is for", AttrType::INTS)
        .Attr("target_ids", "index of the class list that this weight is for", AttrType::INTS)
        .Attr("target_weights", "the weight for the class in target_id", AttrType::FLOATS)
        .Attr("n_targets", "number of regression targets", AttrType::INT)
        .Attr("post_transform", "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT", AttrType::INT)
        .Attr("aggregate_function", "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZER", AttrType::INT)
        .Attr("base_values", "base values for regression, added to final score, size must be the same as n_outputs or can be left unassigned (assumed 0)", AttrType::FLOATS);



}
