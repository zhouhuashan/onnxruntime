// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

TEST_METHOD(test_HTK_LSTM_Truncated_Distributed) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"cntk\\test_HTK_LSTM_Truncated_Distributed");
}
TEST_METHOD(test_Office) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"cntk\\test_Office");
}
TEST_METHOD(coreml_DecisionTreeClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DecisionTreeClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_DecisionTreeClassifier_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DecisionTreeClassifier_OpenML_312_scene");
}
TEST_METHOD(coreml_DecisionTreeClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DecisionTreeClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_DecisionTreeClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DecisionTreeClassifier_sklearn_load_wine");
}
TEST_METHOD(coreml_DecisionTreeRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DecisionTreeRegressor_sklearn_load_boston");
}
TEST_METHOD(coreml_DecisionTreeRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DecisionTreeRegressor_sklearn_load_diabetes");
}
TEST_METHOD(coreml_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer-GradientBoostingRegressor_sklearn_load_boston");
}
TEST_METHOD(coreml_DictVectorizer_LinearRegression_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer-LinearRegression_sklearn_load_diabetes");
}
TEST_METHOD(coreml_DictVectorizer_LinearSVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer-LinearSVC_sklearn_load_wine");
}
TEST_METHOD(coreml_DictVectorizer_RandomForestClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer-RandomForestClassifier_sklearn_load_wine");
}
TEST_METHOD(coreml_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer-RandomForestRegressor_sklearn_load_diabetes");
}
TEST_METHOD(coreml_DictVectorizer_OpenML_31_credit) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer_OpenML_31_credit");
}
TEST_METHOD(coreml_DictVectorizer_sklearn_load_diabetes_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer_sklearn_load_diabetes_missing");
}
TEST_METHOD(coreml_DictVectorizer_sklearn_load_digits) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer_sklearn_load_digits");
}
TEST_METHOD(coreml_DictVectorizer_sklearn_load_Iris_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_DictVectorizer_sklearn_load_Iris_missing");
}
TEST_METHOD(coreml_GradientBoostingClassifier_BingClick) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingClassifier_BingClick");
}
TEST_METHOD(coreml_GradientBoostingClassifier_Criteo) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingClassifier_Criteo");
}
TEST_METHOD(coreml_GradientBoostingClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_GradientBoostingClassifier_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingClassifier_OpenML_312_scene");
}
TEST_METHOD(coreml_GradientBoostingClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_GradientBoostingClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingClassifier_sklearn_load_wine");
}
TEST_METHOD(coreml_GradientBoostingRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingRegressor_sklearn_load_boston");
}
TEST_METHOD(coreml_GradientBoostingRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_GradientBoostingRegressor_sklearn_load_diabetes");
}
TEST_METHOD(coreml_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Imputer-GradientBoostingClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Imputer-GradientBoostingClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Imputer-LogisticRegression_OpenML_1464_blood_transfusion_missing");
}
TEST_METHOD(coreml_Imputer_OpenML_1464_blood_transfusion_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Imputer_OpenML_1464_blood_transfusion_missing");
}
TEST_METHOD(coreml_Imputer_OpenML_1492_plants_Missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Imputer_OpenML_1492_plants_Missing");
}
TEST_METHOD(coreml_Imputer_sklearn_load_diabetes_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Imputer_sklearn_load_diabetes_missing");
}
TEST_METHOD(coreml_Imputer_sklearn_load_Iris_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Imputer_sklearn_load_Iris_missing");
}
TEST_METHOD(coreml_LinearRegression_FlightDelay) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearRegression_FlightDelay");
}
TEST_METHOD(coreml_LinearRegression_NYCTaxi) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearRegression_NYCTaxi");
}
TEST_METHOD(coreml_LinearRegression_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearRegression_sklearn_load_boston");
}
TEST_METHOD(coreml_LinearRegression_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearRegression_sklearn_load_diabetes");
}
TEST_METHOD(coreml_LinearSVC_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearSVC_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_LinearSVC_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearSVC_OpenML_312_scene");
}
TEST_METHOD(coreml_LinearSVC_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearSVC_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_LinearSVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearSVC_sklearn_load_wine");
}
TEST_METHOD(coreml_LinearSVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearSVR_sklearn_load_boston");
}
TEST_METHOD(coreml_LinearSVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LinearSVR_sklearn_load_diabetes");
}
TEST_METHOD(coreml_LogisticRegression_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LogisticRegression_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_LogisticRegression_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LogisticRegression_OpenML_312_scene");
}
TEST_METHOD(coreml_LogisticRegression_OpenML_31_credit) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LogisticRegression_OpenML_31_credit");
}
TEST_METHOD(coreml_LogisticRegression_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_LogisticRegression_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_Normalizer_GradientBoostingRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer-GradientBoostingRegressor_sklearn_load_boston");
}
TEST_METHOD(coreml_Normalizer_LinearRegression_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer-LinearRegression_sklearn_load_boston");
}
TEST_METHOD(coreml_Normalizer_LinearSVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer-LinearSVC_sklearn_load_wine");
}
TEST_METHOD(coreml_Normalizer_RandomForestClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer-RandomForestClassifier_sklearn_load_wine");
}
TEST_METHOD(coreml_Normalizer_RandomForestRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer-RandomForestRegressor_sklearn_load_diabetes");
}
TEST_METHOD(coreml_Normalizer_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_Normalizer_OpenML_1492_plants) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer_OpenML_1492_plants");
}
TEST_METHOD(coreml_Normalizer_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer_OpenML_312_scene");
}
TEST_METHOD(coreml_Normalizer_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_Normalizer_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer_sklearn_load_diabetes");
}
TEST_METHOD(coreml_Normalizer_sklearn_load_linnerud) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer_sklearn_load_linnerud");
}
TEST_METHOD(coreml_Normalizer_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Normalizer_sklearn_load_wine");
}
TEST_METHOD(coreml_OneHotEncoder_BikeSharing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_OneHotEncoder_BikeSharing");
}
TEST_METHOD(coreml_OneHotEncoder_OpenML_31_credit) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_OneHotEncoder_OpenML_31_credit");
}
TEST_METHOD(coreml_OneHotEncoder_OpenML_3_chess) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_OneHotEncoder_OpenML_3_chess");
}
TEST_METHOD(coreml_OneHotEncoder_sklearn_load_digits) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_OneHotEncoder_sklearn_load_digits");
}
TEST_METHOD(coreml_RandomForestClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_RandomForestClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_RandomForestClassifier_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_RandomForestClassifier_OpenML_312_scene");
}
TEST_METHOD(coreml_RandomForestClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_RandomForestClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_RandomForestClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_RandomForestClassifier_sklearn_load_wine");
}
TEST_METHOD(coreml_RandomForestRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_RandomForestRegressor_sklearn_load_boston");
}
TEST_METHOD(coreml_RandomForestRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_RandomForestRegressor_sklearn_load_diabetes");
}
TEST_METHOD(coreml_Scaler_LogisticRegression_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Scaler-LogisticRegression_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_Scaler_LogisticRegression_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Scaler-LogisticRegression_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_Scaler_SVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Scaler-SVR_sklearn_load_boston");
}
TEST_METHOD(coreml_Scaler_SVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Scaler-SVR_sklearn_load_diabetes");
}
TEST_METHOD(coreml_Scaler_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Scaler_OpenML_312_scene");
}
TEST_METHOD(coreml_Scaler_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Scaler_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_Scaler_sklearn_load_linnerud) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_Scaler_sklearn_load_linnerud");
}
TEST_METHOD(coreml_SVC_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_SVC_OpenML_1464_blood_transfusion");
}
TEST_METHOD(coreml_SVC_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_SVC_OpenML_312_scene");
}
TEST_METHOD(coreml_SVC_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_SVC_sklearn_load_breast_cancer");
}
TEST_METHOD(coreml_SVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_SVC_sklearn_load_wine");
}
TEST_METHOD(coreml_SVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_SVR_sklearn_load_boston");
}
TEST_METHOD(coreml_SVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"coreml\\coreml_SVR_sklearn_load_diabetes");
}
TEST_METHOD(coreml_AgeNet_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\coreml_AgeNet_ImageNet");
}
TEST_METHOD(coreml_FNS_Candy_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\coreml_FNS-Candy_ImageNet");
}
TEST_METHOD(coreml_Inceptionv3_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\coreml_Inceptionv3_ImageNet");
}
TEST_METHOD(coreml_Resnet50_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\coreml_Resnet50_ImageNet");
}
TEST_METHOD(coreml_SqueezeNet_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\coreml_SqueezeNet_ImageNet");
}
TEST_METHOD(coreml_VGG16_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\coreml_VGG16_ImageNet");
}
TEST_METHOD(keras2coreml_AveragePooling_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_AveragePooling_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_Conv2D_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Conv2D_ImageNet");
}
TEST_METHOD(keras2coreml_Conv2D_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Conv2D_ImageNet_large");
}
TEST_METHOD(keras2coreml_Conv2D_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Conv2D_ImageNet_small");
}
TEST_METHOD(keras2coreml_Conv2D_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Conv2D_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_Dense_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Dense_ImageNet");
}
TEST_METHOD(keras2coreml_Dense_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Dense_ImageNet_small");
}
TEST_METHOD(keras2coreml_dense_linear_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_dense_linear_load_boston");
}
TEST_METHOD(keras2coreml_dense_linear_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_dense_linear_load_diabetes");
}
TEST_METHOD(keras2coreml_Dense_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Dense_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_dense_softmax_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_dense_softmax_breast_cancer");
}
TEST_METHOD(keras2coreml_dense_softmax_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_dense_softmax_OpenML_1464_blood_transfusion");
}
TEST_METHOD(keras2coreml_dense_softmax_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_dense_softmax_OpenML_312_scene");
}
TEST_METHOD(keras2coreml_dense_softmax_OpenML_31_credit) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_dense_softmax_OpenML_31_credit");
}
TEST_METHOD(keras2coreml_elu_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_elu_ImageNet");
}
TEST_METHOD(keras2coreml_elu_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_elu_ImageNet_large");
}
TEST_METHOD(keras2coreml_elu_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_elu_ImageNet_small");
}
TEST_METHOD(keras2coreml_elu_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_elu_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_GlobalMaxPooling_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_GlobalMaxPooling_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_hard_sigmoid_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_hard_sigmoid_ImageNet");
}
TEST_METHOD(keras2coreml_hard_sigmoid_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_hard_sigmoid_ImageNet_large");
}
TEST_METHOD(keras2coreml_hard_sigmoid_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_hard_sigmoid_ImageNet_small");
}
TEST_METHOD(keras2coreml_hard_sigmoid_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_hard_sigmoid_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_leakyrelu_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_leakyrelu_ImageNet");
}
TEST_METHOD(keras2coreml_leakyrelu_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_leakyrelu_ImageNet_large");
}
TEST_METHOD(keras2coreml_leakyrelu_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_leakyrelu_ImageNet_small");
}
TEST_METHOD(keras2coreml_leakyrelu_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_leakyrelu_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_linear_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_linear_ImageNet");
}
TEST_METHOD(keras2coreml_linear_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_linear_ImageNet_large");
}
TEST_METHOD(keras2coreml_linear_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_linear_ImageNet_small");
}
TEST_METHOD(keras2coreml_linear_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_linear_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_MaxPoolingPooling_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_MaxPoolingPooling_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_MNIST) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_MNIST");
}
TEST_METHOD(keras2coreml_Normalizer_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Normalizer_ImageNet");
}
TEST_METHOD(keras2coreml_Normalizer_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Normalizer_ImageNet_large");
}
TEST_METHOD(keras2coreml_Normalizer_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Normalizer_ImageNet_small");
}
TEST_METHOD(keras2coreml_Normalizer_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Normalizer_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_Padding_Upsampling_Normalizer_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding-Upsampling-Normalizer_ImageNet");
}
TEST_METHOD(keras2coreml_Padding_Upsampling_Normalizer_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding-Upsampling-Normalizer_ImageNet_large");
}
TEST_METHOD(keras2coreml_Padding_Upsampling_Normalizer_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding-Upsampling-Normalizer_ImageNet_small");
}
TEST_METHOD(keras2coreml_Padding_Upsampling_Normalizer_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding-Upsampling-Normalizer_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_Padding_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding_ImageNet");
}
TEST_METHOD(keras2coreml_Padding_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding_ImageNet_large");
}
TEST_METHOD(keras2coreml_Padding_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding_ImageNet_small");
}
TEST_METHOD(keras2coreml_Padding_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Padding_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_Permute_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Permute_ImageNet");
}
TEST_METHOD(keras2coreml_Permute_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Permute_ImageNet_large");
}
TEST_METHOD(keras2coreml_Permute_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Permute_ImageNet_small");
}
TEST_METHOD(keras2coreml_Permute_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Permute_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_Pooling_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Pooling_ImageNet");
}
TEST_METHOD(keras2coreml_ReLU_ImageNet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_ReLU_ImageNet");
}
TEST_METHOD(keras2coreml_ReLU_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_ReLU_ImageNet_large");
}
TEST_METHOD(keras2coreml_ReLU_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_ReLU_ImageNet_small");
}
TEST_METHOD(keras2coreml_relu_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_relu_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_Reshape_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Reshape_ImageNet_large");
}
TEST_METHOD(keras2coreml_Reshape_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Reshape_ImageNet_small");
}
TEST_METHOD(keras2coreml_Reshape_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_Reshape_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_sigmoid_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_sigmoid_ImageNet_large");
}
TEST_METHOD(keras2coreml_sigmoid_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_sigmoid_ImageNet_small");
}
TEST_METHOD(keras2coreml_sigmoid_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_sigmoid_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_softplus_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_softplus_ImageNet_small");
}
TEST_METHOD(keras2coreml_softsign_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_softsign_ImageNet_large");
}
TEST_METHOD(keras2coreml_softsign_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_softsign_ImageNet_small");
}
TEST_METHOD(keras2coreml_softsign_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_softsign_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_tanh_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_tanh_ImageNet_large");
}
TEST_METHOD(keras2coreml_tanh_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_tanh_ImageNet_small");
}
TEST_METHOD(keras2coreml_tanh_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_tanh_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_thresholdedrelu_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_thresholdedrelu_ImageNet_large");
}
TEST_METHOD(keras2coreml_thresholdedrelu_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_thresholdedrelu_ImageNet_small");
}
TEST_METHOD(keras2coreml_thresholdedrelu_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_thresholdedrelu_sklearn_load_breast_cancer");
}
TEST_METHOD(keras2coreml_UpSampling_ImageNet_large) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_UpSampling_ImageNet_large");
}
TEST_METHOD(keras2coreml_UpSampling_ImageNet_small) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_UpSampling_ImageNet_small");
}
TEST_METHOD(keras2coreml_UpSampling_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"keras2coreml\\keras2coreml_UpSampling_sklearn_load_breast_cancer");
}
TEST_METHOD(test_bvlc_alexnet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_bvlc_alexnet");
}
TEST_METHOD(test_bvlc_googlenet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_bvlc_googlenet");
}
TEST_METHOD(test_bvlc_reference_caffenet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_bvlc_reference_caffenet");
}
TEST_METHOD(test_bvlc_reference_rcnn_ilsvrc13) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_bvlc_reference_rcnn_ilsvrc13");
}
TEST_METHOD(test_densenet121) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_densenet121");
}
TEST_METHOD(test_emotion_ferplus) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_emotion_ferplus");
}
TEST_METHOD(test_inception_v1) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_inception_v1");
}
TEST_METHOD(test_inception_v2) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_inception_v2");
}
TEST_METHOD(test_mnist) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_mnist");
}
TEST_METHOD(test_resnet50) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_resnet50");
}
TEST_METHOD(test_shufflenet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_shufflenet");
}
TEST_METHOD(test_squeezenet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_squeezenet");
}
TEST_METHOD(test_tiny_yolov2) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_tiny_yolov2");
}
TEST_METHOD(test_vgg19) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_vgg19");
}
TEST_METHOD(test_zfnet) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"modelzoo\\test_zfnet");
}
TEST_METHOD(keras_onnxruntime_resnet3D) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"others\\keras_onnxruntime_resnet3D");
}
TEST_METHOD(mobilenetv2_1_0) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"others\\mobilenetv2-1.0");
}
TEST_METHOD(scikit_Binarization_DecisionTreeClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization-DecisionTreeClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_Binarization_DecisionTreeClassifier_OpenML_1492_plants) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization-DecisionTreeClassifier_OpenML_1492_plants");
}
TEST_METHOD(scikit_Binarization_LinearSVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization-LinearSVR_sklearn_load_boston");
}
TEST_METHOD(scikit_Binarization_LinearSVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization-LinearSVR_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Binarization_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_Binarization_OpenML_1492_plants) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization_OpenML_1492_plants");
}
TEST_METHOD(scikit_Binarization_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Binarization_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Binarization_sklearn_load_linnerud) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization_sklearn_load_linnerud");
}
TEST_METHOD(scikit_Binarization_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Binarization_sklearn_load_wine");
}
TEST_METHOD(scikit_DecisionTreeClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DecisionTreeClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_DecisionTreeClassifier_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DecisionTreeClassifier_OpenML_312_scene");
}
TEST_METHOD(scikit_DecisionTreeClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DecisionTreeClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_DecisionTreeClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DecisionTreeClassifier_sklearn_load_wine");
}
TEST_METHOD(scikit_DecisionTreeRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DecisionTreeRegressor_sklearn_load_boston");
}
TEST_METHOD(scikit_DecisionTreeRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DecisionTreeRegressor_sklearn_load_diabetes");
}
TEST_METHOD(scikit_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer-GradientBoostingRegressor_sklearn_load_boston");
}
TEST_METHOD(scikit_DictVectorizer_LinearRegression_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer-LinearRegression_sklearn_load_diabetes");
}
TEST_METHOD(scikit_DictVectorizer_LinearSVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer-LinearSVC_sklearn_load_wine");
}
TEST_METHOD(scikit_DictVectorizer_RandomForestClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer-RandomForestClassifier_sklearn_load_wine");
}
TEST_METHOD(scikit_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer-RandomForestRegressor_sklearn_load_diabetes");
}
TEST_METHOD(scikit_DictVectorizer_OpenML_31_credit) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer_OpenML_31_credit");
}
TEST_METHOD(scikit_DictVectorizer_sklearn_load_diabetes_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer_sklearn_load_diabetes_missing");
}
TEST_METHOD(scikit_DictVectorizer_sklearn_load_digits) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer_sklearn_load_digits");
}
TEST_METHOD(scikit_DictVectorizer_sklearn_load_Iris_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_DictVectorizer_sklearn_load_Iris_missing");
}
TEST_METHOD(scikit_GradientBoostingClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_GradientBoostingClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_GradientBoostingClassifier_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_GradientBoostingClassifier_OpenML_312_scene");
}
TEST_METHOD(scikit_GradientBoostingClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_GradientBoostingClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_GradientBoostingClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_GradientBoostingClassifier_sklearn_load_wine");
}
TEST_METHOD(scikit_GradientBoostingRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_GradientBoostingRegressor_sklearn_load_boston");
}
TEST_METHOD(scikit_GradientBoostingRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_GradientBoostingRegressor_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer-GradientBoostingClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer-GradientBoostingClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer-LogisticRegression_OpenML_1464_blood_transfusion_missing");
}
TEST_METHOD(scikit_Imputer_LogisticRegression_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer-LogisticRegression_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Imputer_OpenML_1464_blood_transfusion_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer_OpenML_1464_blood_transfusion_missing");
}
TEST_METHOD(scikit_Imputer_OpenML_1492_plants_Missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer_OpenML_1492_plants_Missing");
}
TEST_METHOD(scikit_Imputer_sklearn_load_diabetes_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer_sklearn_load_diabetes_missing");
}
TEST_METHOD(scikit_Imputer_sklearn_load_Iris_missing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Imputer_sklearn_load_Iris_missing");
}
TEST_METHOD(scikit_LabelEncoder_BikeSharing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LabelEncoder_BikeSharing");
}
TEST_METHOD(scikit_LabelEncoder_OpenML_3_chess) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LabelEncoder_OpenML_3_chess");
}
TEST_METHOD(scikit_LinearRegression_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearRegression_sklearn_load_boston");
}
TEST_METHOD(scikit_LinearRegression_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearRegression_sklearn_load_diabetes");
}
TEST_METHOD(scikit_LinearSVC_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearSVC_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_LinearSVC_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearSVC_OpenML_312_scene");
}
TEST_METHOD(scikit_LinearSVC_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearSVC_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_LinearSVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearSVC_sklearn_load_wine");
}
TEST_METHOD(scikit_LinearSVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearSVR_sklearn_load_boston");
}
TEST_METHOD(scikit_LinearSVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LinearSVR_sklearn_load_diabetes");
}
TEST_METHOD(scikit_LogisticRegression_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LogisticRegression_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_LogisticRegression_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LogisticRegression_OpenML_312_scene");
}
TEST_METHOD(scikit_LogisticRegression_OpenML_31_credit) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LogisticRegression_OpenML_31_credit");
}
TEST_METHOD(scikit_LogisticRegression_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_LogisticRegression_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Normalizer_DecisionTreeRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer-DecisionTreeRegressor_sklearn_load_boston");
}
TEST_METHOD(scikit_Normalizer_GradientBoostingRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer-GradientBoostingRegressor_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Normalizer_LinearRegression_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer-LinearRegression_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Normalizer_LinearSVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer-LinearSVC_sklearn_load_wine");
}
TEST_METHOD(scikit_Normalizer_RandomForestClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer-RandomForestClassifier_sklearn_load_wine");
}
TEST_METHOD(scikit_Normalizer_RandomForestRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer-RandomForestRegressor_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Normalizer_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_Normalizer_OpenML_1492_plants) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer_OpenML_1492_plants");
}
TEST_METHOD(scikit_Normalizer_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer_OpenML_312_scene");
}
TEST_METHOD(scikit_Normalizer_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Normalizer_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Normalizer_sklearn_load_linnerud) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer_sklearn_load_linnerud");
}
TEST_METHOD(scikit_Normalizer_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Normalizer_sklearn_load_wine");
}
TEST_METHOD(scikit_Nu_SVC_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Nu_SVC_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_Nu_SVC_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Nu_SVC_OpenML_312_scene");
}
TEST_METHOD(scikit_Nu_SVC_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Nu_SVC_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Nu_SVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Nu_SVC_sklearn_load_wine");
}
TEST_METHOD(scikit_Nu_SVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Nu_SVR_sklearn_load_boston");
}
TEST_METHOD(scikit_Nu_SVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Nu_SVR_sklearn_load_diabetes");
}
TEST_METHOD(scikit_OneHotEncoder_BikeSharing) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_OneHotEncoder_BikeSharing");
}
TEST_METHOD(scikit_OneHotEncoder_OpenML_31_credit) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_OneHotEncoder_OpenML_31_credit");
}
TEST_METHOD(scikit_OneHotEncoder_OpenML_3_chess) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_OneHotEncoder_OpenML_3_chess");
}
TEST_METHOD(scikit_OneHotEncoder_sklearn_load_digits) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_OneHotEncoder_sklearn_load_digits");
}
TEST_METHOD(scikit_RandomForestClassifier_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_RandomForestClassifier_OpenML_312_scene");
}
TEST_METHOD(scikit_RandomForestClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_RandomForestClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_RandomForestClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_RandomForestClassifier_sklearn_load_wine");
}
TEST_METHOD(scikit_RandomForestRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_RandomForestRegressor_sklearn_load_boston");
}
TEST_METHOD(scikit_RandomForestRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_RandomForestRegressor_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Scalar_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_Scalar_OpenML_1492_plants) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_OpenML_1492_plants");
}
TEST_METHOD(scikit_Scalar_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_OpenML_312_scene");
}
TEST_METHOD(scikit_Scalar_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_sklearn_load_boston");
}
TEST_METHOD(scikit_Scalar_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Scalar_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_sklearn_load_diabetes");
}
TEST_METHOD(scikit_Scalar_sklearn_load_linnerud) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_sklearn_load_linnerud");
}
TEST_METHOD(scikit_Scalar_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scalar_sklearn_load_wine");
}
TEST_METHOD(scikit_Scaler_LogisticRegression_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scaler-LogisticRegression_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_Scaler_LogisticRegression_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scaler-LogisticRegression_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_Scaler_SVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scaler-SVR_sklearn_load_boston");
}
TEST_METHOD(scikit_Scaler_SVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_Scaler-SVR_sklearn_load_diabetes");
}
TEST_METHOD(scikit_SVC_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_SVC_OpenML_1464_blood_transfusion");
}
TEST_METHOD(scikit_SVC_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_SVC_OpenML_312_scene");
}
TEST_METHOD(scikit_SVC_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_SVC_sklearn_load_breast_cancer");
}
TEST_METHOD(scikit_SVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_SVC_sklearn_load_wine");
}
TEST_METHOD(scikit_SVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_SVR_sklearn_load_boston");
}
TEST_METHOD(scikit_SVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"scikit\\scikit_SVR_sklearn_load_diabetes");
}
TEST_METHOD(inception_v3) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"tensorflow\\inception_v3");
}
TEST_METHOD(libsvm_Nu_SVC_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\libsvm_Nu_SVC_OpenML_1464_blood_transfusion");
}
TEST_METHOD(libsvm_Nu_SVC_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\libsvm_Nu_SVC_OpenML_312_scene");
}
TEST_METHOD(libsvm_Nu_SVC_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\libsvm_Nu_SVC_sklearn_load_breast_cancer");
}
TEST_METHOD(libsvm_Nu_SVC_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\libsvm_Nu_SVC_sklearn_load_wine");
}
TEST_METHOD(libsvm_Nu_SVR_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\libsvm_Nu_SVR_sklearn_load_boston");
}
TEST_METHOD(libsvm_Nu_SVR_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\libsvm_Nu_SVR_sklearn_load_diabetes");
}
TEST_METHOD(XGBoost_XGClassifier_OpenML_1464_blood_transfusion) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\XGBoost_XGClassifier_OpenML_1464_blood_transfusion");
}
TEST_METHOD(XGBoost_XGClassifier_OpenML_312_scene) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\XGBoost_XGClassifier_OpenML_312_scene");
}
TEST_METHOD(XGBoost_XGClassifier_sklearn_load_breast_cancer) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\XGBoost_XGClassifier_sklearn_load_breast_cancer");
}
TEST_METHOD(XGBoost_XGClassifier_sklearn_load_wine) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\XGBoost_XGClassifier_sklearn_load_wine");
}
TEST_METHOD(XGBoost_XGRegressor_sklearn_load_boston) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\XGBoost_XGRegressor_sklearn_load_boston");
}
TEST_METHOD(XGBoost_XGRegressor_sklearn_load_diabetes) {
  SessionFactory sf({::onnxruntime::kCpuExecutionProvider}, true, true);
  run(sf, L"xgboost_and_libsvm\\XGBoost_XGRegressor_sklearn_load_diabetes");
}
