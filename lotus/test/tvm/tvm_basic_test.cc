#include "gtest/gtest.h"
#include <tvm/tvm.h>

TEST(TVMTest, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A = placeholder({m, l}, Float(32), "A");
  Tensor B = placeholder({n, l}, Float(32), "B");
  auto C = compute({m, n}, [&](Var i, Var j) {
    return A[i][j];
  }, "C");

  Tensor::Slice x = A[n];
}
