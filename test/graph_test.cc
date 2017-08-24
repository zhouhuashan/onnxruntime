#include <iostream>
#include "gtest/gtest.h"
#include "graph.h"

namespace CommonIR
{
    namespace Test
    {
        TEST(GraphTest, GraphConstruction)
        {
            Graph graph("graph_1", 1);

            EXPECT_EQ(1, graph.Version());
            EXPECT_EQ("graph_1", graph.Name());
        }
    }
}