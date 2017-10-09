#!/usr/bin/env python3

# Wraps a protobuf generated header file with some preprocessor directives to
# locally suppress certain warnings. Modifies the given file.

import os
import sys

PREFIX = """
// disable warnings from generated proto headers
#ifdef _MSC_VER
#pragma warning(push)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable:4800)
#endif // _MSC_VER
// ----- BEGIN WRAPPED CODE -----
"""

SUFFIX = """
// ----- END WRAPPED CODE -----
// re-enable warnings
#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER
"""

if len(sys.argv) < 2:
    print("usage: <proto header to wrap (e.g. x.pb.h)>", file=sys.stderr)
    sys.exit(1)

pb_header_file = sys.argv[1]
wrapped_pb_header_file = pb_header_file + ".wrapped"
with open(pb_header_file, "r") as pb_header_content, \
     open(wrapped_pb_header_file, "w") as wrapped_pb_header_content:
    wrapped_pb_header_content.write(PREFIX)
    for line in pb_header_content: wrapped_pb_header_content.write(line)
    wrapped_pb_header_content.write(SUFFIX)

os.replace(wrapped_pb_header_file, pb_header_file)

sys.exit(0)
