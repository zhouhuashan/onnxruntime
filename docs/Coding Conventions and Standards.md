# Lotus team coding conventions and standards


## Code Style

Google style from https://google.github.io/styleguide/cppguide.html with 3 minor alterations:

*  Max line length 120
  *	Aim for 80, but up to 120 is fine.
*	Exceptions
  *	Allowed to throw fatal errors that are expected to result in a top level handler catching them, logging them and terminating the program.
*	Non-const references
  *	Allowed
  *	However const correctness and usage of smart pointers (shared_ptr and unique_ptr) is expected, so a non-const reference equates to “this is a non-null object that you can change but are not being given ownership of”.

#### Clang-format
Clang-format will handle automatically formatting code to these rules, and there’s a Visual Studio plugin that can format on save at https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.ClangFormat. 

There is a .clang-format file in the Lotus vNext repository in the 'lotus' directory that has the max line length override and defaults to the google rules. With the plugin installed and format on save enabled, whenever you save a file it will update the formatting for you. 

See [Setting up clang-format Visual Studio plugin
](https://microsoft.sharepoint.com/teams/lotusdnnteam/_layouts/OneNote.aspx?id=%2Fteams%2Flotusdnnteam%2FSiteAssets%2FLotus%20DNN%20Team%20Notebook&wd=target%28Development.one%7C63D3AB47-51D1-4A62-9965-66882234BD44%2FSetting%20up%20clang-format%20Visual%20Studio%20plugin%7C096EC1F4-5162-4C4C-A055-44D782A06EC9%2F%29) in the Lotus DNN OneNote.

Hopefully there won’t be any issues that can’t be worked around so that automatic formatting can be applied to all edited files before creating a review. 

Personally I find the layout when formatted to that style way too busy and hard to read (the equivalent of being shouted at by six people at once – I need more whitespace and indentation), so am planning on using a different .clang-format file when coding, and running clang-format prior to creating a PR to reset the formatting using the team rules. Of course that requires automatic formatting to always work to be a viable approach.

See /lotus/ReformatSource.ps1 for a script that will run clang-format on all the source files beneath that directory.


## Code analysis

I looked at clang-tidy and the Visual Studio Code Analysis as options. Ideally we’re checking against the C++ Core guidelines as well to catch usage errors and bad patterns: See https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md for what those involve. 

Using Visual Studio Code Analysis is proposed, at least initially, due to issues getting clang-tidy to run.

Clang-tidy

* Supports checking for google style rules and a range of other things
  * http://clang.llvm.org/extra/clang-tidy/checks/list.html
* I wasn’t able to generate the necessary input for it to run from the command line on x64 Windows
  *  Needs a compile_commands.json file, but I wasn’t able to get CMake to generate one
    * There may be a way and it could just be my lack of CMake knowledge
  * Was able to run it via a plugin in Visual Studio though
    * https://marketplace.visualstudio.com/items?itemName=vs-publisher-690586.ClangPowerTools 
* Unfortunately some of the Windows header files that we use break clang parsing, and running clang-tidy results in errors. 
  * It still provides valid warnings but the errors make it hard to use in an automated fashion. 
  * We could hack something to replace that header for a clang-tidy build
  * Bugs with this issue date back almost a year, so not expecting a fix any time soon
* It’s pretty slow to run as it appears to startup a new compiler instance for every file.
  * That may only affect the VS plugin
* Would be an entirely new external component to setup in the automated build system if we could get the compile_commands.json file generated
* C++ Core rules don’t appear to be as comprehensive as the Visual Studio ones
  * 17 categories vs over 50 checks in VS, however some of those categories may cover multipe

Visual Studio Code Analysis

* We already have infrastructure on LotusRT for running Visual Studio Code Analysis and ignoring warnings from external files, so can leverage that 
* I have a change to the CMake setup to enable C++ Core code analysis rules on specific projects. 
  * Would disable running Code Analysis on build in VS so they don’t slow development, but they can be manually run on a specific project via Visual Studio as needed
* Updating code to conform to these rules will be a little onerous initially as you get familiar with what is required, however such checks should result in more correct and robust code. 
  * If certain rules are causing unnecessary grief we can discuss and consider disabling them. 
* I will drive getting the vNext code to be code analysis warning free so that we can change the static analysis build to fail if new warnings appear
  * Unless some build fails, nobody will notice or address the warnings

## Unit Testing and Code Coverage

There should be unit tests that cover the core functionality of the product, expected edge cases, and expected errors. 
Code coverage from these tests should aim for 80%. 

Generally there isn't a need to write class level tests for every class in a strict ‘unit’ testing way. Most classes get tested indirectly by testing the core functionality, and thinking about tests in a BDD way (test a specific behaviour rather than a class) may help pick the right level component to test.

Code coverage needs to be correctly setup in the new Visual Studio solution. Being able to see what lines are covered and which ones aren’t is a great way to know if changes you made are being tested, so it’s important to have this infrastructure. 
