This is used to clarify some engineering related topics.

1. Coding guideline
We're following Bing c++ coding guideline as https://www.bingwiki.com/Coding_Guidelines.
NOTE: Please disregard sections about exception handling. Exceptions should NOT be thrown intentionally。

2. Engineering process
2.1 Coding
Please do read thru and follow it before you make any code changes in LotusIR.
2.2 Adding Unit Test
For any functional codes, especially bug fixing, please do add unit test to ensure there's no regression in future. Unit test should be added in directory \test.
2.3 Build & Running Unit Test locally
Please kick off full build locally and run unit test by running LotusIR_UT.exe in directory .\cmake\build\debug\LotusIR_UT.exe. When running Unit Test, the working
directory should be same as the directory LotusIR_UT.exe locates.
The unit test can also be run with ctest - navigate to the build directory and then run "ctest -C <config (e.g. Debug)>".
2.4 Creating Pull Request for code review
After getting a successful local build and no Unit Test failure, go to https://aiinfra.visualstudio.com/530acbc4-21bc-487d-8cd8-348ff451d2ff/_git/LotusIR/pullrequests?_a=mine
to create a PR merging your branch to master. LotusIR team will be added as reviewer by default. You may want to add specific reviewers with their alias, which will
make the review process faster (since a notification will be sent to the specific reviewer).
2.5 Complete Pull Request
After at least one LotusIR team member approves your changes and having CI build passed, you may go to complete your pull request and merge your codes into master.
