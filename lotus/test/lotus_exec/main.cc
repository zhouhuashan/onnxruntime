#include <iostream>

#include "CmdParser.h"
#include "Model.h"

void print_cmd_option() {
  std::cerr << "lotusrt_exec.exe -m model_file [-t testdata]" << std::endl;
}

int main(int argc, const char* args[]) {
  CmdParser parser(argc, args);
  const std::string* modelfile = parser.GetCommandArg("-m");
  if (!modelfile) {
    std::cerr << "WinML model file is required." << std::endl;
    print_cmd_option();
    return -1;
  }

  Model model(*modelfile);

  if (model.GetStatus() == ExecutionStatus::OK) {
    std::cerr << "Done loading model: " << modelfile->c_str() << std::endl;
    const std::string* testfile = parser.GetCommandArg("-t");
    if (testfile) {
      model.Execute(*testfile);
    }
  }

  std::cerr << "Execution Status: " << model.GetStatusString() << std::endl;
}
