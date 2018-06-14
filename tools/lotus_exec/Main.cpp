#include<iostream>
#include "Model.h"

void print_cmd_option()
{
    std::cerr << "lotusrt_exec.exe -m model_file [-t testdata]" << std::endl;
}

int main(int argc, char *args[])
{
    CmdParser parser(argc, args);
    std::string* modelfile = parser.GetCommandArg((char*)"-m");
    if (!modelfile)
    {
        std::cerr << "WinML model file is required." << std::endl;
        print_cmd_option();
        return -1;
    }

    Model model(*modelfile);

    if (model.GetStatus() == ExecutionStatus::OK)
    {
        std::cerr << "Done loading model: " << modelfile->c_str() << std::endl;
        std::string* testfile = parser.GetCommandArg((char*)"-t");
        if (testfile)
        {
            model.Execute(*testfile);
        }
    }

    std::cerr << "Execution Status: " << model.GetStatusString() << std::endl;
}
