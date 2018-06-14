//
// Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once
#include <map>

class CmdParser
{
public:
    CmdParser(int argc, char* argsv[]) {
        if (argc > 2)
        {
            for (int i = 1; i < argc; i += 2)
            {
                cmdMap[argsv[i]] = argsv[i + 1];
            }
        }
    }
    ~CmdParser() {

    }
    std::string* GetCommandArg(char* option)
    {
        if (cmdMap.count(option))
        {
            return &cmdMap[option];
        }
        return NULL;
    }
private:
    std::map<std::string, std::string> cmdMap;
};
