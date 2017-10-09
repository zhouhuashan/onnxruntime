include (ExternalProject)

set(models models)
set(models_URL  https://aiinfra.visualstudio.com/Lotus/_git/models)
set(models_dir ${CMAKE_CURRENT_BINARY_DIR}/models/models)

ExternalProject_Add(models
    PREFIX ${models}
    GIT_REPOSITORY ${models_URL}
    DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/models
    SOURCE_DIR ${models_dir}
    DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E remove_directory ${models_dir} && git clone --depth 1 ${models_URL}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
