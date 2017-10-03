include (ExternalProject)

set(models models)
set(models_URL  https://aiinfra.visualstudio.com/Lotus/_git/models)

ExternalProject_Add(models
    PREFIX ${models}
    GIT_REPOSITORY ${models_URL}
    DOWNLOAD_DIR ${DOWNLOAD_LOCATION}
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/models/models
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
