include (ExternalProject)

set(eigen_URL "https://ossmsft.visualstudio.com/DefaultCollection/Project15/_git/Archive_Eigen")
set(eigen_TAG "Microsoft/3.3.4")
#set(eigen_BUILD ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen)
#set(eigen_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/eigen/install)

if (lotus_USE_PREINSTALLED_EIGEN)
    set(eigen_INCLUDE_DIRS ${eigen_SOURCE_PATH})
    ExternalProject_Add(eigen
        PREFIX eigen
        GIT_REPOSITORY ${eigen_URL}
        GIT_TAG ${eigen_TAG}
        SOURCE_DIR ${eigen_SOURCE_PATH}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
    )
else ()
    set(eigen_INCLUDE_DIRS
        ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen/eigen-eigen-5a0156e40feb
    )
    ExternalProject_Add(eigen
        PREFIX eigen
        GIT_REPOSITORY ${eigen_URL}
        GIT_TAG ${eigen_TAG}
        CONFIGURE_COMMAND cmake -E echo "Skipping configure step."
        BUILD_COMMAND cmake -E echo "Skipping build step."
        INSTALL_COMMAND cmake -E echo "Skipping install step."
        #DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    )
endif()
