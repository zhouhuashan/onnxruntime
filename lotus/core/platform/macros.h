#ifndef LOTUS_PLATFORM_MACROS_H_
#define LOTUS_PLATFORM_MACROS_H_

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define LOTUS_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;         \
    TypeName &operator=(const TypeName &) = delete

// A macro to disallow move construction and assignment
// This is usually placed in the private: declarations for a class.
#define LOTUS_DISALLOW_MOVE(TypeName)     \
    TypeName(const TypeName &&) = delete; \
    TypeName &operator=(const TypeName &&) = delete

// A macro to disallow the copy constructor, operator= functions, move construction and move assignment
// This is usually placed in the private: declarations for a class.
#define LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TypeName) \
    LOTUS_DISALLOW_COPY_AND_ASSIGN(TypeName);         \
    LOTUS_DISALLOW_MOVE(TypeName)

#endif  // LOTUS_PLATFORM_MACROS_H_
