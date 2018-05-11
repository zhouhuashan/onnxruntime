#pragma once
#if defined(_DEBUG)
void *DebugHeapAlloc(size_t size, unsigned framesToSkip = 0);
void *DebugHeapReAlloc(void *p, size_t size);
void DebugHeapFree(void *p) noexcept;

#define calloc CallocNotImplemented
#define malloc DebugHeapAlloc
#define realloc DebugHeapReAlloc
#define free DebugHeapFree
#endif
