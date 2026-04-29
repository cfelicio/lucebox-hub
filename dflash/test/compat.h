// Cross-platform shim for POSIX setenv/unsetenv.
// MSVC provides _putenv_s instead.
#ifndef DFLASH_TEST_COMPAT_H
#define DFLASH_TEST_COMPAT_H

#ifdef _WIN32
#include <cstdlib>
static inline int setenv(const char *name, const char *value, int /*overwrite*/) {
    return _putenv_s(name, value);
}
static inline int unsetenv(const char *name) {
    return _putenv_s(name, "");
}
#endif

#endif // DFLASH_TEST_COMPAT_H
