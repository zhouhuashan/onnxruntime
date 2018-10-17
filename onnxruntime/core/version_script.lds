#_init and _fini should be local
VERS_1.0 {
  # Export symbols in c_api.h.
  global:
    ONNXRuntime*;
    ReleaseONNX*;

  # Hide everything else.
  local:
    *;
};
