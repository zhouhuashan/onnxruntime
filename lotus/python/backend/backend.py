"""
Implements ONNX's backend API.
"""
from onnx.checker import check_model
from onnx.backend.base import Backend
from onnxruntime import InferenceSession, RunOptions, SessionOptions
from onnxruntime.backend.backend_rep import OnnxRuntimeBackendRep


class OnnxRuntimeBackend(Backend):
    """
    Implements ONNX's backend API with
    *onnxruntime*.
    """
    
    @classmethod
    def is_compatible(cls,
                      model,  # type: ModelProto
                      device='CPU',  # type: Text
                      **kwargs  # type: Any
                      ):  # type: (...) -> bool
        """
        Return whether the model is compatible with the backend.
        """
        return cls.supports_device(device)

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        """
        Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return device == 'CPU'

    @classmethod
    def prepare(cls,
                model,
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[BackendRep]
        if isinstance(model, OnnxRuntimeBackendRep):
            return model
        elif isinstance(model, InferenceSession):
            return OnnxRuntimeBackendRep(model)
        elif isinstance(model, (str, bytes)):
            options = SessionOptions()
            for k, v in kwargs.items():
                if hasattr(options, k):
                    setattr(options, k, v)
            inf = InferenceSession(model, options)
            if inf.device != device:
                raise RuntimeError("Incompatible device expected '{0}', got '{1}'".format(device, inf.device))
            return cls.prepare(inf, device, **kwargs)
        else:
            # type: ModelProto
            check_model(model)
            bin = model.SerializeToString()
            return cls.prepare(bin, device, **kwargs)

    @classmethod
    def run_model(cls,
                  model,  # type: ModelProto
                  inputs,  # type: Any
                  device='CPU',  # type: Text
                  **kwargs  # type: Any
                  ):  # type: (...) -> Tuple[Any, ...]
        rep = cls.prepare(model, device, **kwargs)
        options = RunOptions()
        for k, v in kwargs.items():
            if hasattr(options, k):
                setattr(options, k, v)
        return rep.run(inputs, options)

    @classmethod
    def run_node(cls,
                 node,  # type: NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Dict[Text, Any]
                 ):  # type: (...) -> Optional[Tuple[Any, ...]]
        '''
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        '''
        raise NotImplementedError("It is much more efficient to run a whole model than every node independently.")


is_compatible = OnnxRuntimeBackend.is_compatible
prepare = OnnxRuntimeBackend.prepare
run = OnnxRuntimeBackend.run_model
supports_device = OnnxRuntimeBackend.supports_device
