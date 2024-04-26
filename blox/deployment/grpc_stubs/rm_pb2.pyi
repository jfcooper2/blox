from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class JsonResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class RegisterRequest(_message.Message):
    __slots__ = ("ipaddr", "numGPUs", "numCPUcores", "memoryCapacity", "numaAvailable", "cpuMaping", "gpuUUIDs")
    class CpuMapingEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: int
        def __init__(self, key: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...
    IPADDR_FIELD_NUMBER: _ClassVar[int]
    NUMGPUS_FIELD_NUMBER: _ClassVar[int]
    NUMCPUCORES_FIELD_NUMBER: _ClassVar[int]
    MEMORYCAPACITY_FIELD_NUMBER: _ClassVar[int]
    NUMAAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    CPUMAPING_FIELD_NUMBER: _ClassVar[int]
    GPUUUIDS_FIELD_NUMBER: _ClassVar[int]
    ipaddr: str
    numGPUs: int
    numCPUcores: int
    memoryCapacity: int
    numaAvailable: bool
    cpuMaping: _containers.ScalarMap[int, int]
    gpuUUIDs: str
    def __init__(self, ipaddr: _Optional[str] = ..., numGPUs: _Optional[int] = ..., numCPUcores: _Optional[int] = ..., memoryCapacity: _Optional[int] = ..., numaAvailable: bool = ..., cpuMaping: _Optional[_Mapping[int, int]] = ..., gpuUUIDs: _Optional[str] = ...) -> None: ...

class BooleanResponse(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: bool
    def __init__(self, value: bool = ...) -> None: ...

class IntVal(_message.Message):
    __slots__ = ("value",)
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: int
    def __init__(self, value: _Optional[int] = ...) -> None: ...

class EmptyMsg(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
