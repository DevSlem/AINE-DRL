from typing import Dict, Any, List

class StaticRecursiveBuffer:
    """
    It allocates multiple buffers whose memories have fixed capacity. 
    It is recurisve structure which means index where an item is pushed is recurisve. 
    Specifically, you can add a key-value pair dictionary and each value is added to each buffer of the key.
    """
    def __init__(self, keys: tuple[str, ...], capacity: int) -> None:
        self._keys = keys
        self._capacity = capacity
        self.reset()
        
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def latest_idx(self) -> int:
        return self._latest_idx
    
    @property
    def is_full(self) -> bool:
        return self._count == self._capacity
    
    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys
        
    @property
    def latest_item(self) -> Dict[str, Any]:
        return {key: buffer[self._latest_idx] for key, buffer in self._buffer.items()}
    
    @property
    def buffer_dict(self) -> Dict[str, List[Any]]:
        return {key: buffer[:self._count] for key, buffer in self._buffer.items()}
        
    def reset(self):
        self._count = 0
        self._latest_idx = -1
        
        self._buffer: Dict[str, List[Any]] = {key: [None] * self._capacity for key in self._keys}
            
    def add(self, key_value_pair: dict):
        self._latest_idx = (self._latest_idx + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)
        
        for key, value in key_value_pair.items():
            self._buffer[key][self._latest_idx] = value
