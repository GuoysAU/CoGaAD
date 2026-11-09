class Point:
    def __init__(self, timestamp, value):
        self._timestamp = timestamp
        self._value = value

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def value(self):
        return self._value
    
