import struct

class UIntEncoder:
    @staticmethod
    def write(number, output_stream):
        if number > (2 ** 32) - 1 or number < 0:
            raise ValueError(f"Can't save number {number} as unsigned int")
        int_bytes = struct.pack('>I', number)
        output_stream.write(int_bytes)

    @staticmethod
    def read(input_stream):
        int_bytes = input_stream.read(4)
        if len(int_bytes) != 4:
            raise ValueError("Invalid byte length")
        number = struct.unpack('>I', int_bytes)[0]
        return number
