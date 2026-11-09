"""Variable-byte (7 bits per byte) integer coding with zigzag signs, as in Sim-Piece."""

class VariableByteEncoder:
    @staticmethod
    def extract7bits(i, val):
        return (val >> (7 * i)) & ((1 << 7) - 1)
    
    @staticmethod
    def extract7bitsmaskless(i, val):
        result = (val >> (7 * i)) & 0xFF  # low byte
        if result & 0x80:  # sign bit set: extend the sign
            result = result - 256
        return result
    
    @staticmethod
    def write(number, output_stream):
        if number < 0:
            number = ((-number) << 1) | 1  # zigzag: negative numbers to odd codes
        else:
            number = number << 1  # zigzag: positive numbers to even codes
        
        while number >= 128:
            output_stream.write(bytes([number & 0x7F | 0x80]))
            number >>= 7
        
        output_stream.write(bytes([number]))  # last byte, continuation bit clear
    
    @staticmethod
    def read(input_stream):
        number = 0
        shift = 0
        
        while True:
            inp = input_stream.read(1)
            if not inp:
                break
            
            inp = inp[0]
            number |= (inp & 0x7F) << shift
            shift += 7
            
            if not inp & 0x80:
                break
        
        if number & 1:
            number = -(number >> 1)  # undo the zigzag coding
        else:
            number = number >> 1
        
        return number
    
