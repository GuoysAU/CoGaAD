import math

class SimPieceSegment:
    def __init__(self, init_timestamp, a_min, a_max, b):
        self.init_timestamp = init_timestamp
        self.a_min = a_min
        self.a_max = a_max
        self.a = (a_min + a_max) / 2
        self.b = b
        self._a_cached = False

    def round_to_least_decimal(self, aMin, aMax):
        a = (aMin+aMax)/2
        # Determine the number of decimal places in aMin and aMax
        decimal_places_aMin = len(str(aMin).split('.')[1].rstrip('0')) - len(str(aMin).split('.')[1].lstrip('0'))
        decimal_places_aMax = len(str(aMax).split('.')[1].rstrip('0')) - len(str(aMax).split('.')[1].lstrip('0'))

        # Find the minimum decimal places to round 'a'
        min_decimal_places = max(decimal_places_aMin, decimal_places_aMax)+2

        # Round 'a' to the minimum decimal places
        rounded_a = round(a, min_decimal_places)
        return rounded_a
    
    def count_matching_digits(self, num1, num2):
        # Number of leading decimal digits the two slopes share (behavior of the original, quirks included).
        if (math.isinf(num1) or math.isinf(num1)):
            return -1

        s1 = f"{num1:.16f}"
        s2 = f"{num2:.16f}"

        dot1 = s1.find(".")
        dot2 = s2.find(".")
        frac1 = s1[dot1 + 1 :].rstrip("0") if dot1 != -1 else ""
        frac2 = s2[dot2 + 1 :].rstrip("0") if dot2 != -1 else ""

        count = 0
        for digit1, digit2 in zip(frac1, frac2):
            if digit1 != digit2:
                break
            count += 1
        return count

    def truncate_to_n_decimal_places(self, number, n):
        scaled_number = number * (10 ** n)
        truncated_number = int(scaled_number) / (10 ** n)  # drop the digits beyond n decimals
        return truncated_number

    
    # Getters of the Java original, kept as properties.
    @property
    def get_init_timestamp(self):
        return self.init_timestamp

    @property
    def get_a_min(self):
        return self.a_min

    @property
    def get_a_max(self):
        return self.a_max

    @property
    def get_a(self):
        if getattr(self, "_a_cached", False):
            return self.a
        if((self.a_max * self.a_min<0)):
            self.a =  0
            self._a_cached = True
            return self.a
        precision = self.count_matching_digits(self.a_max, self.a_min)
        if(precision==-1):
            self.a =  0
            self._a_cached = True
            return self.a
        if(precision<2):precision =2
        a =  self.truncate_to_n_decimal_places((self.a_max+self.a_min)/2, precision+2 )

        if (not(a>=self.a_min and a<=self.a_max)):
            a = self.round_to_least_decimal(self.a_max, self.a_min)

        if (not(a>=self.a_min and a<=self.a_max)):
            a =  self.truncate_to_n_decimal_places((self.a_max+self.a_min)/2, precision+3 )

        self.a = a
        if (not(a>=self.a_min and a<=self.a_max)):
            self.a = (self.a_max+self.a_min)/2

        leading_num1 = str(self.a_max).split(".")[0]
        leading_num2 = str(self.a_min).split(".")[0]
        if(leading_num1!=leading_num2):
            self.a = round((self.a_max+self.a_min)/2, 1)
            self._a_cached = True
            return self.a 

        if(abs(self.a)<1e-4):  # slopes below 1e-4 are snapped to zero
            self.a = 0

        self._a_cached = True
        return self.a

    @property
    def get_b(self):
        return self.b
    
