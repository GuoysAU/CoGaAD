"""SHRINK compressor: shrinking cones, cone merging and the byte layout of the Base.

Taken from the SHRINK code used for the paper. The cone construction, the merging and the slope selection are
unchanged. Differences from the original file: package-relative imports, the residual split is an argument
(`residual_split`) instead of a commented-out block, and the methods that wrote residual files through an
external coder or dumped intermediate files to fixed paths are left out.
"""
import math
import io
import time
import numpy as np
from io import BytesIO
from typing import List
from statsmodels.stats.diagnostic import acorr_ljungbox
from .FloatEncoder import FloatEncoder
from .UIntEncoder import UIntEncoder
from .VariableByteEncoder import VariableByteEncoder
from .Point import Point
from .SimPieceSegment import SimPieceSegment


class Shrink:
    def __init__(self, points=None, epsilon=None, window=None, bytes=None, variable_byte=False, zstd=False,
                 residual_split=False, local_origin=False):
        """Compress a list of Points (``points``, ``epsilon``) or restore a compressor from its bytes (``bytes``).

        epsilon:        quantization step and error bound of the cones.
        window:         window length of the series; bounds the buffer of the local error bound.
        residual_split: end a cone when its residuals fail the Ljung-Box white-noise test.
        local_origin:   quantize cone origins with the local error bound, as the original SHRINK (S2D); else epsilon.
        """
        if points is not None:  # compress the points
            start_time = time.time()
            if not points:
                raise ValueError("No points provided")

            # settings
            self.epsilon = epsilon
            self.window = window
            self.residual_split = residual_split
            self.local_origin = local_origin
            self.alpha = 0.01  # local interval: alpha * length * epsilon samples

            # the series
            self.points = points[:]
            self.values = [point.value for point in points]
            self.length = len(points)
            self.lastTimeStamp = points[-1].timestamp
            self.max, self.min = max(self.values), min(self.values)
            self.buflength = self.local_interval()

            # compression
            self.lengthofSegments = None
            self.segments = self.mergePerB(self.compress(points))
            self.baseTime = int((time.time() - start_time) * 1000)
        elif bytes is not None:  # restore a compressor from its bytes
            self.readByteArray(bytes, variable_byte, zstd)
        else:
            raise ValueError("Either points or bytes must be provided")

    def local_interval(self):
        """Local interval: alpha * length * epsilon samples, raised to the window or capped at 1% of the series."""
        interval = int(self.alpha * self.length * self.epsilon)
        if self.window is not None:
            if interval < self.window:
                return self.window
            if interval > 0.01 * self.length:
                return int(0.01 * self.length)
        return interval

    def dynamicEpsilon(self, startIdx, points):
        """Local error bound of the next buflength samples, epsilon * e^(2/3 - beta), where beta is the ratio of their
        value range to that of the series; returned with the end of the interval."""
        if self.buflength >= len(points):
            return self.epsilon, len(points) - 1
        values = [point.value for point in points[startIdx:startIdx + self.buflength]]
        beta = (max(values) - min(values)) / (self.max - self.min)
        scale = round(math.exp(2 / 3 - beta), 3)
        return round(self.epsilon * scale, 3), startIdx + self.buflength

    
    def quantization(self, value, localEpsilon):
        res = round(value / localEpsilon ) * localEpsilon
        return res
    
    def createSegment(self, startIdx, points, segments, localEpsilon):
        """One shrinking cone from points[startIdx]; appends it to segments and returns the start of the next one."""
        initTimestamp = points[startIdx].timestamp
        step = localEpsilon if self.local_origin else self.epsilon  # grid of the origin: local (SHRINK) or global (CoGa)
        b = round(self.quantization(points[startIdx].value, step), 2)
        if startIdx + 1 == len(points):  # a single point left
            segment = SimPieceSegment(initTimestamp, -math.inf, math.inf, b)
            segments.append(segment)
            return startIdx + 1, segment

        dt = points[startIdx + 1].timestamp - initTimestamp
        aMax = ((points[startIdx + 1].value + localEpsilon) - b) / dt
        aMin = ((points[startIdx + 1].value - localEpsilon) - b) / dt
        end = len(points)
        for idx in range(startIdx + 2, len(points)):
            dt = points[idx].timestamp - initTimestamp
            upValue = points[idx].value + localEpsilon
            downValue = points[idx].value - localEpsilon
            up_lim = aMax * dt + b
            down_lim = aMin * dt + b
            if downValue > up_lim or upValue < down_lim:  # the point leaves the cone
                end = idx
                break
            if upValue < up_lim:
                aMax = max((upValue - b) / dt, aMin)
            if downValue > down_lim:
                aMin = min((downValue - b) / dt, aMax)

            # Residual split: end the cone once its residuals are no longer white noise (Ljung-Box test at lag 1).
            if self.residual_split:
                seg_len = idx - startIdx + 1
                buf = max(self.buflength, 10)
                if seg_len >= 3 * buf:
                    end = idx + 1
                    break
                if seg_len >= buf and seg_len % buf == 0:
                    a = (aMin + aMax) / 2
                    residuals = [points[i].value - (a * (points[i].timestamp - initTimestamp) + b)
                                 for i in range(startIdx, idx + 1)]
                    with np.errstate(divide="ignore", invalid="ignore"):  # constant residuals: NaN, no split
                        p_value = acorr_ljungbox(residuals, lags=1, return_df=True)["lb_pvalue"].values[0]
                    if p_value < 0.05:
                        end = idx + 1
                        break

        segment = SimPieceSegment(initTimestamp, aMin, aMax, b)
        segments.append(segment)
        return end, segment

    

            
    def compress(self, points):
        """Cones from left to right; the local error bound is recomputed once the scan passes the local interval."""
        segments = []
        currentIdx, intervalEnd = 0, -1
        while currentIdx < len(points):
            if currentIdx > intervalEnd:
                localEpsilon, intervalEnd = self.dynamicEpsilon(currentIdx, points)
            currentIdx, _ = self.createSegment(currentIdx, points, segments, localEpsilon)
        return segments
    


        
    def mergePerB(self, segments):
        aMinTemp = float('-inf')
        aMaxTemp = float('inf')
        b = float('nan')
        timestamps = []
        mergedSegments = []
        self.lengthofSegments = 0

        segments.sort(key=lambda segment: (segment.get_b, segment.get_a))
        
        for i in range(len(segments)):
            if b != segments[i].get_b:
                if len(timestamps) == 1:
                    mergedSegments.append(SimPieceSegment(timestamps[0], aMinTemp, aMaxTemp, b))
                    self.lengthofSegments += 1
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                        self.lengthofSegments += 1

                
                timestamps.clear()
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = segments[i].get_a_min
                aMaxTemp = segments[i].get_a_max
                b = segments[i].get_b
                continue
            
            if segments[i].get_a_min <= aMaxTemp and segments[i].get_a_max >= aMinTemp:
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = max(aMinTemp, segments[i].get_a_min)
                aMaxTemp = min(aMaxTemp, segments[i].get_a_max)
            else:
                if len(timestamps) == 1:
                    mergedSegments.append(segments[i - 1])
                    self.lengthofSegments += 1
                else:
                    for timestamp in timestamps:
                        mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                        self.lengthofSegments += 1
                
                timestamps.clear()
                timestamps.append(segments[i].get_init_timestamp)
                aMinTemp = segments[i].get_a_min
                aMaxTemp = segments[i].get_a_max
        
        if timestamps:
            if len(timestamps) == 1:
                mergedSegments.append(SimPieceSegment(timestamps[0], aMinTemp, aMaxTemp, b))
                self.lengthofSegments += 1

            else:
                for timestamp in timestamps:
                    mergedSegments.append(SimPieceSegment(timestamp, aMinTemp, aMaxTemp, b))
                    self.lengthofSegments += 1

        return mergedSegments
    

    def decompress(self):
        start_time = time.time()
        
        # Pre-calculate the initial timestamps and other values
        init_timestamps = [segment.get_init_timestamp for segment in self.segments]
        a_values = [segment.a for segment in self.segments]
        b_values = [segment.get_b for segment in self.segments]
        points = []

        # Loop over segments, avoiding method calls within the loop
        for i in range(len(self.segments) - 1):
            # Calculate the range of timestamps for the current segment
            timestamps = range(init_timestamps[i], init_timestamps[i + 1])
            # Use a list comprehension to generate points
            points += [Point(ts, a_values[i] * (ts - init_timestamps[i]) + b_values[i]) for ts in timestamps]

        # Handle the last segment
        last_segment_timestamps = range(init_timestamps[-1], self.lastTimeStamp + 1)
        points += [Point(ts, a_values[-1] * (ts - init_timestamps[-1]) + b_values[-1]) for ts in last_segment_timestamps]
        

        end_time = time.time()
        decompBaseTime = int((end_time - start_time) * 1000)

        return points, decompBaseTime
    


        
    def toByteArrayPerBSegments(self, segments: List[SimPieceSegment], variableByte: bool, outStream: io.BytesIO) -> None:
        # Initialize a dictionary to organize segments by 'b' value
        input = {}

        for segment in segments:
            a = segment.get_a
            b = round(segment.get_b / self.epsilon)
            t = segment.get_init_timestamp
            
            if b not in input:
                input[b] = {}
            
            if a not in input[b]:
                input[b][a] = []
            
            input[b][a].append(t)
        
        # Write the size of the dictionary
        VariableByteEncoder.write(len(input), outStream)
        
        if not input.items():
            return
        
        previousB = min(input.keys())
        VariableByteEncoder.write(previousB, outStream)
        
        for b, aSegments in input.items():
            VariableByteEncoder.write(b - previousB, outStream)
            previousB = b
            VariableByteEncoder.write(len(aSegments), outStream)

            
            for a, timestamps in aSegments.items():
                # Custom method to encode the float 'a' value
                FloatEncoder.write(float(a), outStream)
                len(aSegments)
                
                VariableByteEncoder.write(len(timestamps), outStream)
                for timestamp in timestamps:
                    UIntEncoder.write(timestamp, outStream)

    def toByteArray(self, variableByte: bool, zstd: bool) -> bytes:
        """Byte layout of the Base: epsilon, then per intercept its slopes and their start times, then the last time
        stamp. Only the fixed-width layout (variableByte=False) is implemented."""
        if variableByte:
            raise NotImplementedError("variable-byte time stamps are not supported")
        outStream = BytesIO()
        bytes = None

        FloatEncoder.write(float(self.epsilon), outStream)

        self.toByteArrayPerBSegments(self.segments, variableByte, outStream)

        UIntEncoder.write(self.lastTimeStamp, outStream)

        if zstd:
            bytes = zstd.compress(outStream.getvalue())
        else:
            bytes = outStream.getvalue()

        outStream.close()
        return bytes
    
    def readMergedPerBSegments(self, variableByte, inStream):
        segments = []
        numB = VariableByteEncoder.read(inStream)

        if numB == 0:
            return segments

        previousB = VariableByteEncoder.read(inStream)

        for _ in range(numB):
            b = VariableByteEncoder.read(inStream) + previousB
            previousB = b
            numA = VariableByteEncoder.read(inStream)

            for _ in range(numA):
                a = FloatEncoder.read(inStream)
                numTimestamps = VariableByteEncoder.read(inStream)

                for _ in range(numTimestamps):
                    timestamp = UIntEncoder.read(inStream)
                    segments.append(SimPieceSegment(timestamp, a, a, b * self.epsilon))

        return segments

    def readByteArray(self, input, variableByte, zstd):
        if variableByte:
            raise NotImplementedError("variable-byte time stamps are not supported")
        if zstd:
            binary = zstd.decompress(input)
        else:
            binary = input

        inStream = BytesIO(binary)

        self.epsilon = FloatEncoder.read(inStream)
        self.segments = self.readMergedPerBSegments(variableByte, inStream)

        self.lastTimeStamp = UIntEncoder.read(inStream)

        inStream.close()


    
