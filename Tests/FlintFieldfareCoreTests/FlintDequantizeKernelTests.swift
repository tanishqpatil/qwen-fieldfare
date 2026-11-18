import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintDequantizeKernelTests {

    private static func parseHexToData(_ hex: String) -> Data {
        var data = Data(capacity: hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let nextIndex = hex.index(index, offsetBy: 2)
            let byteStr = hex[index..<nextIndex]
            if let byte = UInt8(byteStr, radix: 16) {
                data.append(byte)
            }
            index = nextIndex
        }
        return data
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private struct DequantGoldenCase: Decodable {
        let name: String
        let bits: Int
        let rows: Int
        let cols: Int
        let out_dtype: String
        let packed_hex: String
        let scales_hex: String
        let biases_hex: String
        let expected_hex: String
    }

    private struct DequantGoldenFixture: Decodable {
        let metadata: [String: AnyCodable]
        let cases: [DequantGoldenCase]
    }

    private struct AnyCodable: Decodable {}

    private static func loadGoldenFixture() throws -> DequantGoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_dequant_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(DequantGoldenFixture.self, from: data)
    }

    // MARK: - 1. Pinned Independent Reference Qualification (vs MLX 0.32.4)

    @Test func independentReferenceQualificationVsMLX() throws {
        guard let fixture = try Self.loadGoldenFixture() else {
            Issue.record("flint_dequant_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintDequantizeKernel(context: context)

        print("\n=== FLINT 3-BIT & 4-BIT DEQUANTIZATION NUMERICAL QUALIFICATION (vs MLX 0.32.4) ===")
        print(String(format: "%-50@ | %4@ | %6@ | %-8@ | %12@ | %12@ | %@",
                     "Case Name", "Bits", "Shape", "Dtype", "Max Abs Diff", "Rel L2", "Status"))
        print(String(repeating: "-", count: 115))

        for c in fixture.cases {
            let packedData = Self.parseHexToData(c.packed_hex)
            let scalesData = Self.parseHexToData(c.scales_hex)
            let biasesData = Self.parseHexToData(c.biases_hex)
            let expData = Self.parseHexToData(c.expected_hex)

            let bits: FlintDequantizeKernel.QuantBits = (c.bits == 3) ? .three : .four
            let outDtype: FlintDequantizeKernel.Dtype = (c.out_dtype == "bfloat16") ? .bfloat16 : .float32

            let packedBuf = try packedData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: packedData.count) }
            let scalesBuf = try scalesData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: scalesData.count) }
            let biasesBuf = try biasesData.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: biasesData.count) }
            let outBuf = try context.makeBuffer(byteCount: expData.count)

            try kernel.executeDequantize(
                packedWeights: packedBuf,
                scales: scalesBuf,
                biases: biasesBuf,
                output: outBuf,
                rows: c.rows,
                cols: c.cols,
                bits: bits,
                outDtype: outDtype,
                waitUntilCompleted: true
            )

            let totalElements = c.rows * c.cols
            var maxAbsDiff: Float = 0.0
            var l2DiffSum: Double = 0.0
            var l2RefSum: Double = 0.0

            if outDtype == .float32 {
                let metalFloats = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: totalElements))
                let expFloats = expData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

                for i in 0..<totalElements {
                    let diff = abs(metalFloats[i] - expFloats[i])
                    if diff > maxAbsDiff { maxAbsDiff = diff }
                    let diffD = Double(diff)
                    let refD = Double(expFloats[i])
                    l2DiffSum += diffD * diffD
                    l2RefSum += refD * refD
                }
            } else {
                // bfloat16: unpack uint16 and cast to float
                let metalU16 = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: UInt16.self), count: totalElements))
                let expU16 = expData.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }

                for i in 0..<totalElements {
                    // Exact bit comparison on bfloat16 representation
                    let mVal = Float(bitPattern: UInt32(metalU16[i]) << 16)
                    let eVal = Float(bitPattern: UInt32(expU16[i]) << 16)
                    let diff = abs(mVal - eVal)
                    if diff > maxAbsDiff { maxAbsDiff = diff }
                    let diffD = Double(diff)
                    let refD = Double(eVal)
                    l2DiffSum += diffD * diffD
                    l2RefSum += refD * refD
                }
            }

            let relL2 = (l2RefSum > 1e-12) ? sqrt(l2DiffSum / l2RefSum) : sqrt(l2DiffSum)
            #expect(maxAbsDiff <= 1e-4, "Case \(c.name) max abs diff \(maxAbsDiff) exceeds 1e-4")
            #expect(relL2 <= 2e-3, "Case \(c.name) relative L2 \(relL2) exceeds 2e-3")

            let shapeStr = "\(c.rows)x\(c.cols)"
            print(String(format: "%-50@ | %4d | %6@ | %-8@ | %12.6f | %12.8f | PASS",
                         c.name, c.bits, shapeStr, c.out_dtype, maxAbsDiff, relL2))
        }
        print(String(repeating: "-", count: 115))
    }

    // MARK: - 2. Analytical Boundary-Crossing & CPU Reference Contract

    @Test func analyticalBoundaryCrossingContract() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintDequantizeKernel(context: context)

        // 3-bit matrix with word-boundary crossings
        let rows = 4
        let cols = 128 // 2 groups
        let wordsPerRow = (cols / 64) * 6 // 12 words

        var packedWords = [UInt32](repeating: 0, count: rows * wordsPerRow)
        var scales = [Float](repeating: 0, count: rows * 2)
        var biases = [Float](repeating: 0, count: rows * 2)

        for r in 0..<rows {
            scales[r * 2 + 0] = 1.5 + Float(r) * 0.25
            scales[r * 2 + 1] = 0.75 + Float(r) * 0.1
            biases[r * 2 + 0] = -1.0 - Float(r) * 0.5
            biases[r * 2 + 1] = 0.5 + Float(r) * 0.2

            for w in 0..<wordsPerRow {
                packedWords[r * wordsPerRow + w] = UInt32.random(in: 0...UInt32.max)
            }
            // Explicitly test word boundary crossing in group 0:
            // Element 10 crosses word 0 (bits 30..31) and word 1 (bit 0)
            packedWords[r * wordsPerRow + 0] |= (1 << 30) | (1 << 31)
            packedWords[r * wordsPerRow + 1] |= (1 << 0)
            // Element 21 crosses word 1 (bit 31) and word 2 (bits 0..1)
            packedWords[r * wordsPerRow + 1] |= (1 << 31)
            packedWords[r * wordsPerRow + 2] |= (3 << 0)
        }

        let scalesBF16Data = scales.map { UInt16(truncatingIfNeeded: ($0.bitPattern >> 16)) }
        let biasesBF16Data = biases.map { UInt16(truncatingIfNeeded: ($0.bitPattern >> 16)) }
        let scalesBF16Floats = scalesBF16Data.map { Float(bitPattern: UInt32($0) << 16) }
        let biasesBF16Floats = biasesBF16Data.map { Float(bitPattern: UInt32($0) << 16) }

        let refOut = FlintDequantizeReference.evaluate3Bit(
            packedWeights: packedWords,
            scales: scalesBF16Floats,
            biases: biasesBF16Floats,
            rows: rows,
            cols: cols
        )

        // Run Metal dequantize to FP32
        let packedBuf = try packedWords.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let scalesBuf = try scalesBF16Data.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let biasesBuf = try biasesBF16Data.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let outBuf = try context.makeBuffer(byteCount: rows * cols * 4)

        try kernel.executeDequantize(
            packedWeights: packedBuf,
            scales: scalesBuf,
            biases: biasesBuf,
            output: outBuf,
            rows: rows,
            cols: cols,
            bits: .three,
            outDtype: .float32,
            waitUntilCompleted: true
        )

        let metalOut = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: Float.self), count: rows * cols))
        var maxDiff: Float = 0
        for i in 0..<metalOut.count {
            let diff = abs(metalOut[i] - refOut[i])
            if diff > maxDiff { maxDiff = diff }
        }
        #expect(maxDiff <= 1e-4, "Metal 3-bit output differs from CPU reference by \(maxDiff)")

        // Also test 4-bit CPU reference equivalence
        let wordsPerRow4 = (cols / 64) * 8
        var packedWords4 = [UInt32](repeating: 0, count: rows * wordsPerRow4)
        for i in 0..<packedWords4.count {
            packedWords4[i] = UInt32.random(in: 0...UInt32.max)
        }
        let refOut4 = FlintDequantizeReference.evaluate4Bit(
            packedWeights: packedWords4,
            scales: scalesBF16Floats,
            biases: biasesBF16Floats,
            rows: rows,
            cols: cols
        )

        let packedBuf4 = try packedWords4.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let outBuf4 = try context.makeBuffer(byteCount: rows * cols * 4)
        try kernel.executeDequantize(
            packedWeights: packedBuf4,
            scales: scalesBuf,
            biases: biasesBuf,
            output: outBuf4,
            rows: rows,
            cols: cols,
            bits: .four,
            outDtype: .float32,
            waitUntilCompleted: true
        )

        let metalOut4 = Array(UnsafeBufferPointer(start: outBuf4.contents().assumingMemoryBound(to: Float.self), count: rows * cols))
        var maxDiff4: Float = 0
        for i in 0..<metalOut4.count {
            let diff = abs(metalOut4[i] - refOut4[i])
            if diff > maxDiff4 { maxDiff4 = diff }
        }
        #expect(maxDiff4 <= 1e-4, "Metal 4-bit output differs from CPU reference by \(maxDiff4)")
    }

    // MARK: - 3. Repeat-Run Bitwise Determinism

    @Test func repeatRunDeterminism() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintDequantizeKernel(context: context)

        let rows = 8
        let cols = 128
        let packedWords = (0..<rows * (cols / 64) * 6).map { _ in UInt32.random(in: 0...UInt32.max) }
        let scalesBF16: [UInt16] = (0..<rows * 2).map { _ in 0x3F80 } // 1.0 in BF16
        let biasesBF16: [UInt16] = (0..<rows * 2).map { _ in 0x0000 }

        let packedBuf = try packedWords.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let scalesBuf = try scalesBF16.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let biasesBuf = try biasesBF16.withUnsafeBytes { try context.makeBuffer(bytes: $0.baseAddress!, byteCount: $0.count) }
        let outBuf = try context.makeBuffer(byteCount: rows * cols * 2)

        try kernel.executeDequantize(
            packedWeights: packedBuf,
            scales: scalesBuf,
            biases: biasesBuf,
            output: outBuf,
            rows: rows,
            cols: cols,
            bits: .three,
            outDtype: .bfloat16,
            waitUntilCompleted: true
        )
        let baselineData = Data(bytes: outBuf.contents(), count: rows * cols * 2)
        let baselineHash = Self.sha256Hex(baselineData)

        for run in 1...50 {
            try kernel.executeDequantize(
                packedWeights: packedBuf,
                scales: scalesBuf,
                biases: biasesBuf,
                output: outBuf,
                rows: rows,
                cols: cols,
                bits: .three,
                outDtype: .bfloat16,
                waitUntilCompleted: true
            )
            let runData = Data(bytes: outBuf.contents(), count: rows * cols * 2)
            let runHash = Self.sha256Hex(runData)
            #expect(runHash == baselineHash, "Run \(run) output differed bitwise from baseline!")
        }
    }

    // MARK: - 4. Buffer Bounds & Parameter Validation

    @Test func bufferBoundsAndParameterValidation() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintDequantizeKernel(context: context)

        let packedBuf = try context.makeBuffer(byteCount: 1024)
        let scalesBuf = try context.makeBuffer(byteCount: 256)
        let biasesBuf = try context.makeBuffer(byteCount: 256)
        let outBuf = try context.makeBuffer(byteCount: 2048)

        // 1. Invalid columns (not multiple of 64)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDequantize(
                packedWeights: packedBuf, scales: scalesBuf, biases: biasesBuf, output: outBuf,
                rows: 2, cols: 63, bits: .three
            )
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDequantize(
                packedWeights: packedBuf, scales: scalesBuf, biases: biasesBuf, output: outBuf,
                rows: 2, cols: 65, bits: .three
            )
        }

        // 2. Negative dimensions
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDequantize(
                packedWeights: packedBuf, scales: scalesBuf, biases: biasesBuf, output: outBuf,
                rows: -1, cols: 64, bits: .three
            )
        }

        // 3. Negative / unaligned offsets
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDequantize(
                packedWeights: packedBuf, packedWeightsOffset: -1,
                scales: scalesBuf, biases: biasesBuf, output: outBuf,
                rows: 1, cols: 64, bits: .three
            )
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDequantize(
                packedWeights: packedBuf, packedWeightsOffset: 1, // Not 4-byte aligned
                scales: scalesBuf, biases: biasesBuf, output: outBuf,
                rows: 1, cols: 64, bits: .three
            )
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDequantize(
                packedWeights: packedBuf, scales: scalesBuf, scalesOffset: 1, // Not 2-byte aligned
                biases: biasesBuf, output: outBuf,
                rows: 1, cols: 64, bits: .three
            )
        }

        // 4. Undersized buffer checks
        let smallOut = try context.makeBuffer(byteCount: 32)
        #expect(throws: FlintMetalError.self) {
            try kernel.executeDequantize(
                packedWeights: packedBuf, scales: scalesBuf, biases: biasesBuf, output: smallOut,
                rows: 2, cols: 64, bits: .three
            )
        }
    }

    // MARK: - 5. Real Checkpoint Routed Expert Integration & Immutability

    @Test func realCheckpointRoutedExpertIntegration() throws {
        let modelDir = "/Users/tanishq/models/flint-mixed-3-4bit"
        let shard1URL = URL(fileURLWithPath: modelDir).appendingPathComponent("model-00001-of-00004.safetensors")
        guard FileManager.default.fileExists(atPath: shard1URL.path) else {
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintDequantizeKernel(context: context)
        let shard1 = try FlintSafetensorsFile(url: shard1URL)

        // Read single expert slice from Layer 2 switch_mlp.gate_proj
        // Tensor shape is [256, 512, 192] UInt32.
        // Slice for expert 0: 512 rows, 192 packed UInt32 columns (unquantized 512x2048).
        // 512 * 192 * 4 = 393,216 bytes.
        let expertGateWeightBytes = 512 * 192 * 4
        let expertScalesBytes = 512 * 32 * 2
        let expertBiasesBytes = 512 * 32 * 2
        let expertOutBytes = 512 * 2048 * 2 // 2 MiB BF16

        let packedBuf = try context.makeBuffer(byteCount: expertGateWeightBytes)
        let scalesBuf = try context.makeBuffer(byteCount: expertScalesBytes)
        let biasesBuf = try context.makeBuffer(byteCount: expertBiasesBytes)
        let outBuf = try context.makeBuffer(byteCount: expertOutBytes)

        let packedSlice = UnsafeMutableRawBufferPointer(start: packedBuf.contents(), count: expertGateWeightBytes)
        let scalesSlice = UnsafeMutableRawBufferPointer(start: scalesBuf.contents(), count: expertScalesBytes)
        let biasesSlice = UnsafeMutableRawBufferPointer(start: biasesBuf.contents(), count: expertBiasesBytes)

        try shard1.readTensorSlice("model.layers.2.mlp.switch_mlp.gate_proj.weight", byteRange: 0..<expertGateWeightBytes, into: packedSlice)
        try shard1.readTensorSlice("model.layers.2.mlp.switch_mlp.gate_proj.scales", byteRange: 0..<expertScalesBytes, into: scalesSlice)
        try shard1.readTensorSlice("model.layers.2.mlp.switch_mlp.gate_proj.biases", byteRange: 0..<expertBiasesBytes, into: biasesSlice)

        let packedBeforeHash = Self.sha256Hex(Data(bytes: packedBuf.contents(), count: expertGateWeightBytes))
        let scalesBeforeHash = Self.sha256Hex(Data(bytes: scalesBuf.contents(), count: expertScalesBytes))
        let biasesBeforeHash = Self.sha256Hex(Data(bytes: biasesBuf.contents(), count: expertBiasesBytes))

        try kernel.executeDequantize(
            packedWeights: packedBuf,
            scales: scalesBuf,
            biases: biasesBuf,
            output: outBuf,
            rows: 512,
            cols: 2048,
            bits: .three,
            outDtype: .bfloat16,
            waitUntilCompleted: true
        )

        // Verify input immutability
        let packedAfterHash = Self.sha256Hex(Data(bytes: packedBuf.contents(), count: expertGateWeightBytes))
        let scalesAfterHash = Self.sha256Hex(Data(bytes: scalesBuf.contents(), count: expertScalesBytes))
        let biasesAfterHash = Self.sha256Hex(Data(bytes: biasesBuf.contents(), count: expertBiasesBytes))

        #expect(packedBeforeHash == packedAfterHash, "Packed weights corrupted during dequantization!")
        #expect(scalesBeforeHash == scalesAfterHash, "Scales corrupted during dequantization!")
        #expect(biasesBeforeHash == biasesAfterHash, "Biases corrupted during dequantization!")

        // Verify output buffer contains valid non-zero values
        let outU16 = Array(UnsafeBufferPointer(start: outBuf.contents().assumingMemoryBound(to: UInt16.self), count: 512 * 2048))
        var nonZeroCount = 0
        for val in outU16 {
            if val != 0 { nonZeroCount += 1 }
        }
        #expect(nonZeroCount > 1000, "Output buffer contains all zeros after dequantizing real expert!")
    }

    // MARK: - 6. Apple M5 Hardware Benchmark & Telemetry

    @Test func benchmarkDequantizePerformanceOnM5() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintDequantizeKernel(context: context)

        print("\n=== FLINT DEQUANTIZATION HARDWARE BENCHMARK (APPLE M5) ===")

        let benchmarks: [(name: String, rows: Int, cols: Int, bits: FlintDequantizeKernel.QuantBits, dtype: FlintDequantizeKernel.Dtype)] = [
            ("Shared Expert Gate Proj (4-bit)", 1024, 2048, .four, .bfloat16),
            ("Routed Expert Gate Proj (3-bit)", 512, 2048, .three, .bfloat16),
            ("Routed Expert Down Proj (3-bit)", 2048, 512, .three, .bfloat16),
            ("Routed Expert Gate Proj FP32 (3-bit)", 512, 2048, .three, .float32)
        ]

        let warmupCount = 20
        let benchCount = 100

        for b in benchmarks {
            let numGroups = b.cols / 64
            let wordsPerRow = numGroups * b.bits.wordsPerGroup
            let packedBytes = b.rows * wordsPerRow * 4
            let scalesBytes = b.rows * numGroups * 2
            let biasesBytes = scalesBytes
            let outBytes = b.rows * b.cols * b.dtype.elementByteWidth

            let packedBuf = try context.makeBuffer(byteCount: packedBytes)
            let scalesBuf = try context.makeBuffer(byteCount: scalesBytes)
            let biasesBuf = try context.makeBuffer(byteCount: biasesBytes)
            let outBuf = try context.makeBuffer(byteCount: outBytes)

            // Warmup
            for _ in 0..<warmupCount {
                try kernel.executeDequantize(
                    packedWeights: packedBuf,
                    scales: scalesBuf,
                    biases: biasesBuf,
                    output: outBuf,
                    rows: b.rows,
                    cols: b.cols,
                    bits: b.bits,
                    outDtype: b.dtype,
                    waitUntilCompleted: true
                )
            }

            // Timed iterations
            var durations = [Double]()
            durations.reserveCapacity(benchCount)

            for _ in 0..<benchCount {
                let dur = try kernel.executeDequantize(
                    packedWeights: packedBuf,
                    scales: scalesBuf,
                    biases: biasesBuf,
                    output: outBuf,
                    rows: b.rows,
                    cols: b.cols,
                    bits: b.bits,
                    outDtype: b.dtype,
                    waitUntilCompleted: true
                )
                if dur > 0 {
                    durations.append(dur * 1e6) // convert to µs
                }
            }

            if !durations.isEmpty {
                durations.sort()
                let p50 = durations[durations.count / 2]
                let minVal = durations.first!
                let p95 = durations[Int(Double(durations.count) * 0.95)]
                let maxVal = durations.last!
                let jitter = p95 - minVal

                let totalDataBytes = Double(packedBytes + scalesBytes + biasesBytes + outBytes)
                let throughputGBps = (totalDataBytes / (p50 * 1e-6)) / 1e9

                print(String(format: "%-38@ | P50: %6.2f µs | Min: %6.2f µs | P95: %6.2f µs | Max: %6.2f µs | Jitter: %5.2f µs | %6.2f GB/s",
                             b.name, p50, minVal, p95, maxVal, jitter, throughputGBps))
            }
        }
        print("=========================================================\n")
    }

    // MARK: - 7. Canonical Metal Shader Source Equivalence

    @Test func canonicalMetalShaderSourceEquivalence() throws {
        let metalPath = "Sources/FlintFieldfareCore/Metal/flint_dequantize.metal"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(metalPath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            Issue.record("flint_dequantize.metal not found at path: \(fullPath.path)")
            return
        }
        let diskSource = try String(contentsOf: fullPath, encoding: .utf8)
        #expect(diskSource == FlintDequantizeKernel.embeddedShaderSource, "Disk flint_dequantize.metal and FlintDequantizeKernel.embeddedShaderSource have drifted!")
    }
}
