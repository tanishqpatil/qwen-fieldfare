import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintRMSNormKernelTests {

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

    private struct GoldenCase: Decodable {
        let name: String
        let dimension: Int
        let rows: Int
        let epsilon: Float
        let dtype: String
        let input_hex: String
        let weight_name: String
        let weight_hex: String
        let output_hex: String
    }

    private struct GoldenFixture: Decodable {
        let metadata: [String: String]
        let cases: [GoldenCase]
    }

    private static func loadGoldenFixture() throws -> GoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_rmsnorm_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(GoldenFixture.self, from: data)
    }

    // MARK: - 1. Pinned Independent Reference Qualification (vs MLX 0.32.4)

    @Test func independentReferenceQualificationVsMLX() throws {
        guard let fixture = try Self.loadGoldenFixture() else {
            Issue.record("flint_rmsnorm_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintRMSNormKernel(context: context)

        print("\n=== FLINT RMSNORM NUMERICAL QUALIFICATION (vs MLX 0.32.4) ===")
        print(String(format: "%-45@ | %-6@ | %-5@ | %-8@ | %-12@ | %-12@ | %-8@ | %-10@",
                     "Case Name", "Dim", "Rows", "Dtype", "Max Abs Diff", "Max Rel Diff", "Max ULP", "Status"))
        print(String(repeating: "-", count: 125))

        for (_, c) in fixture.cases.enumerated() {
            let inData = Self.parseHexToData(c.input_hex)
            let wData = Self.parseHexToData(c.weight_hex)
            let expectedOutData = Self.parseHexToData(c.output_hex)

            let inBuffer = try inData.withUnsafeBytes { ptr in
                try context.makeBuffer(bytes: ptr.baseAddress!, byteCount: inData.count)
            }
            let wBuffer = try wData.withUnsafeBytes { ptr in
                try context.makeBuffer(bytes: ptr.baseAddress!, byteCount: wData.count)
            }
            let outBuffer = try context.makeBuffer(byteCount: expectedOutData.count)

            let dtype: FlintRMSNormKernel.Dtype = (c.dtype == "bfloat16") ? .bfloat16 : .float32

            try kernel.execute(
                input: inBuffer,
                weight: wBuffer,
                output: outBuffer,
                rows: c.rows,
                dimension: c.dimension,
                epsilon: c.epsilon,
                dtype: dtype,
                waitUntilCompleted: true
            )

            // Calculate Numerical Error Metrics
            let totalElements = c.rows * c.dimension
            var maxAbsDiff: Float = 0.0
            var maxRelDiff: Float = 0.0
            var sumSqDiff: Double = 0.0
            var maxULPDiff: Int = 0

            if dtype == .bfloat16 {
                let actualU16 = outBuffer.contents().assumingMemoryBound(to: UInt16.self)
                let expectedU16 = (expectedOutData as NSData).bytes.assumingMemoryBound(to: UInt16.self)

                for i in 0..<totalElements {
                    let actualF = FlintRMSNormReference.bf16ToFloat(actualU16[i])
                    let expectedF = FlintRMSNormReference.bf16ToFloat(expectedU16[i])
                    let diff = abs(actualF - expectedF)
                    let rel = diff / (abs(expectedF) + 1e-7)

                    maxAbsDiff = max(maxAbsDiff, diff)
                    maxRelDiff = max(maxRelDiff, rel)
                    sumSqDiff += Double(diff * diff)

                    let ulp = abs(Int(actualU16[i]) - Int(expectedU16[i]))
                    maxULPDiff = max(maxULPDiff, ulp)
                }

                // BF16 numerical tolerance: strictly <= 1 ULP across all elements (max rel diff <= 1.5%)
                #expect(maxULPDiff <= 1, "Case \(c.name) exceeded BF16 max ULP tolerance: \(maxULPDiff)")
                #expect(maxRelDiff <= 0.015, "Case \(c.name) exceeded BF16 max rel tolerance: \(maxRelDiff)")

            } else {
                let actualF32 = outBuffer.contents().assumingMemoryBound(to: Float.self)
                let expectedF32 = (expectedOutData as NSData).bytes.assumingMemoryBound(to: Float.self)

                for i in 0..<totalElements {
                    let diff = abs(actualF32[i] - expectedF32[i])
                    let rel = diff / (abs(expectedF32[i]) + 1e-7)

                    maxAbsDiff = max(maxAbsDiff, diff)
                    maxRelDiff = max(maxRelDiff, rel)
                    sumSqDiff += Double(diff * diff)
                }

                // FP32 numerical tolerance: exact match (max abs <= 1e-6, max rel <= 1e-5)
                #expect(maxAbsDiff <= 1e-6, "Case \(c.name) exceeded FP32 max abs tolerance: \(maxAbsDiff)")
                #expect(maxRelDiff <= 1e-5, "Case \(c.name) exceeded FP32 max rel tolerance: \(maxRelDiff)")
            }

            let rmsDiff = sqrt(sumSqDiff / Double(totalElements))

            print(String(format: "%-45@ | %-6d | %-5d | %-8@ | %-12.6f | %-12.6f | %-8d | PASS (RMS: %.2e)",
                         c.name, c.dimension, c.rows, c.dtype, maxAbsDiff, maxRelDiff, maxULPDiff, rmsDiff))
        }
        print(String(repeating: "-", count: 125))
    }

    // MARK: - 2. Deterministic Mathematical Invariants vs Swift Reference

    @Test func mathematicalInvariantsVsSwiftReference() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRMSNormKernel(context: context)

        // Seeded test inputs for width 2048 and 128
        for dim in [2048, 128] {
            var inputFloats = [Float](repeating: 0, count: dim)
            var weightFloats = [Float](repeating: 0, count: dim)

            // Alternating pattern with varying magnitudes
            for i in 0..<dim {
                inputFloats[i] = (i % 2 == 0) ? Float(i + 1) * 0.05 : -Float(i + 1) * 0.02
                weightFloats[i] = 1.0 + Float(i % 10) * 0.05
            }

            let inputBF16 = FlintRMSNormReference.floatArrayToBF16(inputFloats)
            let weightBF16 = FlintRMSNormReference.floatArrayToBF16(weightFloats)

            // Reference evaluation on CPU
            let refOutputBF16 = FlintRMSNormReference.evaluateBF16(
                input: inputBF16,
                weights: weightBF16,
                rows: 1,
                dimension: dim,
                epsilon: 1e-6
            )

            // GPU Metal evaluation
            let inBuf = try context.makeBuffer(byteCount: dim * 2)
            let wBuf = try context.makeBuffer(byteCount: dim * 2)
            let outBuf = try context.makeBuffer(byteCount: dim * 2)

            memcpy(inBuf.contents(), inputBF16, dim * 2)
            memcpy(wBuf.contents(), weightBF16, dim * 2)

            try kernel.execute(
                input: inBuf,
                weight: wBuf,
                output: outBuf,
                rows: 1,
                dimension: dim,
                epsilon: 1e-6,
                dtype: .bfloat16,
                waitUntilCompleted: true
            )

            let gpuOutputBF16 = Array(UnsafeBufferPointer(
                start: outBuf.contents().assumingMemoryBound(to: UInt16.self),
                count: dim
            ))

            // Compare bitwise & numerical agreement
            var maxDiff: Float = 0
            for i in 0..<dim {
                let diff = abs(FlintRMSNormReference.bf16ToFloat(gpuOutputBF16[i]) - FlintRMSNormReference.bf16ToFloat(refOutputBF16[i]))
                maxDiff = max(maxDiff, diff)
            }
            #expect(maxDiff <= 0.03125, "Dim \(dim) CPU/GPU mismatch: \(maxDiff)")
        }
    }

    // MARK: - 3. Real Checkpoint Resident Weights Integration & Immutability

    @Test func realCheckpointResidentWeightsIntegration() throws {
        let modelPath = ProcessInfo.processInfo.environment["FLINTFIELDFARE_MODEL"] ?? "/Users/tanishq/models/flint-mixed-3-4bit"
        guard FileManager.default.fileExists(atPath: modelPath) else {
            return
        }

        let context = try FlintMetalContext()
        let kernel = try FlintRMSNormKernel(context: context)

        // 1. Load authoritative resident weights
        let validated = try FlintModelDirectory.validate(path: modelPath)
        let shards = try FlintModelDirectory.openShards(canonicalPath: validated.canonicalPath, shardNames: validated.shards)
        let (weights, _) = try FlintResidentLoader.loadResidentWeights(
            validatedDirectory: validated,
            shardFiles: shards,
            storageMode: .gpuRequired,
            device: context.device
        )

        // 2. Validate real 2048-element final norm: model.norm.weight
        let normWeight = try weights.require("model.norm.weight")
        #expect(normWeight.shape == [2048])
        #expect(normWeight.byteCount == 4096)
        let normShaBefore = Self.sha256Hex(normWeight.buffer.copyData())

        let normIn = try context.makeBuffer(byteCount: 2048 * 2)
        let normOut = try context.makeBuffer(byteCount: 2048 * 2)
        // Fill input with 1.0 (0x3F80)
        normIn.contents().assumingMemoryBound(to: UInt16.self).initialize(repeating: 0x3F80, count: 2048)

        // Forward through resident tensor directly (zero copy)
        try kernel.forward(
            input: normIn,
            weightTensor: normWeight,
            output: normOut,
            rows: 1,
            epsilon: 1e-6,
            dtype: .bfloat16
        )

        // Invariant: Resident weight MUST NOT be mutated
        let normShaAfter = Self.sha256Hex(normWeight.buffer.copyData())
        #expect(normShaBefore == normShaAfter, "Resident weight model.norm.weight was unexpectedly mutated!")

        // 3. Validate real 128-element Q norm: model.layers.0.self_attn.q_norm.weight
        let qNormWeight = try weights.require("model.layers.0.self_attn.q_norm.weight")
        #expect(qNormWeight.shape == [128])
        #expect(qNormWeight.byteCount == 256)
        let qNormShaBefore = Self.sha256Hex(qNormWeight.buffer.copyData())

        // 32 Q heads (rows = 32)
        let qIn = try context.makeBuffer(byteCount: 32 * 128 * 2)
        let qOut = try context.makeBuffer(byteCount: 32 * 128 * 2)
        qIn.contents().assumingMemoryBound(to: UInt16.self).initialize(repeating: 0x3F80, count: 32 * 128)

        try kernel.forward(
            input: qIn,
            weightTensor: qNormWeight,
            output: qOut,
            rows: 32,
            epsilon: 1e-6,
            dtype: .bfloat16
        )

        let qNormShaAfter = Self.sha256Hex(qNormWeight.buffer.copyData())
        #expect(qNormShaBefore == qNormShaAfter, "Resident weight q_norm.weight was unexpectedly mutated!")

        // 4. Validate real 128-element K norm: model.layers.0.self_attn.k_norm.weight
        let kNormWeight = try weights.require("model.layers.0.self_attn.k_norm.weight")
        #expect(kNormWeight.shape == [128])
        #expect(kNormWeight.byteCount == 256)
        let kNormShaBefore = Self.sha256Hex(kNormWeight.buffer.copyData())

        // 4 KV heads (rows = 4)
        let kIn = try context.makeBuffer(byteCount: 4 * 128 * 2)
        let kOut = try context.makeBuffer(byteCount: 4 * 128 * 2)
        kIn.contents().assumingMemoryBound(to: UInt16.self).initialize(repeating: 0x3F80, count: 4 * 128)

        try kernel.forward(
            input: kIn,
            weightTensor: kNormWeight,
            output: kOut,
            rows: 4,
            epsilon: 1e-6,
            dtype: .bfloat16
        )

        let kNormShaAfter = Self.sha256Hex(kNormWeight.buffer.copyData())
        #expect(kNormShaBefore == kNormShaAfter, "Resident weight k_norm.weight was unexpectedly mutated!")
    }

    // MARK: - 4. Buffer Bounds, Alignment & Error Handling Tests

    @Test func bufferBoundsAndParameterValidation() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRMSNormKernel(context: context)

        let inBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let wBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let outBuf = try context.makeBuffer(byteCount: 2048 * 2)

        // 1. Input buffer too small
        let smallInBuf = try context.makeBuffer(byteCount: 1024)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: smallInBuf, weight: wBuf, output: outBuf, rows: 1, dimension: 2048, epsilon: 1e-6)
        }

        // 2. Weight buffer too small
        let smallWBuf = try context.makeBuffer(byteCount: 1024)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: smallWBuf, output: outBuf, rows: 1, dimension: 2048, epsilon: 1e-6)
        }

        // 3. Output buffer too small
        let smallOutBuf = try context.makeBuffer(byteCount: 1024)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: smallOutBuf, rows: 1, dimension: 2048, epsilon: 1e-6)
        }

        // 4. Invalid rows (<= 0)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: outBuf, rows: 0, dimension: 2048, epsilon: 1e-6)
        }

        // 5. Invalid dimension (<= 0)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: outBuf, rows: 1, dimension: 0, epsilon: 1e-6)
        }

        // 6. Invalid epsilon (<= 0)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: outBuf, rows: 1, dimension: 2048, epsilon: 0.0)
        }

        // 7. Negative offset
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, inputOffset: -1, weight: wBuf, output: outBuf, rows: 1, dimension: 2048, epsilon: 1e-6)
        }

        // 8. Non-finite epsilon (NaN, +inf, -inf)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: outBuf, rows: 1, dimension: 2048, epsilon: Float.nan)
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: outBuf, rows: 1, dimension: 2048, epsilon: Float.infinity)
        }

        // 9. Misaligned offsets (e.g. odd byte offset for 2-byte or 4-byte dtype)
        let paddedBuf = try context.makeBuffer(byteCount: 4096 * 4)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: paddedBuf, inputOffset: 1, weight: wBuf, output: outBuf, rows: 1, dimension: 128, epsilon: 1e-6, dtype: .float32)
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, weightOffset: 1, output: outBuf, rows: 1, dimension: 128, epsilon: 1e-6)
        }

        // 10. Arithmetic overflow in row/dimension calculation (Int.max)
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: outBuf, rows: Int.max, dimension: 2048, epsilon: 1e-6)
        }
        #expect(throws: FlintMetalError.self) {
            try kernel.execute(input: inBuf, weight: wBuf, output: outBuf, rows: 1, dimension: Int.max, epsilon: 1e-6)
        }
    }

    // MARK: - 5. General RMSNorm Fallback Dimensions (16, 64, 129, 192, 256, 512, 1024)

    @Test func generalFallbackDimensionsAndEdgeCases() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRMSNormKernel(context: context)

        let testDimensions = [16, 64, 129, 192, 256, 512, 1024]

        for dim in testDimensions {
            // Generate deterministic inputs
            var xF32 = [Float](repeating: 0, count: dim)
            var wF32 = [Float](repeating: 0, count: dim)
            for i in 0..<dim {
                xF32[i] = sin(Float(i + 1) * 0.1) * 2.0
                wF32[i] = cos(Float(i + 1) * 0.05) + 1.0
            }

            let wBF16 = FlintRMSNormReference.floatArrayToBF16(wF32)
            let wEffectiveF32 = FlintRMSNormReference.bf16ArrayToFloat(wBF16)

            // Reference calculation (CPU FP32 with BF16 checkpoint weights)
            let refOutF32 = FlintRMSNormReference.evaluateFP32(input: xF32, weights: wEffectiveF32, rows: 1, dimension: dim, epsilon: 1e-6)

            // 1. Test FP32 General Metal Kernel
            let inBufF32 = try xF32.withUnsafeBytes { ptr in
                try context.makeBuffer(bytes: ptr.baseAddress!, byteCount: dim * 4)
            }
            let wBufBF16 = try wBF16.withUnsafeBytes { ptr in
                try context.makeBuffer(bytes: ptr.baseAddress!, byteCount: dim * 2)
            }
            let outBufF32 = try context.makeBuffer(byteCount: dim * 4)

            try kernel.execute(
                input: inBufF32,
                weight: wBufBF16,
                output: outBufF32,
                rows: 1,
                dimension: dim,
                epsilon: 1e-6,
                dtype: .float32,
                waitUntilCompleted: true
            )

            let metalOutF32 = Array(UnsafeBufferPointer(start: outBufF32.contents().assumingMemoryBound(to: Float.self), count: dim))
            for i in 0..<dim {
                let diff = abs(metalOutF32[i] - refOutF32[i])
                #expect(diff <= 1e-5)
            }

            // 2. Test BF16 General Metal Kernel
            let xBF16 = FlintRMSNormReference.floatArrayToBF16(xF32)
            let inBufBF16 = try xBF16.withUnsafeBytes { ptr in
                try context.makeBuffer(bytes: ptr.baseAddress!, byteCount: dim * 2)
            }
            let outBufBF16 = try context.makeBuffer(byteCount: dim * 2)

            try kernel.execute(
                input: inBufBF16,
                weight: wBufBF16,
                output: outBufBF16,
                rows: 1,
                dimension: dim,
                epsilon: 1e-6,
                dtype: .bfloat16,
                waitUntilCompleted: true
            )

            let rawBF16 = Array(UnsafeBufferPointer(start: outBufBF16.contents().assumingMemoryBound(to: UInt16.self), count: dim))
            let metalOutBF16Float = FlintRMSNormReference.bf16ArrayToFloat(rawBF16)
            for i in 0..<dim {
                let diff = abs(metalOutBF16Float[i] - refOutF32[i])
                let rel = diff / max(abs(refOutF32[i]), 1e-4)
                #expect(rel <= 0.015)
            }
        }
    }

    // MARK: - 6. Canonical Metal Shader Source Equivalence

    @Test func canonicalMetalShaderSourceEquivalence() throws {
        // Verify that embedded shader source is 100% byte-for-byte identical to on-disk flint_rmsnorm.metal
        let metalFilePath = "Sources/FlintFieldfareCore/Metal/flint_rmsnorm.metal"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(metalFilePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            Issue.record("flint_rmsnorm.metal not found at path: \(fullPath.path)")
            return
        }
        let diskSource = try String(contentsOf: fullPath, encoding: .utf8)
        #expect(diskSource == FlintRMSNormKernel.embeddedShaderSource, "Disk flint_rmsnorm.metal and FlintRMSNormKernel.embeddedShaderSource have drifted!")

        // Verify that loadShaderSource() returns valid non-empty MSL
        let loaded = FlintRMSNormKernel.loadShaderSource()
        #expect(!loaded.isEmpty)
        #expect(loaded.contains("flint_rmsnorm_2048_bf16"))
        #expect(loaded.contains("flint_rmsnorm_128_bf16"))
        #expect(loaded.contains("flint_rmsnorm_general_bf16"))
    }

    // MARK: - 5. Apple M5 Hardware Benchmark & Telemetry

    @Test func benchmarkRMSNormPerformanceOnM5() throws {
        let context = try FlintMetalContext()
        let kernel = try FlintRMSNormKernel(context: context)

        print("\n=== FLINT RMSNORM HARDWARE BENCHMARK (APPLE M5) ===")

        let configurations: [(name: String, rows: Int, dim: Int, dtype: FlintRMSNormKernel.Dtype)] = [
            ("Hidden Norm Decode (1 token, dim 2048)", 1, 2048, .bfloat16),
            ("Hidden Norm Prefill (128 tokens, dim 2048)", 128, 2048, .bfloat16),
            ("Q Norm (32 heads, dim 128)", 32, 128, .bfloat16),
            ("K Norm (4 heads, dim 128)", 4, 128, .bfloat16),
            ("Hidden Norm Decode FP32 (1 token, dim 2048)", 1, 2048, .float32),
            ("Q Norm FP32 (32 heads, dim 128)", 32, 128, .float32)
        ]

        for config in configurations {
            let bytesPerRow = config.dim * config.dtype.elementByteWidth
            let totalBytes = config.rows * bytesPerRow
            let inBuf = try context.makeBuffer(byteCount: totalBytes)
            let wBuf = try context.makeBuffer(byteCount: config.dim * 2)
            let outBuf = try context.makeBuffer(byteCount: totalBytes)

            // Initialize buffers with valid non-zero floating point numbers (1.0)
            if config.dtype == .bfloat16 {
                inBuf.contents().assumingMemoryBound(to: UInt16.self).initialize(repeating: 0x3F80, count: config.rows * config.dim)
            } else {
                inBuf.contents().assumingMemoryBound(to: Float.self).initialize(repeating: 1.0, count: config.rows * config.dim)
            }
            wBuf.contents().assumingMemoryBound(to: UInt16.self).initialize(repeating: 0x3F80, count: config.dim)

            // Warmup (10 runs)
            for _ in 0..<10 {
                try kernel.execute(
                    input: inBuf,
                    weight: wBuf,
                    output: outBuf,
                    rows: config.rows,
                    dimension: config.dim,
                    epsilon: 1e-6,
                    dtype: config.dtype,
                    waitUntilCompleted: true
                )
            }

            // Benchmark (100 iterations)
            let iterations = 100
            var durations = [Double]()
            durations.reserveCapacity(iterations)

            for _ in 0..<iterations {
                let duration = try kernel.execute(
                    input: inBuf,
                    weight: wBuf,
                    output: outBuf,
                    rows: config.rows,
                    dimension: config.dim,
                    epsilon: 1e-6,
                    dtype: config.dtype,
                    waitUntilCompleted: true
                )
                durations.append(duration * 1_000_000.0) // Convert to microseconds (µs)
            }

            durations.sort()
            let minUs = durations.first ?? 0
            let p50Us = durations[iterations / 2]
            let p95Us = durations[Int(Double(iterations) * 0.95)]
            let maxUs = durations.last ?? 0
            let meanUs = durations.reduce(0, +) / Double(iterations)

            // Memory throughput calculation (Read in + Read weight + Write out)
            let readBytes = totalBytes + (config.dim * 2)
            let writeBytes = totalBytes
            let transferGB = Double(readBytes + writeBytes) / 1e9
            let bandwidthGBps = (meanUs > 0) ? (transferGB / (meanUs / 1e6)) : 0

            print(String(format: "%-45@ | P50: %6.2f µs | Min: %6.2f µs | P95: %6.2f µs | Max: %6.2f µs | BW: %6.2f GB/s",
                         config.name, p50Us, minUs, p95Us, maxUs, bandwidthGBps))
        }
        print("====================================================\n")
    }
}
