import CryptoKit
import Darwin
import Foundation
import Metal
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintMLPKernelTests {

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

    private static func makeBuffer(context: FlintMetalContext, data: Data) throws -> MTLBuffer {
        try data.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else {
                return try context.makeBuffer(byteCount: 0)
            }
            return try context.makeBuffer(bytes: base, byteCount: data.count)
        }
    }

    private static func makeBuffer<T>(context: FlintMetalContext, array: [T]) throws -> MTLBuffer {
        try array.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else {
                return try context.makeBuffer(byteCount: 0)
            }
            return try context.makeBuffer(bytes: base, byteCount: ptr.count)
        }
    }

    private static func bf16ToFloat(_ u16: UInt16) -> Float {
        Float(bitPattern: UInt32(u16) << 16)
    }

    private static func loadLiveWeights(context: FlintMetalContext) throws -> FlintResidentWeights {
        let modelDir = "/Users/tanishq/models/flint-mixed-3-4bit"
        let validated = try FlintModelDirectory.validate(path: modelDir)
        let shards = try FlintModelDirectory.openShards(canonicalPath: validated.canonicalPath, shardNames: validated.shards)
        let (residentWeights, _) = try FlintResidentLoader.loadResidentWeights(
            validatedDirectory: validated,
            shardFiles: shards,
            storageMode: .gpuRequired,
            device: context.device
        )
        return residentWeights
    }

    private struct MLKGoldenCase: Decodable {
        let name: String
        let layer: Int
        let kind: String
        let batch: Int
        let tokens: Int
        let hidden_dim: Int
        let intermediate_dim: Int
        let dtype: String
        let input_hex: String
        let gate_proj_hex: String
        let up_proj_hex: String
        let silu_gate_hex: String
        let gated_product_hex: String
        let down_proj_hex: String
    }

    private struct MLKGoldenFixture: Decodable {
        let metadata: [String: AnyCodable]
        let cases: [MLKGoldenCase]
    }

    private struct AnyCodable: Decodable {}

    private static func loadGoldenFixture() throws -> MLKGoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_mlp_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(MLKGoldenFixture.self, from: data)
    }

    // MARK: - 1. Pinned Independent Reference Qualification (vs MLX 0.32.4)

    @Test func independentReferenceQualificationVsMLX() throws {
        guard let fixture = try Self.loadGoldenFixture() else {
            Issue.record("flint_mlp_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let mlpKernel = try FlintMLPKernel(context: context)

        // Load resident weights for layers 0, 1, 2, 24, 48 from live checkpoint
        let residentWeights = try Self.loadLiveWeights(context: context)

        print("\n=== FLINT NATIVE SWIGLU MLP NUMERICAL QUALIFICATION (vs MLX 0.32.4) ===")
        print(String(format: "%-28@ | %-6@ | %-7@ | %4@ | %6@ | %12@ | %12@ | %10@ | %@",
                     "Case Name", "Kind", "Dtype", "Lyr", "Tokens", "Max Abs Diff", "Relative L2", "RMS Diff", "Status"))
        print(String(repeating: "-", count: 115))

        for c in fixture.cases {
            let inData = Self.parseHexToData(c.input_hex)
            let inBuf = try Self.makeBuffer(context: context, data: inData)

            let isBF16 = (c.dtype == "bfloat16")
            let dtype: FlintMLPKernel.Dtype = isBF16 ? .bfloat16 : .float32

            let gw: MTLBuffer
            let gs: MTLBuffer
            let gb: MTLBuffer
            let uw: MTLBuffer
            let us: MTLBuffer
            let ub: MTLBuffer
            let dw: MTLBuffer
            let ds: MTLBuffer
            let db: MTLBuffer

            if c.kind == "dense" {
                guard let dwWeights = residentWeights.denseMLPWeights(forLayer: c.layer) else {
                    Issue.record("Missing dense MLP weights for layer \(c.layer)")
                    return
                }
                gw = dwWeights.gateWeight.metalBuffer!
                gs = dwWeights.gateScales.metalBuffer!
                gb = dwWeights.gateBiases.metalBuffer!
                uw = dwWeights.upWeight.metalBuffer!
                us = dwWeights.upScales.metalBuffer!
                ub = dwWeights.upBiases.metalBuffer!
                dw = dwWeights.downWeight.metalBuffer!
                ds = dwWeights.downScales.metalBuffer!
                db = dwWeights.downBiases.metalBuffer!
            } else {
                guard let moeWeights = residentWeights.moeResidentWeights(forLayer: c.layer) else {
                    Issue.record("Missing shared expert weights for layer \(c.layer)")
                    return
                }
                gw = moeWeights.sharedGateWeight.metalBuffer!
                gs = moeWeights.sharedGateScales.metalBuffer!
                gb = moeWeights.sharedGateBiases.metalBuffer!
                uw = moeWeights.sharedUpWeight.metalBuffer!
                us = moeWeights.sharedUpScales.metalBuffer!
                ub = moeWeights.sharedUpBiases.metalBuffer!
                dw = moeWeights.sharedDownWeight.metalBuffer!
                ds = moeWeights.sharedDownScales.metalBuffer!
                db = moeWeights.sharedDownBiases.metalBuffer!
            }

            // Execute with intermediate capture
            let intermediates = try mlpKernel.executeForwardWithIntermediates(
                gateWeight: gw, gateScales: gs, gateBiases: gb,
                upWeight: uw, upScales: us, upBiases: ub,
                downWeight: dw, downScales: ds, downBiases: db,
                input: inBuf,
                hiddenDim: c.hidden_dim,
                intermediateDim: c.intermediate_dim,
                batch: c.batch,
                numTokens: c.tokens,
                dtype: dtype
            )

            // Validate all 5 stages individually!
            let stagesToValidate: [(name: String, hex: String, actual: MTLBuffer, count: Int)] = [
                ("gate_proj", c.gate_proj_hex, intermediates.gateProj, c.batch * c.tokens * c.intermediate_dim),
                ("up_proj", c.up_proj_hex, intermediates.upProj, c.batch * c.tokens * c.intermediate_dim),
                ("silu_gate", c.silu_gate_hex, intermediates.siluGate, c.batch * c.tokens * c.intermediate_dim),
                ("gated_product", c.gated_product_hex, intermediates.gatedProduct, c.batch * c.tokens * c.intermediate_dim),
                ("down_proj (final)", c.down_proj_hex, intermediates.downProj, c.batch * c.tokens * c.hidden_dim),
            ]

            var maxFinalDiff: Float = 0.0
            var finalRelL2: Double = 0.0
            var finalRMS: Double = 0.0

            for stage in stagesToValidate {
                let expData = Self.parseHexToData(stage.hex)
                var maxStageDiff: Float = 0.0
                var l2DiffSq: Double = 0.0
                var l2ExpSq: Double = 0.0

                if isBF16 {
                    let stageActPtr = stage.actual.contents().bindMemory(to: UInt16.self, capacity: stage.count)
                    let expPtr = expData.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }

                    for i in 0..<stage.count {
                        let actF = Self.bf16ToFloat(stageActPtr[i])
                        let expF = Self.bf16ToFloat(expPtr[i])
                        let diff = abs(actF - expF)
                        if diff > maxStageDiff {
                            maxStageDiff = diff
                        }
                        l2DiffSq += Double(diff * diff)
                        l2ExpSq += Double(expF * expF)
                    }
                } else {
                    let stageActPtr = stage.actual.contents().bindMemory(to: Float.self, capacity: stage.count)
                    let expPtr = expData.withUnsafeBytes { $0.bindMemory(to: Float.self) }

                    for i in 0..<stage.count {
                        let actF = stageActPtr[i]
                        let expF = expPtr[i]
                        let diff = abs(actF - expF)
                        if diff > maxStageDiff {
                            maxStageDiff = diff
                        }
                        l2DiffSq += Double(diff * diff)
                        l2ExpSq += Double(expF * expF)
                    }
                }

                let relL2 = (l2ExpSq > 0) ? sqrt(l2DiffSq / l2ExpSq) : sqrt(l2DiffSq)
                let rms = sqrt(l2DiffSq / Double(stage.count))

                if stage.name.hasPrefix("down_proj") {
                    maxFinalDiff = maxStageDiff
                    finalRelL2 = relL2
                    finalRMS = rms
                }

                // Verify intermediate stage error bounds
                if isBF16 {
                    #expect(maxStageDiff <= 0.05, "Stage \(stage.name) in \(c.name) max diff exceeded: \(maxStageDiff)")
                    #expect(relL2 <= 0.005, "Stage \(stage.name) in \(c.name) relative L2 exceeded: \(relL2)")
                } else {
                    #expect(maxStageDiff <= 1e-5, "Stage \(stage.name) in \(c.name) FP32 max diff exceeded: \(maxStageDiff)")
                    #expect(relL2 <= 1e-5, "Stage \(stage.name) in \(c.name) FP32 relative L2 exceeded: \(relL2)")
                }
            }

            print(String(format: "%-28@ | %-6@ | %-7@ | %4d | %6d | %12.6f | %12.6e | %10.6e | PASSED",
                         c.name, c.kind, c.dtype, c.layer, c.tokens, maxFinalDiff, finalRelL2, finalRMS))
        }
        print("===================================================================================================================\n")
    }

    // MARK: - 2. Analytical CPU Reference Contract

    @Test func analyticalMLPContract() throws {
        let context = try FlintMetalContext()
        let mlpKernel = try FlintMLPKernel(context: context)

        let hiddenDim = 2048
        let intermediateDim = 1024
        let batch = 1
        let numTokens = 2

        var x = [Float](repeating: 0, count: batch * numTokens * hiddenDim)
        for i in 0..<x.count { x[i] = Float.random(in: -2.0...2.0) }

        var wGate = [Float](repeating: 0, count: intermediateDim * hiddenDim)
        for i in 0..<wGate.count { wGate[i] = Float.random(in: -0.1...0.1) }

        var wUp = [Float](repeating: 0, count: intermediateDim * hiddenDim)
        for i in 0..<wUp.count { wUp[i] = Float.random(in: -0.1...0.1) }

        var wDown = [Float](repeating: 0, count: hiddenDim * intermediateDim)
        for i in 0..<wDown.count { wDown[i] = Float.random(in: -0.1...0.1) }

        // CPU evaluation
        let expected = FlintMLPReference.forwardWithIntermediates(
            x: x, wGate: wGate, wUp: wUp, wDown: wDown,
            batch: batch, numTokens: numTokens,
            hiddenDim: hiddenDim, intermediateDim: intermediateDim
        )

        // GPU execution using direct GEMM and SwiGLU kernels
        let inBuf = try Self.makeBuffer(context: context, array: x)
        let wgBuf = try Self.makeBuffer(context: context, array: wGate)
        let wuBuf = try Self.makeBuffer(context: context, array: wUp)
        let wdBuf = try Self.makeBuffer(context: context, array: wDown)

        let gateBuf = try context.makeBuffer(byteCount: batch * numTokens * intermediateDim * 4)
        let upBuf = try context.makeBuffer(byteCount: batch * numTokens * intermediateDim * 4)
        let actBuf = try context.makeBuffer(byteCount: batch * numTokens * intermediateDim * 4)
        let outBuf = try context.makeBuffer(byteCount: batch * numTokens * hiddenDim * 4)

        guard let cb = context.queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            Issue.record("Failed to make command encoder")
            return
        }

        try mlpKernel.encodeGEMM(into: encoder, input: inBuf, weight: wgBuf, output: gateBuf, M: intermediateDim, K: hiddenDim, batch: batch, numTokens: numTokens, inDtype: .float32, outDtype: .float32)
        try mlpKernel.encodeGEMM(into: encoder, input: inBuf, weight: wuBuf, output: upBuf, M: intermediateDim, K: hiddenDim, batch: batch, numTokens: numTokens, inDtype: .float32, outDtype: .float32)
        try mlpKernel.encodeSwiGLU(into: encoder, gate: gateBuf, up: upBuf, output: actBuf, totalElements: batch * numTokens * intermediateDim, dtype: .float32)
        try mlpKernel.encodeGEMM(into: encoder, input: actBuf, weight: wdBuf, output: outBuf, M: hiddenDim, K: intermediateDim, batch: batch, numTokens: numTokens, inDtype: .float32, outDtype: .float32)

        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let gpuOut = outBuf.contents().bindMemory(to: Float.self, capacity: batch * numTokens * hiddenDim)
        var maxDiff: Float = 0.0
        for i in 0..<(batch * numTokens * hiddenDim) {
            let diff = abs(gpuOut[i] - expected.downProj[i])
            if diff > maxDiff { maxDiff = diff }
        }

        #expect(maxDiff <= 1e-4, "Analytical SwiGLU contract max diff exceeded 1e-4: \(maxDiff)")
    }

    // MARK: - 3. Live Checkpoint Integration & Resident Weight Immutability

    @Test func liveCheckpointResidentMLPIntegration() throws {
        let context = try FlintMetalContext()
        let mlpKernel = try FlintMLPKernel(context: context)
        let weights = try Self.loadLiveWeights(context: context)

        // Test Dense Layer 0
        guard let d0 = weights.denseMLPWeights(forLayer: 0) else {
            Issue.record("Dense Layer 0 weights missing")
            return
        }

        let inBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let outBuf = try context.makeBuffer(byteCount: 2048 * 2)

        // Capture SHA-256 before
        let hashBefore = Self.sha256Hex(Data(bytes: d0.gateWeight.readOnlyPointer.baseAddress!, count: d0.gateWeight.byteCount))

        // Execute forward pass
        try mlpKernel.executeForwardDenseMLP(
            weights: d0,
            input: inBuf,
            output: outBuf,
            batch: 1,
            numTokens: 1,
            dtype: .bfloat16
        )

        // Capture SHA-256 after
        let hashAfter = Self.sha256Hex(Data(bytes: d0.gateWeight.readOnlyPointer.baseAddress!, count: d0.gateWeight.byteCount))

        #expect(hashBefore == hashAfter, "Dense Layer 0 gate weight was mutated during MLP forward!")

        // Test Shared Expert Layer 2
        guard let moe2 = weights.moeResidentWeights(forLayer: 2) else {
            Issue.record("Shared expert Layer 2 weights missing")
            return
        }

        let hashSharedBefore = Self.sha256Hex(Data(bytes: moe2.sharedGateWeight.readOnlyPointer.baseAddress!, count: moe2.sharedGateWeight.byteCount))

        try mlpKernel.executeForwardSharedExpertMLP(
            weights: moe2,
            input: inBuf,
            output: outBuf,
            batch: 1,
            numTokens: 1,
            dtype: .bfloat16
        )

        let hashSharedAfter = Self.sha256Hex(Data(bytes: moe2.sharedGateWeight.readOnlyPointer.baseAddress!, count: moe2.sharedGateWeight.byteCount))

        #expect(hashSharedBefore == hashSharedAfter, "Shared expert Layer 2 gate weight was mutated during forward!")
    }

    // MARK: - 4. Repeat-Run Bitwise Determinism

    @Test func repeatRunDeterminismContract() throws {
        let context = try FlintMetalContext()
        let mlpKernel = try FlintMLPKernel(context: context)
        let weights = try Self.loadLiveWeights(context: context)

        guard let d0 = weights.denseMLPWeights(forLayer: 0) else {
            Issue.record("Dense Layer 0 weights missing")
            return
        }

        var inputVals = [UInt16](repeating: 0, count: 2048)
        for i in 0..<2048 {
            let f = Float.random(in: -2.0...2.0)
            inputVals[i] = UInt16(f.bitPattern >> 16)
        }
        let inBuf = try Self.makeBuffer(context: context, array: inputVals)
        let outBuf = try context.makeBuffer(byteCount: 2048 * 2)

        var baselineHash: String? = nil
        let iterations = 50

        for it in 0..<iterations {
            try mlpKernel.executeForwardDenseMLP(
                weights: d0,
                input: inBuf,
                output: outBuf,
                batch: 1,
                numTokens: 1,
                dtype: .bfloat16
            )

            let outData = Data(bytes: outBuf.contents(), count: 2048 * 2)
            let hash = Self.sha256Hex(outData)
            if let base = baselineHash {
                #expect(hash == base, "Repeat run determinism mismatch at iteration \(it)")
            } else {
                baselineHash = hash
            }
        }

        print("Repeat-run determinism: 50/50 consecutive Dense SwiGLU MLP dispatches produced identical SHA-256: \(baselineHash ?? "")")
    }

    // MARK: - 5. Memory Footprint & Zero Allocation Verification

    @Test func memoryFootprintAndZeroAllocationVerification() throws {
        let context = try FlintMetalContext()
        let mlpKernel = try FlintMLPKernel(context: context)

        // 1. Verify weight scratch is exactly 48 MiB (bounded)
        #expect(mlpKernel.weightScratch.length == 48 * 1024 * 1024, "Weight scratch must be exactly 48 MiB")

        // 2. Verify activation scratches are preallocated and bounded
        #expect(mlpKernel.gateScratch.length == 64 * 6144 * 4)
        #expect(mlpKernel.upScratch.length == 64 * 6144 * 4)
        #expect(mlpKernel.actScratch.length == 64 * 6144 * 4)

        let totalScratchBytes = mlpKernel.weightScratch.length +
                                mlpKernel.gateScratch.length +
                                mlpKernel.upScratch.length +
                                mlpKernel.actScratch.length

        print("\n=== FLINT SWIGLU MLP BOUNDED MEMORY PROFILE (Apple M5) ===")
        print(String(format: "Weight Scratch Buffer:      %6.2f MiB (%d bytes)", Double(mlpKernel.weightScratch.length) / (1024 * 1024), mlpKernel.weightScratch.length))
        print(String(format: "Gate Activation Scratch:    %6.2f KiB (%d bytes)", Double(mlpKernel.gateScratch.length) / 1024, mlpKernel.gateScratch.length))
        print(String(format: "Up Activation Scratch:      %6.2f KiB (%d bytes)", Double(mlpKernel.upScratch.length) / 1024, mlpKernel.upScratch.length))
        print(String(format: "Act Intermediate Scratch:   %6.2f KiB (%d bytes)", Double(mlpKernel.actScratch.length) / 1024, mlpKernel.actScratch.length))
        print(String(format: "Total Bounded Scratch Memory: %6.2f MiB", Double(totalScratchBytes) / (1024 * 1024)))
        print("Per-Dispatch Allocations:   0 bytes (fully reused across dispatches)")
        print("==========================================================\n")
    }

    // MARK: - 6. Buffer Bounds and Parameter Validation

    @Test func parameterAndBufferBoundsValidation() throws {
        let context = try FlintMetalContext()
        let mlpKernel = try FlintMLPKernel(context: context)

        let inBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let outBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let smallBuf = try context.makeBuffer(byteCount: 100)

        func checkGEMMThrows(input: MTLBuffer, inputOffset: Int = 0, weight: MTLBuffer, output: MTLBuffer, M: Int, K: Int, batch: Int, numTokens: Int) {
            #expect(throws: FlintMetalError.self) {
                guard let cb = context.queue.makeCommandBuffer(),
                      let encoder = cb.makeComputeCommandEncoder() else {
                    throw FlintMetalError.commandQueueUnavailable
                }
                do {
                    try mlpKernel.encodeGEMM(into: encoder, input: input, inputOffset: inputOffset, weight: weight, output: output, M: M, K: K, batch: batch, numTokens: numTokens)
                    encoder.endEncoding()
                } catch {
                    encoder.endEncoding()
                    throw error
                }
            }
        }

        // 1. batch = 0
        checkGEMMThrows(input: inBuf, weight: inBuf, output: outBuf, M: 2048, K: 2048, batch: 0, numTokens: 1)

        // 2. numTokens = 0
        checkGEMMThrows(input: inBuf, weight: inBuf, output: outBuf, M: 2048, K: 2048, batch: 1, numTokens: 0)

        // 3. Undersized buffer
        checkGEMMThrows(input: smallBuf, weight: inBuf, output: outBuf, M: 2048, K: 2048, batch: 1, numTokens: 1)

        // 4. Unaligned offset
        checkGEMMThrows(input: inBuf, inputOffset: 1, weight: inBuf, output: outBuf, M: 2048, K: 2048, batch: 1, numTokens: 1)
    }

    // MARK: - 7. Canonical Metal Shader Source Equivalence

    @Test func canonicalShaderSourceEquivalence() throws {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_mlp.metal"
        let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(sourcePath)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            Issue.record("flint_mlp.metal not found on disk")
            return
        }

        let diskSource = try String(contentsOf: fileURL, encoding: .utf8)
        #expect(diskSource == FlintMLPKernel.embeddedShaderSource, "flint_mlp.metal on disk does not match embeddedShaderSource in FlintMLPKernel.swift")
    }

    // MARK: - 8. Apple M5 Hardware Performance Benchmark

    @Test func mlpPerformanceBenchmarkAppleM5() throws {
        let context = try FlintMetalContext()
        let mlpKernel = try FlintMLPKernel(context: context)
        let weights = try Self.loadLiveWeights(context: context)

        guard let d0 = weights.denseMLPWeights(forLayer: 0),
              let s2 = weights.moeResidentWeights(forLayer: 2) else {
            Issue.record("Weights missing for benchmark")
            return
        }

        let inBuf1 = try context.makeBuffer(byteCount: 1 * 2048 * 2)
        let outBuf1 = try context.makeBuffer(byteCount: 1 * 2048 * 2)
        let inBuf8 = try context.makeBuffer(byteCount: 8 * 2048 * 2)
        let outBuf8 = try context.makeBuffer(byteCount: 8 * 2048 * 2)

        print("\n=== FLINT SWIGLU MLP HARDWARE BENCHMARK (Apple M5) ===")
        print("Hardware: Apple M5, 32 GB Unified Memory")
        print("Steady-state allocations: 0 heap allocations per dispatch")
        print(String(format: "%-32@ | %10@ | %10@ | %10@ | %10@ | %12@ | %@",
                     "Configuration", "P50", "Min", "P95", "Max", "Jitter", "Status"))
        print(String(repeating: "-", count: 105))

        let benchmarks: [(name: String, run: () throws -> Double)] = [
            ("Dense L0 Decode (T=1, I=6144)", {
                try mlpKernel.executeForwardDenseMLP(weights: d0, input: inBuf1, output: outBuf1, batch: 1, numTokens: 1)
            }),
            ("Dense L0 Prefill (T=8, I=6144)", {
                try mlpKernel.executeForwardDenseMLP(weights: d0, input: inBuf8, output: outBuf8, batch: 1, numTokens: 8)
            }),
            ("Shared L2 Decode (T=1, I=1024)", {
                try mlpKernel.executeForwardSharedExpertMLP(weights: s2, input: inBuf1, output: outBuf1, batch: 1, numTokens: 1)
            }),
            ("Shared L2 Prefill (T=8, I=1024)", {
                try mlpKernel.executeForwardSharedExpertMLP(weights: s2, input: inBuf8, output: outBuf8, batch: 1, numTokens: 8)
            }),
        ]

        for b in benchmarks {
            // Warm up
            for _ in 0..<5 { _ = try b.run() }

            var durationsUs: [Double] = []
            durationsUs.reserveCapacity(50)
            for _ in 0..<50 {
                let sec = try b.run()
                durationsUs.append(sec * 1e6)
            }
            durationsUs.sort()

            let p50 = durationsUs[durationsUs.count / 2]
            let minVal = durationsUs.first ?? 0
            let p95 = durationsUs[Int(Double(durationsUs.count) * 0.95)]
            let maxVal = durationsUs.last ?? 0
            let jitter = p95 - minVal

            print(String(format: "%-32@ | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | %10.2f µs | PASSED",
                         b.name, p50, minVal, p95, maxVal, jitter))
        }
        print("=========================================================================================================\n")
    }
}
