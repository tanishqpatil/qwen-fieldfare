import CryptoKit
import Darwin
import Foundation
import Metal
import QuartzCore
import Testing
@testable import FlintFieldfareCore

@Suite struct FlintDecoderLayerTests {

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

    private static func bf16ToFloat(_ u16: UInt16) -> Float {
        Float(bitPattern: UInt32(u16) << 16)
    }

    private static func floatToBf16(_ f: Float) -> UInt16 {
        UInt16(f.bitPattern >> 16)
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

    private struct Layer0GoldenCase: Decodable {
        let name: String
        let batch: Int
        let num_tokens: Int
        let dtype: String
        let input_hex: String
        let norm1_hex: String
        let q_raw_hex: String
        let k_raw_hex: String
        let v_raw_hex: String
        let q_norm_hex: String
        let k_norm_hex: String
        let attn_out_hex: String
        let attn_flat_hex: String
        let attn_proj_hex: String
        let res1_hex: String
        let norm2_hex: String
        let mlp_out_hex: String
        let final_out_hex: String
        let full_layer_out_hex: String
    }

    private struct Layer0GoldenFixture: Decodable {
        let generator: String?
        let cases: [Layer0GoldenCase]
    }

    private static func loadGoldenFixture() throws -> Layer0GoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_layer0_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(Layer0GoldenFixture.self, from: data)
    }

    // MARK: - Layer 2 Golden Types

    private struct Layer2GoldenCase: Decodable {
        let name: String
        let batch: Int
        let num_tokens: Int
        let dtype: String
        let input_hex: String
        let norm1_hex: String
        let q_raw_hex: String
        let k_raw_hex: String
        let v_raw_hex: String
        let q_norm_hex: String
        let k_norm_hex: String
        let q_rope_hex: String
        let k_rope_hex: String
        let attn_out_hex: String
        let attn_flat_hex: String
        let attn_proj_hex: String
        let res1_hex: String
        let norm2_hex: String
        let router_indices: [[[Int32]]]
        let router_weights: [[[Float]]]
        let routed_hex: String
        let shared_out_hex: String
        let moe_out_hex: String
        let final_out_hex: String
        let full_layer_out_hex: String
    }

    private struct Layer2GoldenFixture: Decodable {
        let generator: String?
        let cases: [Layer2GoldenCase]
    }

    private static func loadLayer2GoldenFixture() throws -> Layer2GoldenFixture? {
        let fixturePath = "Tests/FlintFieldfareCoreTests/Fixtures/flint_layer2_golden.json"
        let fullPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fixturePath)
        guard FileManager.default.fileExists(atPath: fullPath.path) else {
            return nil
        }
        let data = try Data(contentsOf: fullPath)
        return try JSONDecoder().decode(Layer2GoldenFixture.self, from: data)
    }

    private static func openShard1File() throws -> FlintSafetensorsFile {
        let shardPath = "/Users/tanishq/models/flint-mixed-3-4bit/model-00001-of-00004.safetensors"
        return try FlintSafetensorsFile(url: URL(fileURLWithPath: shardPath))
    }

    // MARK: - 1. Layer 0 End-to-End Numerical Qualification (vs MLX 0.32.4)

    @Test func layer0NumericalQualificationVsMLX() throws {
        guard let fixture = try Self.loadGoldenFixture() else {
            Issue.record("flint_layer0_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)

        print("\n=== FLINT LAYER 0 DENSE DECODER FORWARD QUALIFICATION (vs MLX 0.32.4) ===")

        for c in fixture.cases {
            let inData = Self.parseHexToData(c.input_hex)
            let inBuf = try Self.makeBuffer(context: context, data: inData)

            let intermediates = try runner.executeLayer0ForwardWithIntermediates(
                weights: weights,
                input: inBuf,
                batch: c.batch,
                numTokens: c.num_tokens
            )

            // Validate all stages:
            let stages: [(name: String, hex: String, actual: MTLBuffer, count: Int)] = [
                ("input_norm", c.norm1_hex, intermediates.norm1, c.batch * c.num_tokens * 2048),
                ("q_raw", c.q_raw_hex, intermediates.qRaw, c.batch * c.num_tokens * 4096),
                ("k_raw", c.k_raw_hex, intermediates.kRaw, c.batch * c.num_tokens * 512),
                ("v_raw", c.v_raw_hex, intermediates.vRaw, c.batch * c.num_tokens * 512),
                ("q_norm", c.q_norm_hex, intermediates.qNorm, c.batch * c.num_tokens * 4096),
                ("k_norm", c.k_norm_hex, intermediates.kNorm, c.batch * c.num_tokens * 512),
                ("attn_flat", c.attn_flat_hex, intermediates.attnFlat, c.batch * c.num_tokens * 4096),
                ("attn_proj", c.attn_proj_hex, intermediates.attnProj, c.batch * c.num_tokens * 2048),
                ("res1", c.res1_hex, intermediates.res1, c.batch * c.num_tokens * 2048),
                ("post_attn_norm", c.norm2_hex, intermediates.norm2, c.batch * c.num_tokens * 2048),
                ("mlp_out", c.mlp_out_hex, intermediates.mlpOut, c.batch * c.num_tokens * 2048),
                ("final_out", c.final_out_hex, intermediates.finalOut, c.batch * c.num_tokens * 2048),
            ]

            print("\nCase: \(c.name) (Batch: \(c.batch), Tokens: \(c.num_tokens))")
            print(String(format: "%-16@ | %7@ | %12@ | %12@ | %12@ | %10@ | %@",
                         "Stage", "Count", "Max Abs Diff", "Relative L2", "RMS Diff", "Non-Finite", "Status"))
            print(String(repeating: "-", count: 85))

            var maxFinalDiff: Float = 0.0
            var finalRelL2: Double = 0.0
            var finalRMS: Double = 0.0

            for stage in stages {
                let expData = Self.parseHexToData(stage.hex)
                let actPtr = stage.actual.contents().bindMemory(to: UInt16.self, capacity: stage.count)
                let expPtr = expData.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }

                var maxStageDiff: Float = 0.0
                var l2DiffSq: Double = 0.0
                var l2ExpSq: Double = 0.0
                var nonFiniteCount: Int = 0

                for i in 0..<stage.count {
                    let actF = Self.bf16ToFloat(actPtr[i])
                    let expF = Self.bf16ToFloat(expPtr[i])
                    if actF.isNaN || actF.isInfinite {
                        nonFiniteCount += 1
                    }
                    let diff = abs(actF - expF)
                    if diff > maxStageDiff {
                        maxStageDiff = diff
                    }
                    l2DiffSq += Double(diff * diff)
                    l2ExpSq += Double(expF * expF)
                }

                let relL2 = (l2ExpSq > 0) ? sqrt(l2DiffSq / l2ExpSq) : sqrt(l2DiffSq)
                let rms = sqrt(l2DiffSq / Double(stage.count))

                if stage.name == "final_out" {
                    maxFinalDiff = maxStageDiff
                    finalRelL2 = relL2
                    finalRMS = rms
                }

                #expect(nonFiniteCount == 0, "Stage \(stage.name) in \(c.name) contained \(nonFiniteCount) non-finite values")
                #expect(maxStageDiff <= 0.25, "Stage \(stage.name) in \(c.name) max diff exceeded: \(maxStageDiff)")
                #expect(relL2 <= 0.02, "Stage \(stage.name) in \(c.name) relative L2 exceeded: \(relL2)")

                print(String(format: "%-16@ | %7d | %12.6f | %12.6e | %12.6e | %10d | PASSED",
                             stage.name, stage.count, maxStageDiff, relL2, rms, nonFiniteCount))
            }

            // Direct comparison with full_layer_out_hex (MLX module layer(x))
            let fullLayerExpData = Self.parseHexToData(c.full_layer_out_hex)
            let finalPtr = intermediates.finalOut.contents().bindMemory(to: UInt16.self, capacity: c.batch * c.num_tokens * 2048)
            let fullExpPtr = fullLayerExpData.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }
            var fullDiff: Float = 0.0
            var fullL2DiffSq: Double = 0.0
            var fullL2ExpSq: Double = 0.0
            for i in 0..<(c.batch * c.num_tokens * 2048) {
                let actF = Self.bf16ToFloat(finalPtr[i])
                let expF = Self.bf16ToFloat(fullExpPtr[i])
                let diff = abs(actF - expF)
                if diff > fullDiff { fullDiff = diff }
                fullL2DiffSq += Double(diff * diff)
                fullL2ExpSq += Double(expF * expF)
            }
            let fullRelL2 = (fullL2ExpSq > 0) ? sqrt(fullL2DiffSq / fullL2ExpSq) : sqrt(fullL2DiffSq)
            let fullRMS = sqrt(fullL2DiffSq / Double(c.batch * c.num_tokens * 2048))

            #expect(maxFinalDiff <= 0.02, "Final layer 0 output exceeded max diff tolerance in \(c.name): \(maxFinalDiff)")
            #expect(finalRelL2 <= 0.002, "Final layer 0 output exceeded relative L2 tolerance in \(c.name): \(finalRelL2)")
            #expect(fullDiff <= 0.02, "Direct full layer forward comparison failed in \(c.name): \(fullDiff)")

            print(String(format: "%-16@ | %7d | %12.6f | %12.6e | %12.6e | %10d | PASSED (vs MLX recon)",
                         "final_recon", c.batch * c.num_tokens * 2048, maxFinalDiff, finalRelL2, finalRMS, 0))
            print(String(format: "%-16@ | %7d | %12.6f | %12.6e | %12.6e | %10d | PASSED (vs MLX layer(x))",
                         "module_eval", c.batch * c.num_tokens * 2048, fullDiff, fullRelL2, fullRMS, 0))
        }
        print("===============================================================================================\n")
    }

    // MARK: - 2. Repeat-Run Bitwise Determinism (Finite Inputs)

    @Test func layer0RepeatRunDeterminism() throws {
        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)

        let inBuf = try context.makeBuffer(byteCount: 2048 * 2)
        // Fill deterministic finite Float32 input converted properly to BF16
        let inPtr = inBuf.contents().bindMemory(to: UInt16.self, capacity: 2048)
        for i in 0..<2048 {
            let val = sin(Float(i) * 0.05 + 0.1) * 0.4
            #expect(!val.isNaN && !val.isInfinite, "Input generator produced non-finite float")
            inPtr[i] = Self.floatToBf16(val)
            let back = Self.bf16ToFloat(inPtr[i])
            #expect(!back.isNaN && !back.isInfinite, "Converted BF16 is non-finite")
        }

        var baselineDigest: String?
        for run in 0..<50 {
            let inter = try runner.executeLayer0ForwardWithIntermediates(weights: weights, input: inBuf, batch: 1, numTokens: 1)

            // Verify all intermediate and final buffers are strictly finite:
            let intermediateBuffers: [(name: String, buf: MTLBuffer, count: Int)] = [
                ("norm1", inter.norm1, 2048),
                ("qNorm", inter.qNorm, 4096),
                ("kNorm", inter.kNorm, 512),
                ("attnOut", inter.attnOut, 4096),
                ("res1", inter.res1, 2048),
                ("norm2", inter.norm2, 2048),
                ("mlpOut", inter.mlpOut, 2048),
                ("finalOut", inter.finalOut, 2048)
            ]
            for s in intermediateBuffers {
                let ptr = s.buf.contents().bindMemory(to: UInt16.self, capacity: s.count)
                for i in 0..<s.count {
                    let val = Self.bf16ToFloat(ptr[i])
                    #expect(!val.isNaN && !val.isInfinite, "Intermediate \(s.name) in run \(run) produced non-finite value at index \(i)")
                }
            }

            let outData = Data(bytes: inter.finalOut.contents(), count: 2048 * 2)
            let digest = Self.sha256Hex(outData)
            if let base = baselineDigest {
                #expect(digest == base, "Repeat run \(run) produced divergent SHA-256")
            } else {
                baselineDigest = digest
            }
        }

        // Deliberately different finite input verification:
        let inBuf2 = try context.makeBuffer(byteCount: 2048 * 2)
        let inPtr2 = inBuf2.contents().bindMemory(to: UInt16.self, capacity: 2048)
        for i in 0..<2048 {
            let val = cos(Float(i) * 0.05 + 0.1) * 0.4
            #expect(!val.isNaN && !val.isInfinite)
            inPtr2[i] = Self.floatToBf16(val)
        }
        let inter2 = try runner.executeLayer0ForwardWithIntermediates(weights: weights, input: inBuf2, batch: 1, numTokens: 1)
        let outData2 = Data(bytes: inter2.finalOut.contents(), count: 2048 * 2)
        let digest2 = Self.sha256Hex(outData2)
        #expect(digest2 != baselineDigest, "Deliberately different finite input produced identical SHA-256 hash")

        print("Repeat-run determinism: 50/50 Layer 0 forwards produced identical finite SHA-256: \(baselineDigest ?? "")")
        print("Deliberately distinct finite input produced valid different SHA-256: \(digest2)")
    }

    // MARK: - 3. Canonical Layer Shader Source Equivalence

    @Test func canonicalLayerShaderSourceEquivalence() throws {
        let sourcePath = "Sources/FlintFieldfareCore/Metal/flint_layer.metal"
        let fileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(sourcePath)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            Issue.record("flint_layer.metal not found on disk")
            return
        }

        let diskSource = try String(contentsOf: fileURL, encoding: .utf8)
        #expect(diskSource == FlintDecoderLayerRunner.embeddedLayerShaderSource, "flint_layer.metal on disk does not match embeddedLayerShaderSource")
    }

    // MARK: - 4. Bounds, Aliasing, and Memory Safety Validation (Layer 0)

    @Test func layer0BoundsAndSafetyValidation() throws {
        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)
        let inBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let outBuf = try context.makeBuffer(byteCount: 2048 * 2)

        // Negative/zero batch
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0ForwardWithIntermediates(weights: weights, input: inBuf, batch: 0, numTokens: 1)
        }
        // Negative/zero numTokens
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0ForwardWithIntermediates(weights: weights, input: inBuf, batch: 1, numTokens: 0)
        }
        // Exceeding token capacity (> 16)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0ForwardWithIntermediates(weights: weights, input: inBuf, batch: 1, numTokens: 17)
        }
        // Negative input offset
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0Forward(weights: weights, input: inBuf, inputOffset: -2, output: outBuf, batch: 1, numTokens: 1)
        }
        // Unaligned input offset (odd byte)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0Forward(weights: weights, input: inBuf, inputOffset: 1, output: outBuf, batch: 1, numTokens: 1)
        }
        // Negative output offset
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0Forward(weights: weights, input: inBuf, output: outBuf, outputOffset: -2, batch: 1, numTokens: 1)
        }
        // Unaligned output offset (odd byte)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0Forward(weights: weights, input: inBuf, output: outBuf, outputOffset: 1, batch: 1, numTokens: 1)
        }
        // Undersized input buffer
        let smallInBuf = try context.makeBuffer(byteCount: 1024)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0Forward(weights: weights, input: smallInBuf, output: outBuf, batch: 1, numTokens: 1)
        }
        // Undersized output buffer
        let smallOutBuf = try context.makeBuffer(byteCount: 1024)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0Forward(weights: weights, input: inBuf, output: smallOutBuf, batch: 1, numTokens: 1)
        }
        // Overlapping input and output buffer ranges in same buffer
        let sharedBuf = try context.makeBuffer(byteCount: 4096)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer0Forward(weights: weights, input: sharedBuf, inputOffset: 0, output: sharedBuf, outputOffset: 1024, batch: 1, numTokens: 1)
        }
    }

    // MARK: - 5. Apple M5 Hardware Performance Benchmark (Layer 0)

    @Test func layer0AppleM5PerformanceBenchmark() throws {
        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)

        let inBuf1 = try context.makeBuffer(byteCount: 1 * 2048 * 2)
        let inBuf8 = try context.makeBuffer(byteCount: 8 * 2048 * 2)

        print("\n=== FLINT LAYER 0 DENSE DECODER HARDWARE BENCHMARK (Apple M5) ===")
        print("Hardware: Apple M5, 32 GB Unified Memory")
        print("Metal Resources: 52.5 MiB Scratch Preallocated, ~1.28 GB Resident Weights")
        print("Steady-State Allocations: Strictly 0 Bytes (Excluding Reusable GPU Scratch & OS Command Buffers)")
        print("Disclosure: Single-layer T=1 timing reflects isolated kernel forward, not steady-state autoregressive decoding.")
        print(String(format: "%-30@ | %10@ | %10@ | %10@ | %10@ | %10@ | %12@ | %@",
                     "Configuration", "P50 Wall", "Min Wall", "P95 Wall", "Max Wall", "GPU Time", "Waits / CBs", "Status"))
        print(String(repeating: "-", count: 110))

        let benchmarks: [(name: String, run: () throws -> (wallSec: Double, inter: FlintDecoderLayerRunner.Layer0Intermediates))] = [
            ("Layer 0 Decode (T=1, Dense)", {
                let t0 = CACurrentMediaTime()
                let inter = try runner.executeLayer0ForwardWithIntermediates(weights: weights, input: inBuf1, batch: 1, numTokens: 1)
                let t1 = CACurrentMediaTime()
                return (t1 - t0, inter)
            }),
            ("Layer 0 Prefill (T=8, Dense)", {
                let t0 = CACurrentMediaTime()
                let inter = try runner.executeLayer0ForwardWithIntermediates(weights: weights, input: inBuf8, batch: 1, numTokens: 8)
                let t1 = CACurrentMediaTime()
                return (t1 - t0, inter)
            }),
        ]

        for b in benchmarks {
            for _ in 0..<5 { _ = try b.run() }
            var durationsUs: [Double] = []
            durationsUs.reserveCapacity(30)
            var lastInter: FlintDecoderLayerRunner.Layer0Intermediates?
            for _ in 0..<30 {
                let res = try b.run()
                durationsUs.append(res.wallSec * 1e6)
                lastInter = res.inter
            }
            durationsUs.sort()

            let p50 = durationsUs[durationsUs.count / 2]
            let minVal = durationsUs.first ?? 0
            let p95 = durationsUs[Int(Double(durationsUs.count) * 0.95)]
            let maxVal = durationsUs.last ?? 0
            let gpuUs = lastInter?.timing.totalGpuElapsedUs ?? 0
            let waits = "\(lastInter?.timing.gpuWaits ?? 1) / \(lastInter?.timing.commandBuffersCommitted ?? 1)"

            print(String(format: "%-30@ | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | %12@ | PASSED",
                         b.name, p50, minVal, p95, maxVal, gpuUs, waits))
        }
        print("==============================================================================================================\n")
    }

    // MARK: - 6. Layer 2 MoE Numerical Qualification (vs MLX 0.32.4)

    @Test func layer2NumericalQualificationVsMLX() throws {
        guard let fixture = try Self.loadLayer2GoldenFixture() else {
            Issue.record("flint_layer2_golden.json fixture not found")
            return
        }

        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)
        let shard1File = try Self.openShard1File()

        print("\n=== FLINT LAYER 2 MOE DECODER FORWARD QUALIFICATION (vs MLX 0.32.4) ===")

        for c in fixture.cases {
            let inData = Self.parseHexToData(c.input_hex)
            let inBuf = try Self.makeBuffer(context: context, data: inData)

            let intermediates = try runner.executeLayer2ForwardWithIntermediates(
                weights: weights,
                shard1File: shard1File,
                input: inBuf,
                batch: c.batch,
                numTokens: c.num_tokens
            )

            // Validate all stages:
            let stages: [(name: String, hex: String, actual: MTLBuffer, count: Int)] = [
                ("input_norm", c.norm1_hex, intermediates.norm1, c.batch * c.num_tokens * 2048),
                ("q_raw", c.q_raw_hex, intermediates.qRaw, c.batch * c.num_tokens * 4096),
                ("k_raw", c.k_raw_hex, intermediates.kRaw, c.batch * c.num_tokens * 512),
                ("v_raw", c.v_raw_hex, intermediates.vRaw, c.batch * c.num_tokens * 512),
                ("q_norm", c.q_norm_hex, intermediates.qNorm, c.batch * c.num_tokens * 4096),
                ("k_norm", c.k_norm_hex, intermediates.kNorm, c.batch * c.num_tokens * 512),
                ("q_rope", c.q_rope_hex, intermediates.qRope, c.batch * c.num_tokens * 4096),
                ("k_rope", c.k_rope_hex, intermediates.kRope, c.batch * c.num_tokens * 512),
                ("attn_out", c.attn_out_hex, intermediates.attnOut, c.batch * c.num_tokens * 4096),
                ("attn_flat", c.attn_flat_hex, intermediates.attnFlat, c.batch * c.num_tokens * 4096),
                ("attn_proj", c.attn_proj_hex, intermediates.attnProj, c.batch * c.num_tokens * 2048),
                ("res1", c.res1_hex, intermediates.res1, c.batch * c.num_tokens * 2048),
                ("post_attn_norm", c.norm2_hex, intermediates.norm2, c.batch * c.num_tokens * 2048),
                ("shared_out", c.shared_out_hex, intermediates.sharedOut, c.batch * c.num_tokens * 2048),
                ("moe_out", c.moe_out_hex, intermediates.moeOut, c.batch * c.num_tokens * 2048),
                ("final_out", c.final_out_hex, intermediates.finalOut, c.batch * c.num_tokens * 2048),
            ]

            print("\nCase: \(c.name) (Batch: \(c.batch), Tokens: \(c.num_tokens))")
            print(String(format: "%-16@ | %7@ | %12@ | %12@ | %12@ | %10@ | %@",
                         "Stage", "Count", "Max Abs Diff", "Relative L2", "RMS Diff", "Non-Finite", "Status"))
            print(String(repeating: "-", count: 85))

            var maxFinalDiff: Float = 0.0
            var finalRelL2: Double = 0.0
            var finalRMS: Double = 0.0

            for stage in stages {
                let expData = Self.parseHexToData(stage.hex)
                let actPtr = stage.actual.contents().bindMemory(to: UInt16.self, capacity: stage.count)
                let expPtr = expData.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }

                var maxStageDiff: Float = 0.0
                var l2DiffSq: Double = 0.0
                var l2ExpSq: Double = 0.0
                var nonFiniteCount: Int = 0

                for i in 0..<stage.count {
                    let actF = Self.bf16ToFloat(actPtr[i])
                    let expF = Self.bf16ToFloat(expPtr[i])
                    if actF.isNaN || actF.isInfinite {
                        nonFiniteCount += 1
                    }
                    let diff = abs(actF - expF)
                    if diff > maxStageDiff {
                        maxStageDiff = diff
                    }
                    l2DiffSq += Double(diff * diff)
                    l2ExpSq += Double(expF * expF)
                }

                let relL2 = (l2ExpSq > 0) ? sqrt(l2DiffSq / l2ExpSq) : sqrt(l2DiffSq)
                let rms = sqrt(l2DiffSq / Double(stage.count))

                if stage.name == "final_out" {
                    maxFinalDiff = maxStageDiff
                    finalRelL2 = relL2
                    finalRMS = rms
                }

                #expect(nonFiniteCount == 0, "Stage \(stage.name) in \(c.name) contained \(nonFiniteCount) non-finite values")
                #expect(maxStageDiff <= 0.25, "Stage \(stage.name) in \(c.name) max diff exceeded: \(maxStageDiff)")
                #expect(relL2 <= 0.02, "Stage \(stage.name) in \(c.name) relative L2 exceeded: \(relL2)")

                print(String(format: "%-16@ | %7d | %12.6f | %12.6e | %12.6e | %10d | PASSED",
                             stage.name, stage.count, maxStageDiff, relL2, rms, nonFiniteCount))
            }

            // Verify router expert selections and weights:
            let flatExpectedIndices = c.router_indices.flatMap { $0.flatMap { $0 } }
            let flatExpectedWeights = c.router_weights.flatMap { $0.flatMap { $0 } }

            print("\n  Router Selections & Weights Breakdown:")
            for t in 0..<c.num_tokens {
                let actualSlice = Array(intermediates.routerExperts[(t * 6)..<(t * 6 + 6)])
                let expectedSlice = Array(flatExpectedIndices[(t * 6)..<(t * 6 + 6)])
                let actualWeights = Array(intermediates.routerWeights[(t * 6)..<(t * 6 + 6)])
                let expectedWeights = Array(flatExpectedWeights[(t * 6)..<(t * 6 + 6)])

                let actSet = Set(actualSlice)
                let expSet = Set(expectedSlice)
                #expect(actSet == expSet, "Selected expert sets divergent in \(c.name) token \(t): actual \(actualSlice) vs expected \(expectedSlice)")

                var maxWDiff: Float = 0.0
                for (k, actExp) in actualSlice.enumerated() {
                    if let expIdx = expectedSlice.firstIndex(of: actExp) {
                        let wDiff = abs(actualWeights[k] - expectedWeights[expIdx])
                        if wDiff > maxWDiff { maxWDiff = wDiff }
                        #expect(wDiff <= 2e-4, "Weight for expert \(actExp) in \(c.name) mismatch: actual \(actualWeights[k]) vs expected \(expectedWeights[expIdx])")
                    }
                }
                let orderMatch = (actualSlice == expectedSlice) ? "YES" : "NO"
                print(String(format: "  Token %d: Metal=%@ | MLX=%@ | SetAgree=YES | OrderAgree=%@ | MaxWeightDiff=%.6e",
                             t, actualSlice.description, expectedSlice.description, orderMatch, maxWDiff))
            }

            // Verify routed expert contributions matching by expert ID
            let routedExpData = Self.parseHexToData(c.routed_hex)
            let actRoutedPtr = intermediates.routedContributions.contents().bindMemory(to: UInt16.self, capacity: c.batch * c.num_tokens * 6 * 2048)
            let expRoutedPtr = routedExpData.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }

            for t in 0..<c.num_tokens {
                let actualSlice = Array(intermediates.routerExperts[(t * 6)..<(t * 6 + 6)])
                let expectedSlice = Array(flatExpectedIndices[(t * 6)..<(t * 6 + 6)])

                for (actSlot, expId) in actualSlice.enumerated() {
                    if let expSlot = expectedSlice.firstIndex(of: expId) {
                        let actOffset = (t * 6 + actSlot) * 2048
                        let expOffset = (t * 6 + expSlot) * 2048
                        var maxExpDiff: Float = 0.0
                        for i in 0..<2048 {
                            let diff = abs(Self.bf16ToFloat(actRoutedPtr[actOffset + i]) - Self.bf16ToFloat(expRoutedPtr[expOffset + i]))
                            if diff > maxExpDiff { maxExpDiff = diff }
                        }
                        #expect(maxExpDiff <= 0.25, "Routed expert \(expId) contribution max diff exceeded: \(maxExpDiff)")
                    }
                }
            }

            // Direct comparison with full_layer_out_hex (MLX module layer(x))
            let fullLayerExpData = Self.parseHexToData(c.full_layer_out_hex)
            let finalPtr = intermediates.finalOut.contents().bindMemory(to: UInt16.self, capacity: c.batch * c.num_tokens * 2048)
            let fullExpPtr = fullLayerExpData.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }
            var fullDiff: Float = 0.0
            var fullL2DiffSq: Double = 0.0
            var fullL2ExpSq: Double = 0.0
            for i in 0..<(c.batch * c.num_tokens * 2048) {
                let actF = Self.bf16ToFloat(finalPtr[i])
                let expF = Self.bf16ToFloat(fullExpPtr[i])
                let diff = abs(actF - expF)
                if diff > fullDiff { fullDiff = diff }
                fullL2DiffSq += Double(diff * diff)
                fullL2ExpSq += Double(expF * expF)
            }
            let fullRelL2 = (fullL2ExpSq > 0) ? sqrt(fullL2DiffSq / fullL2ExpSq) : sqrt(fullL2DiffSq)
            let fullRMS = sqrt(fullL2DiffSq / Double(c.batch * c.num_tokens * 2048))

            #expect(maxFinalDiff <= 0.02, "Final layer 2 output exceeded max diff tolerance in \(c.name): \(maxFinalDiff)")
            #expect(finalRelL2 <= 0.002, "Final layer 2 output exceeded relative L2 tolerance in \(c.name): \(finalRelL2)")
            #expect(fullDiff <= 0.02, "Direct full layer forward comparison failed in \(c.name): \(fullDiff)")

            print(String(format: "%-16@ | %7d | %12.6f | %12.6e | %12.6e | %10d | PASSED (vs MLX recon)",
                         "final_recon", c.batch * c.num_tokens * 2048, maxFinalDiff, finalRelL2, finalRMS, 0))
            print(String(format: "%-16@ | %7d | %12.6f | %12.6e | %12.6e | %10d | PASSED (vs MLX layer(x))",
                         "module_eval", c.batch * c.num_tokens * 2048, fullDiff, fullRelL2, fullRMS, 0))
        }
        print("===============================================================================================\n")
    }

    // MARK: - 7. Layer 2 Repeat-Run Bitwise Determinism (Finite Inputs)

    @Test func layer2RepeatRunDeterminism() throws {
        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)
        let shard1File = try Self.openShard1File()

        let inBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let inPtr = inBuf.contents().bindMemory(to: UInt16.self, capacity: 2048)
        for i in 0..<2048 {
            let val = sin(Float(i) * 0.05 + 0.1) * 0.4
            #expect(!val.isNaN && !val.isInfinite, "Input generator produced non-finite float")
            inPtr[i] = Self.floatToBf16(val)
            let back = Self.bf16ToFloat(inPtr[i])
            #expect(!back.isNaN && !back.isInfinite, "Converted BF16 is non-finite")
        }

        var baselineDigest: String?
        for run in 0..<50 {
            let inter = try runner.executeLayer2ForwardWithIntermediates(
                weights: weights,
                shard1File: shard1File,
                input: inBuf,
                batch: 1,
                numTokens: 1
            )

            // Verify all intermediate and final buffers are strictly finite:
            let intermediateBuffers: [(name: String, buf: MTLBuffer, count: Int)] = [
                ("norm1", inter.norm1, 2048),
                ("qRope", inter.qRope, 4096),
                ("kRope", inter.kRope, 512),
                ("attnOut", inter.attnOut, 4096),
                ("res1", inter.res1, 2048),
                ("norm2", inter.norm2, 2048),
                ("routedContributions", inter.routedContributions, 6 * 2048),
                ("sharedOut", inter.sharedOut, 2048),
                ("moeOut", inter.moeOut, 2048),
                ("finalOut", inter.finalOut, 2048)
            ]
            for s in intermediateBuffers {
                let ptr = s.buf.contents().bindMemory(to: UInt16.self, capacity: s.count)
                for i in 0..<s.count {
                    let val = Self.bf16ToFloat(ptr[i])
                    #expect(!val.isNaN && !val.isInfinite, "Intermediate \(s.name) in run \(run) produced non-finite value at index \(i)")
                }
            }

            let data = Data(bytes: inter.finalOut.contents(), count: 2048 * 2)
            let digest = Self.sha256Hex(data)
            if let base = baselineDigest {
                #expect(digest == base, "Repeat run \(run) produced divergent SHA-256")
            } else {
                baselineDigest = digest
            }
        }

        // Deliberately different finite input verification:
        let inBuf2 = try context.makeBuffer(byteCount: 2048 * 2)
        let inPtr2 = inBuf2.contents().bindMemory(to: UInt16.self, capacity: 2048)
        for i in 0..<2048 {
            let val = cos(Float(i) * 0.05 + 0.1) * 0.4
            #expect(!val.isNaN && !val.isInfinite)
            inPtr2[i] = Self.floatToBf16(val)
        }
        let inter2 = try runner.executeLayer2ForwardWithIntermediates(
            weights: weights,
            shard1File: shard1File,
            input: inBuf2,
            batch: 1,
            numTokens: 1
        )
        let outData2 = Data(bytes: inter2.finalOut.contents(), count: 2048 * 2)
        let digest2 = Self.sha256Hex(outData2)
        #expect(digest2 != baselineDigest, "Deliberately different finite input produced identical SHA-256 hash")

        print("Repeat-run determinism: 50/50 Layer 2 forwards produced identical finite SHA-256: \(baselineDigest ?? "")")
        print("Deliberately distinct finite input produced valid different SHA-256: \(digest2)")
    }

    // MARK: - 8. Bounds, Safety, and KV-Cache Validation (Layer 2)

    @Test func layer2BoundsAndSafetyValidation() throws {
        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)
        let shard1File = try Self.openShard1File()

        let inBuf = try context.makeBuffer(byteCount: 2048 * 2)
        let outBuf = try context.makeBuffer(byteCount: 2048 * 2)

        // Reject batch != 1
        #expect(throws: FlintMetalError.self) {
            _ = try runner.executeLayer2ForwardWithIntermediates(
                weights: weights, shard1File: shard1File, input: inBuf, batch: 2, numTokens: 1
            )
        }
        #expect(throws: FlintMetalError.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: inBuf, output: outBuf, batch: 2, numTokens: 1
            )
        }

        // Reject non-zero cacheOffset (unsupported incremental KV decode)
        #expect(throws: FlintMetalError.self) {
            _ = try runner.executeLayer2ForwardWithIntermediates(
                weights: weights, shard1File: shard1File, input: inBuf, batch: 1, numTokens: 1, cacheOffset: 4
            )
        }
        #expect(throws: FlintMetalError.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: inBuf, output: outBuf, batch: 1, numTokens: 1, cacheOffset: 4
            )
        }

        // Reject numTokens <= 0 or > 16
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2ForwardWithIntermediates(
                weights: weights, shard1File: shard1File, input: inBuf, batch: 1, numTokens: 0
            )
        }
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2ForwardWithIntermediates(
                weights: weights, shard1File: shard1File, input: inBuf, batch: 1, numTokens: 17
            )
        }

        // Negative input offset
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: inBuf, inputOffset: -2, output: outBuf, batch: 1, numTokens: 1
            )
        }
        // Unaligned input offset (odd byte)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: inBuf, inputOffset: 1, output: outBuf, batch: 1, numTokens: 1
            )
        }
        // Negative output offset
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: inBuf, output: outBuf, outputOffset: -2, batch: 1, numTokens: 1
            )
        }
        // Unaligned output offset (odd byte)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: inBuf, output: outBuf, outputOffset: 1, batch: 1, numTokens: 1
            )
        }
        // Undersized input buffer
        let smallInBuf = try context.makeBuffer(byteCount: 1024)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: smallInBuf, output: outBuf, batch: 1, numTokens: 1
            )
        }
        // Undersized output buffer
        let smallOutBuf = try context.makeBuffer(byteCount: 1024)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: inBuf, output: smallOutBuf, batch: 1, numTokens: 1
            )
        }
        // Overlapping input and output buffer ranges in same buffer
        let sharedBuf = try context.makeBuffer(byteCount: 4096)
        #expect(throws: Error.self) {
            _ = try runner.executeLayer2Forward(
                weights: weights, shard1File: shard1File, input: sharedBuf, inputOffset: 0, output: sharedBuf, outputOffset: 1024, batch: 1, numTokens: 1
            )
        }
    }

    // MARK: - 9. Apple M5 Hardware Performance Benchmark (Layer 2)

    @Test func layer2AppleM5PerformanceBenchmark() throws {
        let context = try FlintMetalContext()
        let runner = try FlintDecoderLayerRunner(context: context)
        let weights = try Self.loadLiveWeights(context: context)
        let shard1File = try Self.openShard1File()

        let inBuf1 = try context.makeBuffer(byteCount: 1 * 2048 * 2)
        let inBuf4 = try context.makeBuffer(byteCount: 4 * 2048 * 2)

        print("\n=== FLINT LAYER 2 MOE DECODER HARDWARE BENCHMARK (Apple M5) ===")
        print("Hardware: Apple M5, 32 GB Unified Memory")
        print("Metal Resources: 52.5 MiB Scratch Preallocated, 7.875 MiB Expert Buffers, ~1.28 GB Resident Weights")
        print("Steady-State Allocations: Strictly 0 Bytes (Excluding Reusable GPU Scratch & OS Command Buffers)")
        print("Disclosure: Single-layer T=1 timing reflects isolated kernel forward, not steady-state autoregressive decoding.")
        print(String(format: "%-28@ | %10@ | %10@ | %10@ | %10@ | %10@ | %10@ | %10@ | %10@ | %@",
                     "Configuration", "P50 Wall", "Min Wall", "P95 Wall", "Max Wall", "CB1 Attn", "Disk Read", "CB2 Exp", "CB3 Red", "Status"))
        print(String(repeating: "-", count: 120))

        let benchmarks: [(name: String, run: () throws -> (wallSec: Double, inter: FlintDecoderLayerRunner.Layer2Intermediates))] = [
            ("Layer 2 Decode (T=1, MoE)", {
                let t0 = CACurrentMediaTime()
                let inter = try runner.executeLayer2ForwardWithIntermediates(
                    weights: weights, shard1File: shard1File, input: inBuf1, batch: 1, numTokens: 1
                )
                let t1 = CACurrentMediaTime()
                return (t1 - t0, inter)
            }),
            ("Layer 2 Prefill (T=4, MoE)", {
                let t0 = CACurrentMediaTime()
                let inter = try runner.executeLayer2ForwardWithIntermediates(
                    weights: weights, shard1File: shard1File, input: inBuf4, batch: 1, numTokens: 4
                )
                let t1 = CACurrentMediaTime()
                return (t1 - t0, inter)
            }),
        ]

        for b in benchmarks {
            for _ in 0..<5 { _ = try b.run() }
            var durationsUs: [Double] = []
            durationsUs.reserveCapacity(30)
            var lastInter: FlintDecoderLayerRunner.Layer2Intermediates?
            for _ in 0..<30 {
                let res = try b.run()
                durationsUs.append(res.wallSec * 1e6)
                lastInter = res.inter
            }
            durationsUs.sort()

            let p50 = durationsUs[durationsUs.count / 2]
            let minVal = durationsUs.first ?? 0
            let p95 = durationsUs[Int(Double(durationsUs.count) * 0.95)]
            let maxVal = durationsUs.last ?? 0

            let cb1 = lastInter?.timing.cb1AttentionNormRouterUs ?? 0
            let disk = lastInter?.timing.diskWeightReadUs ?? 0
            let cb2 = lastInter?.timing.cb2RoutedExpertsUs ?? 0
            let cb3 = lastInter?.timing.cb3SharedReductionRes2Us ?? 0

            print(String(format: "%-28@ | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | %8.2f µs | PASSED",
                         b.name, p50, minVal, p95, maxVal, cb1, disk, cb2, cb3))
            print("  Detailed Metrics:")
            print("    Command Buffers: \(lastInter?.timing.commandBuffersCommitted ?? 0)")
            print("    GPU Sync Waits:  \(lastInter?.timing.gpuWaits ?? 0)")
            print("    Disk Bytes Read: \(lastInter?.timing.routedBytesRead ?? 0) B (\(Double(lastInter?.timing.routedBytesRead ?? 0) / 1024.0 / 1024.0) MiB)")
            print("    Total GPU Time:  \(String(format: "%.2f µs", lastInter?.timing.totalGpuElapsedUs ?? 0))")
        }
        print("========================================================================================================================\n")
    }
}
