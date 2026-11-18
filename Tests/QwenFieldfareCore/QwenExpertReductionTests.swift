import Metal
import Testing
@testable import QwenFieldfareCore

struct QwenExpertReductionTests {
    @Test func cachePartitionsWriteTheSameRouterOrderedContributions() throws {
        let context = try MetalContext()
        let width = 2048
        let slots: [UInt32] = [7, 0, 6, 1, 5, 2, 4, 3]
        let down = try context.pipeline("qwen_cached_moe_down_by_position_q4_simd_f32")
        let combine = try context.pipeline("qwen_combine_cached_moe_by_position_f32")
        let stride = Q4BoundedExpertCache.expertByteLength
        let arena = try buffer(context, Array(repeating: UInt8.zero, count: 8 * stride))
        for slot in 0..<8 {
            let record = arena.contents().advanced(by: slot * stride)
            let downWeights = record.advanced(by: 1_179_648)
            downWeights.initializeMemory(as: UInt8.self, repeating: UInt8(17 * (slot + 1)), count: 524_288)
            let downScales = record.advanced(by: 1_703_936)
            downScales.initializeMemory(as: UInt16.self, repeating: 0x3f80, count: 16_384)
        }

        let acts = try buffer(context, Array(repeating: Float(1), count: 8 * 512))
        let scores = try buffer(context, (1...8).map { Float($0) / 8 })
        let slotIndices = try buffer(context, slots)
        let shared = try buffer(context, Array(repeating: Float(3), count: width))
        let residual = try buffer(context, Array(repeating: Float(-4), count: width))
        let scratch = try buffer(context, Array(repeating: Float.nan, count: 8 * width))
        let output = try buffer(context, Array(repeating: Float.nan, count: width))
        #expect(scratch.length == QwenActiveStreamingMemoryConfiguration.routedContributionBytes)

        let partitions: [[UInt32]] = [[], [0, 2, 4, 6], [1, 3, 5, 7], Array(0..<8)]
        for hits in partitions {
            scratch.contents().assumingMemoryBound(to: Float.self).update(repeating: .nan, count: 8 * width)
            let misses = (0..<8).map(UInt32.init).filter { !hits.contains($0) }
            for positions in [hits, misses] where !positions.isEmpty {
                let command = try #require(context.queue.makeCommandBuffer())
                let encoder = try #require(command.makeComputeCommandEncoder())
                let positionBuffer = try buffer(context, positions)
                let inputs = [arena, acts, scores, scratch, slotIndices, positionBuffer]
                encoder.setComputePipelineState(down)
                for (index, input) in inputs.enumerated() {
                    encoder.setBuffer(input, offset: 0, index: index)
                }
                var count = UInt32(positions.count)
                encoder.setBytes(&count, length: 4, index: 6)
                encoder.dispatchThreadgroups(MTLSize(width: width, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                command.commit()
            }
            let command = try encodeCombine(context, combine, [scratch, shared, residual, output])
            try #require(command.status == .completed)
            let routed = scratch.contents().assumingMemoryBound(to: Float.self)
            let combined = output.contents().assumingMemoryBound(to: Float.self)
            let expected = slots.enumerated().map { position, slot in
                Float(slot + 1) * 512 * Float(position + 1) / 8
            }
            #expect((0..<8).allSatisfy { position in
                (0..<width).allSatisfy { routed[position * width + $0] == expected[position] }
            })
            #expect((0..<width).allSatisfy { combined[$0] == expected.reduce(-1, +) })
        }
    }

    @Test func chosenCombinePreservesRouterOrderAndResidualGrouping() throws {
        let context = try MetalContext()
        let width = 2048
        let large: Float = 1_073_741_824
        var weighted = Array(repeating: Float.zero, count: 8 * width)
        var shared = Array(repeating: Float.zero, count: width)
        var residual = shared
        weighted[0] = large
        shared[0] = -large
        residual[0] = 1
        weighted[1] = large
        weighted[width + 1] = 1
        weighted[2 * width + 1] = -large
        weighted[2] = large
        weighted[width + 2] = -large
        weighted[2 * width + 2] = 1
        let output = try buffer(context, Array(repeating: Float.nan, count: width))
        let inputs = try [buffer(context, weighted), buffer(context, shared), buffer(context, residual), output]
        let pipeline = try context.pipeline("qwen_combine_cached_moe_by_position_f32")
        let command = try encodeCombine(context, pipeline, inputs)
        try #require(command.status == .completed)
        let values = UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: 3)
        #expect(Array(values) == [0, 0, 1])
    }

    private func encodeCombine(
        _ context: MetalContext, _ pipeline: MTLComputePipelineState, _ buffers: [MTLBuffer]
    ) throws -> MTLCommandBuffer {
        let command = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        encoder.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        return command
    }

    private func buffer<T>(_ context: MetalContext, _ data: [T]) throws -> MTLBuffer {
        try data.withUnsafeBytes { bytes in
            let start = try #require(bytes.baseAddress)
            return try #require(context.device.makeBuffer(bytes: start, length: bytes.count, options: .storageModeShared))
        }
    }
}
