import Metal
import Testing
@testable import QwenFieldfareCore

@Suite
struct QwenFP16AttentionBridgeTests {
    @Test
    func kvWriterUsesTokenMajorFP16Layout() throws {
        let context = try MetalContext()
        let device = context.device
        let k = (0..<512).map { Float($0) / 64.0 - 4.0 }
        let v = (0..<512).map { 3.0 - Float($0) / 80.0 }
        let cacheCount = 3 * 2 * 256
        guard let kInput = device.makeBuffer(bytes: k, length: k.count * 4),
              let vInput = device.makeBuffer(bytes: v, length: v.count * 4),
              let kCache = device.makeBuffer(length: cacheCount * 2, options: .storageModeShared),
              let vCache = device.makeBuffer(length: cacheCount * 2, options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("buffer allocation failed")
            return
        }
        memset(kCache.contents(), 0, cacheCount * 2)
        memset(vCache.contents(), 0, cacheCount * 2)

        var position: UInt32 = 1
        var maxSequence: UInt32 = 3
        encoder.setComputePipelineState(try context.pipeline("qwen_update_kv_cache_f16"))
        encoder.setBuffer(kInput, offset: 0, index: 0)
        encoder.setBuffer(vInput, offset: 0, index: 1)
        encoder.setBuffer(kCache, offset: 0, index: 2)
        encoder.setBuffer(vCache, offset: 0, index: 3)
        encoder.setBytes(&position, length: 4, index: 4)
        encoder.setBytes(&maxSequence, length: 4, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actualK = Array(UnsafeBufferPointer(
            start: kCache.contents().assumingMemoryBound(to: Float16.self), count: cacheCount))
        let actualV = Array(UnsafeBufferPointer(
            start: vCache.contents().assumingMemoryBound(to: Float16.self), count: cacheCount))
        let tokenOffset = 2 * 256
        #expect(actualK[..<tokenOffset].allSatisfy { $0 == 0 })
        #expect(actualV[..<tokenOffset].allSatisfy { $0 == 0 })
        for index in 0..<512 {
            #expect(actualK[tokenOffset + index] == Float16(k[index]))
            #expect(actualV[tokenOffset + index] == Float16(v[index]))
        }
    }

    @Test
    func halfAttentionBridgeAppliesGateInFP32() throws {
        let context = try MetalContext()
        let device = context.device
        let attention: [Float16] = [0.5, -1.0, 2.0, -3.0]
        let gate: [Float] = [-2.0, -0.5, 0.5, 2.0]
        guard let attentionBuffer = device.makeBuffer(bytes: attention, length: attention.count * 2),
              let gateBuffer = device.makeBuffer(bytes: gate, length: gate.count * 4),
              let output = device.makeBuffer(length: gate.count * 4, options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            Issue.record("buffer allocation failed")
            return
        }
        var count = UInt32(attention.count)
        encoder.setComputePipelineState(try context.pipeline("qwen_apply_attention_gate_f16_f32"))
        encoder.setBuffer(attentionBuffer, offset: 0, index: 0)
        encoder.setBuffer(gateBuffer, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actual = UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: Float.self), count: attention.count)
        for index in attention.indices {
            let expected = Float(attention[index]) / (1.0 + exp(-gate[index]))
            #expect(abs(actual[index] - expected) < 1e-5)
        }
    }
}
