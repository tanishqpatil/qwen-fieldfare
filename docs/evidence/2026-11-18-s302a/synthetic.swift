// Append this to the exact selector declarations with run_synthetic.py.
struct SyntheticCase {
    let name: String
    let seen: Set<Int32>
    let configuration: QwenGenerationConfiguration
}

struct SyntheticResult: Encodable {
    let round: Int
    let name: String
    let totalNs: UInt64
    let totalThreadCPUNs: UInt64
    let choices: Int
    let minimumSearches: Int
}

let count = 248_320
let winner: Int32 = 1_000
let seen = Set((0..<300).map { Int32(37 + $0 * 811) })
precondition(!seen.contains(winner))
var logits = (0..<count).map { index in
    Float((UInt32(index) &* 2_654_435_761) % 65_521) / 8_192
}
for id in seen { logits[Int(id)] = -12 }
logits[Int(winner)] = 50
let standard = QwenGenerationConfiguration(maxNewTokens: 512, seed: 42)
var kOne = standard
kOne.topK = 1
let cases = [
    SyntheticCase(name: "default-300-seen", seen: seen, configuration: standard),
    SyntheticCase(name: "default-empty-seen", seen: [], configuration: standard),
    SyntheticCase(name: "k1-empty-seen", seen: [], configuration: kOne),
]

var observations: [SyntheticResult] = []
logits.withUnsafeBufferPointer { buffer in
    for round in 0..<6 {
        for offset in cases.indices {
            let test = cases[(offset + round) % cases.count]
            let probe = QwenSelectorProbe()
            let expected = QwenLogitSelector.select(
                logits: buffer.baseAddress!, count: buffer.count, seenTokenIDs: test.seen,
                configuration: test.configuration, samplePosition: 0, selectorProbe: probe)
            precondition(expected == winner)
            let choices = 128
            let started = QwenSelectorProbe.now()
            let cpuStarted = QwenSelectorProbe.threadCPU()
            for position in 1...choices {
                let value = QwenLogitSelector.select(
                    logits: buffer.baseAddress!, count: buffer.count, seenTokenIDs: test.seen,
                    configuration: test.configuration, samplePosition: position)
                precondition(value == winner)
            }
            let cpuNs = QwenSelectorProbe.threadCPU() - cpuStarted
            let wallNs = QwenSelectorProbe.now() - started
            observations.append(SyntheticResult(
                round: round, name: test.name, totalNs: wallNs, totalThreadCPUNs: cpuNs,
                choices: choices, minimumSearches: probe.step.minimumSearches))
        }
    }
}
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
print(String(decoding: try encoder.encode(observations), as: UTF8.self))
