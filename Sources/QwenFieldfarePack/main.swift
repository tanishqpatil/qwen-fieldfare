import Foundation
import QwenFieldfareCore

private let usage = """
Usage:
  QwenFieldfarePack --model <Qwen3.6-35B-A3B-4bit> [--output <directory>] [--resume]
  QwenFieldfarePack --verify --model <Qwen3.6-35B-A3B-4bit> [--output <directory>]
"""

private struct Arguments {
    var modelPath: String?
    var outputDirectory: String?
    var resume = false
    var verify = false

    init(_ values: [String]) throws {
        var index = 1
        while index < values.count {
            switch values[index] {
            case "--help":
                throw ParseError.help
            case "--resume":
                resume = true
                index += 1
            case "--verify":
                verify = true
                index += 1
            case "--model", "--output":
                let option = values[index]
                guard index + 1 < values.count else { throw ParseError.missingValue(option) }
                if option == "--model" { modelPath = values[index + 1] }
                else { outputDirectory = values[index + 1] }
                index += 2
            default:
                throw ParseError.unknown(values[index])
            }
        }
        guard modelPath != nil else { throw ParseError.missingValue("--model") }
        guard !(verify && resume) else { throw ParseError.invalid("--verify and --resume are mutually exclusive") }
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case missingValue(String)
    case unknown(String)
    case invalid(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .missingValue(let option): return "missing value for \(option)"
        case .unknown(let option): return "unknown option \(option)"
        case .invalid(let detail): return detail
        }
    }
}

private func gib(_ bytes: UInt64) -> Double {
    Double(bytes) / 1_073_741_824.0
}

@main
private enum QwenPackMain {
    static func main() {
        do {
            let arguments = try Arguments(CommandLine.arguments)
            let modelPath = arguments.modelPath!
            if arguments.verify {
                let result = try QwenQ4ExpertStoreBuilder.verify(
                    modelPath: modelPath,
                    outputDirectory: arguments.outputDirectory,
                    progress: { print($0) })
                print("Verified \(result.layersVerified) packed expert layers")
                print(String(format: "Verified payload: %.3f GiB", gib(result.totalBytesVerified)))
                print("Store: \(result.outputDirectory)")
            } else {
                let result = try QwenQ4ExpertStoreBuilder.pack(
                    modelPath: modelPath,
                    outputDirectory: arguments.outputDirectory,
                    resume: arguments.resume,
                    progress: { print($0) })
                print("Packed expert layers written: \(result.layersWritten)")
                print(String(format: "Store size: %.3f GiB", gib(result.totalBytes)))
                print("Source index SHA-256: \(result.sourceIndexSHA256)")
                print("Store: \(result.outputDirectory)")
            }
        } catch ParseError.help {
            print(usage)
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n\n\(usage)\n".utf8))
            Darwin.exit(1)
        }
    }
}
