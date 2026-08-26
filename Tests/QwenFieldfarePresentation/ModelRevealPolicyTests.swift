import Foundation
import Testing
@testable import QwenFieldfarePresentation

@Suite struct ModelRevealPolicyTests {
    @Test func installedModelDirectoryIsSelected() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/models/Qwen3.6-35B-A3B-4bit",
            fileExists: { $0 == "/Users/dev/models/Qwen3.6-35B-A3B-4bit" })
        #expect(target == .selectItem(
            URL(fileURLWithPath: "/Users/dev/models/Qwen3.6-35B-A3B-4bit",
                isDirectory: true)))
    }

    @Test func missingModelFallsBackToItsContainer() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/models/Qwen3.6-35B-A3B-4bit",
            fileExists: { $0 == "/Users/dev/models" })
        #expect(target == .openContainer(
            URL(fileURLWithPath: "/Users/dev/models", isDirectory: true)))
    }

    @Test func missingContainerIsUnavailable() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/models/Qwen3.6-35B-A3B-4bit",
            fileExists: { _ in false })
        #expect(target == .unavailable)
    }

    @Test(arguments: ["", "   "])
    func blankPathIsUnavailable(path: String) {
        #expect(ModelRevealPolicy.target(forModelPath: path,
                                         fileExists: { _ in true }) == .unavailable)
    }

    @Test func pathIsStandardizedBeforeReveal() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/models/./archive/../Qwen3.6-35B-A3B-4bit",
            fileExists: { $0 == "/Users/dev/models/Qwen3.6-35B-A3B-4bit" })
        #expect(target == .selectItem(
            URL(fileURLWithPath: "/Users/dev/models/Qwen3.6-35B-A3B-4bit",
                isDirectory: true)))
    }

    @Test func rootPathDoesNotLoopIntoItself() {
        #expect(ModelRevealPolicy.target(forModelPath: "/",
                                         fileExists: { _ in false }) == .unavailable)
    }
}
