import Testing
@testable import QwenFieldfarePresentation

@Suite struct FastMarkdownParserTests {
    @Test func partialStreamingHeadingAlwaysAdvances() {
        let blocks = FastMarkdownParser.parse("Complete paragraph.\n\n#", isStreaming: true)

        #expect(blocks.count == 2)
        #expect(blocks[0].kind == .prose("Complete paragraph."))
        #expect(blocks[1].kind == .prose("#"))
    }

    @Test(arguments: ["#NoSpace", "##NoSpace", "####", "##### heading"])
    func malformedHeadingIsRenderedAsProse(_ source: String) {
        let blocks = FastMarkdownParser.parse(source, isStreaming: true)

        #expect(blocks.count == 1)
        #expect(blocks[0].kind == .prose(source))
    }

    @Test func validHeadingStillParses() {
        let blocks = FastMarkdownParser.parse("## Heading\n\nBody", isStreaming: false)

        #expect(blocks.count == 2)
        #expect(blocks[0].kind == .heading(level: 2, text: "Heading"))
        #expect(blocks[1].kind == .prose("Body"))
    }

    @Test func incompleteCodeFenceConsumesTheStreamingTail() {
        let blocks = FastMarkdownParser.parse("```swift\nlet value = 42", isStreaming: true)

        #expect(blocks.count == 1)
        #expect(blocks[0].kind == .code(language: "swift", code: "let value = 42"))
    }
}
