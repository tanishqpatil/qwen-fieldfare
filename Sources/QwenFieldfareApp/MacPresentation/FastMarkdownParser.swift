import Foundation

public struct ParsedContentBlock: Identifiable, Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case thought(String, isStillThinking: Bool)
        case code(language: String?, code: String)
        case table(headers: [String], rows: [[String]])
        case heading(level: Int, text: String)
        case quote(String)
        case prose(String)
    }

    public let id: String
    public let kind: Kind

    public init(id: String, kind: Kind) {
        self.id = id
        self.kind = kind
    }
}

public enum FastMarkdownParser {
    public static func parse(_ text: String, isStreaming: Bool) -> [ParsedContentBlock] {
        var blocks: [ParsedContentBlock] = []
        var remaining = text

        while let thinkStartRange = remaining.range(of: "<think>") {
            let beforeThink = String(remaining[..<thinkStartRange.lowerBound])
            if !beforeThink.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                parseMarkdownSections(beforeThink, to: &blocks)
            }

            let afterThinkStart = remaining[thinkStartRange.upperBound...]
            if let thinkEndRange = afterThinkStart.range(of: "</think>") {
                let thoughtContent = String(afterThinkStart[..<thinkEndRange.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !thoughtContent.isEmpty {
                    blocks.append(ParsedContentBlock(
                        id: "block_\(blocks.count)_thought",
                        kind: .thought(thoughtContent, isStillThinking: false)))
                }
                remaining = String(afterThinkStart[thinkEndRange.upperBound...])
            } else {
                let thoughtContent = String(afterThinkStart)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !thoughtContent.isEmpty {
                    blocks.append(ParsedContentBlock(
                        id: "block_\(blocks.count)_thought",
                        kind: .thought(thoughtContent, isStillThinking: isStreaming)))
                }
                remaining = ""
                break
            }
        }

        if !remaining.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parseMarkdownSections(remaining, to: &blocks)
        }
        return blocks
    }

    private static func parseMarkdownSections(
        _ text: String,
        to blocks: inout [ParsedContentBlock]
    ) {
        let lines = text.components(separatedBy: "\n")
        var index = 0

        while index < lines.count {
            let iterationStart = index
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("```") {
                let label = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                let language = label.isEmpty ? nil : label
                var codeLines: [String] = []
                index += 1
                while index < lines.count {
                    let codeLine = lines[index]
                    if codeLine.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                        index += 1
                        break
                    }
                    codeLines.append(codeLine)
                    index += 1
                }
                blocks.append(ParsedContentBlock(
                    id: "block_\(blocks.count)_code",
                    kind: .code(language: language, code: codeLines.joined(separator: "\n"))))
                continue
            }

            if isTableStart(lines: lines, index: index) {
                let headers = parseTableRow(trimmed)
                var rows: [[String]] = []
                index += 2
                while index < lines.count {
                    let rowLine = lines[index].trimmingCharacters(in: .whitespaces)
                    guard rowLine.hasPrefix("|") && rowLine.hasSuffix("|") else { break }
                    if !isTableDelimiter(rowLine) {
                        rows.append(parseTableRow(rowLine))
                    }
                    index += 1
                }
                blocks.append(ParsedContentBlock(
                    id: "block_\(blocks.count)_table",
                    kind: .table(headers: headers, rows: rows)))
                continue
            }

            if let heading = parseHeading(trimmed) {
                blocks.append(ParsedContentBlock(
                    id: "block_\(blocks.count)_h\(heading.level)",
                    kind: .heading(level: heading.level, text: heading.text)))
                index += 1
                continue
            }

            if trimmed.hasPrefix(">") {
                var quoteLines: [String] = []
                while index < lines.count {
                    let quoteLine = lines[index].trimmingCharacters(in: .whitespaces)
                    guard quoteLine.hasPrefix(">") else { break }
                    quoteLines.append(String(quoteLine.dropFirst())
                        .trimmingCharacters(in: .whitespaces))
                    index += 1
                }
                blocks.append(ParsedContentBlock(
                    id: "block_\(blocks.count)_quote",
                    kind: .quote(quoteLines.joined(separator: "\n"))))
                continue
            }

            var proseLines: [String] = []
            while index < lines.count {
                let proseLine = lines[index]
                let proseTrimmed = proseLine.trimmingCharacters(in: .whitespaces)
                if isSpecialBlockStart(lines: lines, index: index, trimmed: proseTrimmed) {
                    break
                }
                proseLines.append(proseLine)
                index += 1
                if proseTrimmed.isEmpty { break }
            }

            let prose = proseLines.joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !prose.isEmpty {
                blocks.append(ParsedContentBlock(
                    id: "block_\(blocks.count)_prose",
                    kind: .prose(prose)))
            }

            // Malformed or partially streamed Markdown must never pin the main
            // thread. This is a final progress guarantee in addition to using
            // identical recognition rules above and in isSpecialBlockStart.
            if index == iterationStart {
                blocks.append(ParsedContentBlock(
                    id: "block_\(blocks.count)_prose",
                    kind: .prose(line)))
                index += 1
            }
        }
    }

    private static func isSpecialBlockStart(
        lines: [String],
        index: Int,
        trimmed: String
    ) -> Bool {
        trimmed.hasPrefix("```")
            || isTableStart(lines: lines, index: index)
            || parseHeading(trimmed) != nil
            || trimmed.hasPrefix(">")
    }

    private static func parseHeading(_ line: String) -> (level: Int, text: String)? {
        guard line.hasPrefix("#") else { return nil }
        let level = line.prefix(while: { $0 == "#" }).count
        guard (1...4).contains(level) else { return nil }
        let remainder = line.dropFirst(level)
        guard remainder.hasPrefix(" ") else { return nil }
        return (level, String(remainder).trimmingCharacters(in: .whitespaces))
    }

    private static func isTableStart(lines: [String], index: Int) -> Bool {
        guard index + 1 < lines.count else { return false }
        let line = lines[index].trimmingCharacters(in: .whitespaces)
        guard line.hasPrefix("|") && line.hasSuffix("|") else { return false }
        return isTableDelimiter(lines[index + 1].trimmingCharacters(in: .whitespaces))
    }

    private static func isTableDelimiter(_ line: String) -> Bool {
        let clean = line.replacingOccurrences(of: " ", with: "")
        guard clean.hasPrefix("|") && clean.hasSuffix("|") else { return false }
        let parts = clean.split(separator: "|", omittingEmptySubsequences: true)
        return !parts.isEmpty && parts.allSatisfy { part in
            part.allSatisfy { $0 == "-" || $0 == ":" }
        }
    }

    private static func parseTableRow(_ line: String) -> [String] {
        var cells = line.split(separator: "|", omittingEmptySubsequences: false).map {
            String($0).trimmingCharacters(in: .whitespaces)
        }
        if cells.first?.isEmpty == true { cells.removeFirst() }
        if cells.last?.isEmpty == true { cells.removeLast() }
        return cells
    }
}
