import Foundation

public enum NativeToolError: String, Error, Sendable {
    case bufferLimit, truncated, malformed, unknownFunction, unknownParameter
    case invalidArguments, duplicateParameter, tooManyCalls, suffix, turnNotFinished, closed
}
public struct ParsedTurn: Sendable {
    public let content: String
    public let calls: [FunctionCall]
}
/// Incremental safe-text state plus a bounded structured tail. No call is
/// published until finish validates the WHOLE turn and the actual EOS boundary.
public final class NativeToolParser {
    private enum State { case prose, calls, finished, failed }
    private var state: State = .prose
    var isBufferingCalls: Bool {state == .calls}
    private var pending = ""
    private var content = ""
    private var byteCount = 0
    private let tools: [FunctionTool]
    private let emit: (String) -> Void
    public init(tools: [FunctionTool], emit: @escaping (String) -> Void = { _ in }) { self.tools = tools; self.emit = emit }
    public func append(_ fragment: String) throws {
        guard state != .finished && state != .failed else { throw NativeToolError.closed }
        byteCount += fragment.utf8.count
        guard byteCount <= 262_144 else { state = .failed; throw NativeToolError.bufferLimit }
        pending += fragment
        if state == .calls { return }
        while !pending.isEmpty {
            if pending.hasPrefix("<tool_call>") { state = .calls; return }
            if pending.first == "<" {
                if nativeMarkers.contains(where: { $0.hasPrefix(pending) }) || "<tool_call>".hasPrefix(pending) { return }
                if nativeMarkers.contains(where: { pending.hasPrefix($0) }) { state = .failed; throw NativeToolError.malformed }
                release(String(pending.removeFirst()))
            } else if let index = pending.firstIndex(of: "<") {
                release(String(pending[..<index])); pending = String(pending[index...])
            } else { release(pending); pending = "" }
        }
    }
    private func release(_ value: String) { content += value; emit(value) }
    public func finish(endedNormally: Bool) throws -> ParsedTurn {
        guard state != .finished && state != .failed else { throw NativeToolError.closed }
        do {
            if state == .prose {
                guard pending.isEmpty else { throw NativeToolError.truncated }
                state = .finished; return ParsedTurn(content:content,calls:[])
            }
            guard endedNormally else { throw NativeToolError.turnNotFinished }
            var reader = Calls(bytes:Array(pending.utf8),tools:tools)
            let calls = try reader.parse()
            state = .finished; pending = ""
            return ParsedTurn(content:content,calls:calls)
        } catch { state = .failed; throw error }
    }
    private struct Calls {
        let bytes: [UInt8]; let tools: [FunctionTool]; var index = 0
        mutating func whitespace() { while index < bytes.count && [9,10,13,32].contains(bytes[index]) { index += 1 } }
        func starts(_ token: String) -> Bool { bytes[index...].starts(with: token.utf8) }
        mutating func consume(_ token: String) throws {
            guard starts(token) else { throw index >= bytes.count ? NativeToolError.truncated : NativeToolError.malformed }
            index += token.utf8.count
        }
        mutating func name(after prefix: String) throws -> String {
            try consume(prefix); let start = index
            while index < bytes.count && bytes[index] != 62 { index += 1 }
            guard index < bytes.count else { throw NativeToolError.truncated }
            let name = String(decoding:bytes[start..<index],as:UTF8.self); index += 1
            guard safeToolName(name) else { throw NativeToolError.malformed }; return name
        }
        mutating func parse() throws -> [FunctionCall] {
            var result: [FunctionCall] = []
            while true {
                whitespace(); if index == bytes.count { break }
                guard starts("<tool_call>") else { throw NativeToolError.suffix }
                guard result.count < 8 else { throw NativeToolError.tooManyCalls }
                try consume("<tool_call>"); whitespace()
                let name = try name(after:"<function=")
                guard let tool = tools.first(where: { $0.name == name }) else { throw NativeToolError.unknownFunction }
                var arguments: [JSONMember] = []; var seen = Set<String>()
                while true {
                    whitespace()
                    if starts("</function>") { try consume("</function>"); break }
                    let parameter = try self.name(after:"<parameter=")
                    guard seen.insert(parameter).inserted else { throw NativeToolError.duplicateParameter }
                    guard let property = tool.schema.properties.first(where: { $0.key == parameter }) else { throw NativeToolError.unknownParameter }
                    let schema = try ToolSchema(property.value)
                    let start = index
                    var quoted = false; var escaped = false
                    let jsonValue = !schema.types.contains("string")
                    while index < bytes.count {
                        let b = bytes[index]
                        if jsonValue {
                            if escaped { escaped = false; index += 1; continue }
                            if b == 92 && quoted { escaped = true; index += 1; continue }
                            if b == 34 { quoted.toggle(); index += 1; continue }
                        }
                        if !quoted && starts("</parameter>") { break }
                        index += 1
                    }
                    guard index < bytes.count else { throw NativeToolError.truncated }
                    var raw = String(decoding:bytes[start..<index],as:UTF8.self)
                    // The native serializer surrounds values with exactly one LF.
                    if raw.hasPrefix("\n") { raw.removeFirst() }
                    if raw.hasSuffix("\n") { raw.removeLast() }
                    try consume("</parameter>")
                    guard !nativeMarkers.contains(where:raw.contains) else { throw NativeToolError.malformed }
                    var candidates: [ToolJSON] = []
                    if schema.types.contains("string"), schema.accepts(.string(raw)) { candidates.append(.string(raw)) }
                    if !schema.types.contains("string") || schema.types.count > 1,
                       let value = try? ToolJSON.parse(raw,limit:65_536), schema.accepts(value) { candidates.append(value) }
                    guard candidates.count == 1 else { throw NativeToolError.invalidArguments }
                    arguments.append(.init(parameter,candidates[0]))
                }
                whitespace(); try consume("</tool_call>")
                let value = ToolJSON.object(arguments)
                guard tool.schema.accepts(value) else { throw NativeToolError.invalidArguments }
                result.append(.init(name:name,arguments:value))
            }
            guard !result.isEmpty else { throw NativeToolError.malformed }
            return result
        }
    }
}

/// Split only at Unicode scalar boundaries; concatenation is byte-identical.
public func argumentFragments(_ text: String) -> [String] {
    var result: [String] = []; var pending = ""; var count = 0
    for scalar in text.unicodeScalars {
        pending.unicodeScalars.append(scalar); count += scalar.utf8.count
        if count >= 128 {result.append(pending);pending="";count=0}
    }
    if !pending.isEmpty {result.append(pending)}
    return result
}
