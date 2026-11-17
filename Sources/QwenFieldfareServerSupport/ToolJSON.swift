import Foundation
import Jinja

public struct JSONMember: Sendable, Equatable {
    public let key: String
    public let value: ToolJSON
    public init(_ key: String, _ value: ToolJSON) { self.key = key; self.value = value }
}
/// Ordered, bounded JSON for the tool wire contract. Rejects duplicate keys,
/// excess nesting and non-finite/unrepresentable numbers before Foundation/Jinja.
public indirect enum ToolJSON: Sendable, Equatable {
    case object([JSONMember]), array([ToolJSON]), string(String), number(String), bool(Bool), null
    public var members: [JSONMember]? { if case .object(let v) = self { return v }; return nil }
    public var array: [ToolJSON]? { if case .array(let v) = self { return v }; return nil }
    public var string: String? { if case .string(let v) = self { return v }; return nil }
    public subscript(_ key: String) -> ToolJSON? { members?.first(where: { $0.key == key })?.value }
    public var encoded: String {
        switch self {
        case .object(let v): return "{" + v.map { ToolJSON.string($0.key).encoded + ":" + $0.value.encoded }.joined(separator: ",") + "}"
        case .array(let v): return "[" + v.map(\.encoded).joined(separator: ",") + "]"
        case .string(let v): return String(decoding: try! JSONSerialization.data(withJSONObject: v, options: [.fragmentsAllowed, .withoutEscapingSlashes]), as: UTF8.self)
        case .number(let v): return v
        case .bool(let v): return v ? "true" : "false"
        case .null: return "null"
        }
    }
    var jinja: Value {
        switch self {
        case .object(let values): return .object(.init(uniqueKeysWithValues: values.map { ($0.key, $0.value.jinja) }))
        case .array(let values): return .array(values.map(\.jinja))
        case .string(let value): return .string(value)
        case .number(let value): if let n = Int(value) { return .int(n) }; return .double(Double(value)!)
        case .bool(let value): return .boolean(value)
        case .null: return .null
        }
    }
    func semanticallyEquals(_ other: ToolJSON) -> Bool {
        switch (self,other) {
        case (.number(let a),.number(let b)):
            if let x=Decimal(string:a),let y=Decimal(string:b) {return x == y}
            return Double(a) == Double(b)
        case (.array(let a),.array(let b)): return a.count == b.count && zip(a,b).allSatisfy { $0.semanticallyEquals($1) }
        case (.object(let a),.object(let b)):
            return a.count == b.count && a.allSatisfy { m in b.first(where: {$0.key == m.key})?.value.semanticallyEquals(m.value) == true }
        default: return self == other
        }
    }
    public static func parse(_ text: String, limit: Int = 1_048_576) throws -> ToolJSON {
        guard text.utf8.count <= limit else { throw APIError(400, "JSON size limit exceeded") }
        var reader = Reader(bytes: Array(text.utf8))
        let value = try reader.value(depth: 0); reader.space()
        guard reader.index == reader.bytes.count else { throw APIError(400, "Trailing JSON data") }
        return value
    }
    // Compare decimal spellings without rounding: Jinja's Double conversion
    // must render the same JSON number, or input is rejected rather than changed.
    private static func canonicalNumber(_ raw: String) -> String? {
        guard raw.utf8.count <= 128 else { return nil }
        var s = raw.lowercased(); let negative = s.hasPrefix("-")
        if negative { s.removeFirst() }
        let parts = s.split(separator:"e",omittingEmptySubsequences:false)
        guard parts.count <= 2 else {return nil}
        var exponent = 0
        if parts.count == 2 {
            var e = String(parts[1]);if e.hasPrefix("+") || e.hasPrefix("-") {e.removeFirst()}
            guard !e.isEmpty,e.utf8.allSatisfy({(48...57).contains($0)}),let n=Int(parts[1]) else {return nil}
            exponent=n
        }
        let decimal=parts[0].split(separator:".",omittingEmptySubsequences:false)
        guard decimal.count <= 2,let whole=decimal.first,!whole.isEmpty,
              whole == "0" || whole.first != "0",
              decimal.allSatisfy({!$0.isEmpty && $0.utf8.allSatisfy {(48...57).contains($0)}}) else {return nil}
        let fraction=decimal.count == 2 ? decimal[1].count : 0
        let adjusted=exponent.subtractingReportingOverflow(fraction)
        guard !adjusted.overflow else {return nil};exponent=adjusted.partialValue
        var digits=decimal.joined();while digits.first == "0" {digits.removeFirst()}
        if digits.isEmpty {return "0"}
        while digits.last == "0" {
            digits.removeLast();let next=exponent.addingReportingOverflow(1)
            guard !next.overflow else {return nil};exponent=next.partialValue
        }
        return (negative ? "-":"") + digits + "e" + String(exponent)
    }
    private struct Reader {
        let bytes: [UInt8]; var index = 0
        mutating func space() { while index < bytes.count && [9,10,13,32].contains(bytes[index]) { index += 1 } }
        mutating func consume(_ byte: UInt8) -> Bool { space(); if index < bytes.count && bytes[index] == byte { index += 1; return true }; return false }
        mutating func quoted() throws -> String {
            space(); let start = index
            guard index < bytes.count && bytes[index] == 34 else { throw APIError(400, "Expected JSON string") }
            index += 1
            while index < bytes.count {
                let b = bytes[index]; index += 1
                if b == 92 { guard index < bytes.count else { break }; index += 1 }
                else if b == 34 {
                    guard let value = try? JSONSerialization.jsonObject(with: Data(bytes[start..<index]), options: [.fragmentsAllowed]) as? String else { throw APIError(400, "Invalid JSON string") }
                    return value
                }
            }
            throw APIError(400, "Unterminated JSON string")
        }
        mutating func value(depth: Int) throws -> ToolJSON {
            guard depth <= 16 else { throw APIError(400, "JSON depth exceeds 16") }; space()
            guard index < bytes.count else { throw APIError(400, "Incomplete JSON") }
            if bytes[index] == 34 { return .string(try quoted()) }
            if consume(123) {
                var result: [JSONMember] = []; var keys = Set<String>()
                if consume(125) { return .object(result) }
                repeat {
                    let key = try quoted()
                    guard keys.insert(key).inserted, consume(58) else { throw APIError(400, "Duplicate key or missing colon") }
                    result.append(.init(key, try value(depth: depth+1)))
                    if consume(125) { return .object(result) }
                    guard consume(44) else { throw APIError(400, "Invalid object separator") }
                } while true
            }
            if consume(91) {
                var result: [ToolJSON] = []
                if consume(93) { return .array(result) }
                repeat {
                    result.append(try value(depth: depth+1))
                    if consume(93) { return .array(result) }
                    guard consume(44) else { throw APIError(400, "Invalid array separator") }
                } while true
            }
            let start = index
            while index < bytes.count && ![9,10,13,32,44,93,125].contains(bytes[index]) { index += 1 }
            let raw = String(decoding: bytes[start..<index], as: UTF8.self)
            switch raw { case "true": return .bool(true); case "false": return .bool(false); case "null": return .null; default: break }
            guard let n = try? JSONSerialization.jsonObject(with: Data(raw.utf8), options: [.fragmentsAllowed]) as? NSNumber,
                  n.doubleValue.isFinite, raw.first == "-" || raw.first?.isNumber == true else { throw APIError(400, "Invalid JSON value") }
            guard let canonical = ToolJSON.canonicalNumber(raw) else {throw APIError(400,"Invalid or oversized JSON number")}
            if !raw.contains(".") && !raw.lowercased().contains("e") {
                guard Int(raw) != nil else {throw APIError(400,"Integer exceeds supported Int64 range")}
            } else {
                guard canonical == ToolJSON.canonicalNumber(String(n.doubleValue)) else {throw APIError(400,"Number cannot be rendered without precision loss")}
            }
            return .number(raw)
        }
    }
}

func safeToolName(_ name: String) -> Bool {
    !name.isEmpty && name.utf8.count <= 64 && name.utf8.allSatisfy { (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || [45,95].contains($0) }
}
let nativeMarkers = ["<tool_call", "</tool_call", "<function=", "</function", "<parameter=", "</parameter", "<tool_response", "</tool_response", "<|im_", "<think>", "</think>"]
func rejectNativeMarkers(_ text: String) throws {
    guard !nativeMarkers.contains(where: text.contains) else { throw APIError(400, "Reserved native delimiters are not supported in tool data") }
}
