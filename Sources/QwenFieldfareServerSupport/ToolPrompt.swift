import Foundation
import Jinja

/// Executes the checkpoint template, rather than reproducing its tool grammar.
/// Its tojson override matches HF's ordered, Unicode-preserving JSON rendering.
public struct ToolPrompt: Sendable {
    private let template: Template
    public init(source: String) throws { template = try Template(source) }
    public func render(_ conversation: ToolConversation) throws -> String {
        let messages = conversation.messages.map { message -> ToolJSON in
            var fields: [JSONMember] = [.init("role",.string(message.role)),.init("content",.string(message.content))]
            if !message.calls.isEmpty { fields.append(.init("tool_calls",.array(message.calls.map(\.nativeJSON)))) }
            return .object(fields)
        }
        let env = Environment()
        env["tojson"] = .function { args, _, _ in
            guard let value = args.first else { return .string("null") }
            return .string(try Self.json(value).templateJSON)
        }
        return try template.render([
            "messages": ToolJSON.array(messages).jinja,
            "tools": ToolJSON.array(conversation.allowsCalls ? conversation.tools.map(\.json) : []).jinja,
            "add_generation_prompt": .boolean(true),
            "enable_thinking": .boolean(false),
            "preserve_thinking": .boolean(false)
        ], environment: env)
    }
    private static func json(_ v: Value) throws -> ToolJSON {
        switch v {
        case .null: return .null
        case .string(let x): return .string(x)
        case .boolean(let x): return .bool(x)
        case .int(let x): return .number(String(x))
        case .double(let x): return .number(String(x))
        case .array(let x): return .array(try x.map(json))
        case .object(let x): return .object(try x.map { .init($0.key,try json($0.value)) })
        default: throw APIError(400,"Unsupported template value")
        }
    }
}
extension ToolJSON {
    var templateJSON: String {
        switch self {
        case .object(let v): return "{" + v.map { ToolJSON.string($0.key).encoded + ": " + $0.value.templateJSON }.joined(separator:", ") + "}"
        case .array(let v): return "[" + v.map(\.templateJSON).joined(separator:", ") + "]"
        default: return encoded
        }
    }
}
