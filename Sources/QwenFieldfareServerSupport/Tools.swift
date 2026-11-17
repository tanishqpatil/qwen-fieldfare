import Foundation

public struct ToolSchema: Sendable {
    public let json: ToolJSON
    public let types: [String]
    public let properties: [JSONMember]
    public let required: Set<String>
    public init(_ json: ToolJSON, depth: Int = 0) throws {
        guard depth <= 12, let entries = json.members,
              Set(entries.map(\.key)).isSubset(of: ["type","properties","required","additionalProperties","items","enum","description","title"]) else { throw APIError(400, "Unsupported JSON Schema structure or keyword") }
        let raw = json["type"]
        if let type = raw?.string { types = [type] }
        else if let list = raw?.array, !list.isEmpty, list.allSatisfy({ $0.string != nil }) { types = list.map { $0.string! } }
        else { throw APIError(400, "Schema requires explicit type") }
        guard Set(types).count == types.count, Set(types).isSubset(of: ["object","array","string","number","integer","boolean","null"]) else { throw APIError(400, "Unsupported schema type") }
        for key in ["description","title"] { if let v = json[key] { guard let s = v.string else { throw APIError(400, "Invalid schema text") }; try rejectNativeMarkers(s) } }
        if let v = json["enum"] { guard let values = v.array, !values.isEmpty else { throw APIError(400, "Invalid schema enum") } }
        if let v = json["properties"] { guard types.contains("object"), let p = v.members, p.count <= 64 else { throw APIError(400, "Invalid schema properties") }; properties = p }
        else { properties = [] }
        for p in properties { guard safeToolName(p.key) else { throw APIError(400, "Invalid parameter name") }; _ = try ToolSchema(p.value, depth: depth+1) }
        if let v = json["required"] {
            guard types.contains("object"), let list = v.array, list.allSatisfy({ $0.string != nil }), Set(list.compactMap(\.string)).count == list.count else { throw APIError(400, "Invalid required parameters") }
            required = Set(list.compactMap(\.string))
        } else { required = [] }
        guard required.isSubset(of: Set(properties.map(\.key))) else { throw APIError(400, "Required property not defined") }
        if let v = json["additionalProperties"] { guard types.contains("object"), case .bool = v else { throw APIError(400, "Only boolean additionalProperties supported") } }
        if types.contains("array") { guard let item = json["items"] else { throw APIError(400, "Array items required") }; _ = try ToolSchema(item, depth: depth+1) }
        else if json["items"] != nil { throw APIError(400, "items requires array type") }
        self.json = json
    }
    public func accepts(_ value: ToolJSON, depth: Int = 0) -> Bool {
        guard depth <= 16 else { return false }
        let match: Bool
        switch value {
        case .string: match = types.contains("string")
        case .bool: match = types.contains("boolean")
        case .null: match = types.contains("null")
        case .number(let n): guard let value=Double(n),value.isFinite else {return false};match = types.contains("number") || (types.contains("integer") && value.rounded() == value)
        case .array(let values):
            guard types.contains("array"), let raw = json["items"], let schema = try? ToolSchema(raw) else { return false }
            match = values.allSatisfy { schema.accepts($0, depth: depth+1) }
        case .object(let values):
            guard types.contains("object"), required.isSubset(of: Set(values.map(\.key))) else { return false }
            match = values.allSatisfy { member in
                if let p = properties.first(where: { $0.key == member.key }), let schema = try? ToolSchema(p.value) { return schema.accepts(member.value, depth: depth+1) }
                return json["additionalProperties"] != .bool(false)
            }
        }
        guard match else { return false }
        if let allowed = json["enum"]?.array { return allowed.contains(where: { $0.semanticallyEquals(value) }) }
        return true
    }
}
public struct FunctionTool: Sendable {
    public let name: String
    public let schema: ToolSchema
    public let json: ToolJSON
    public init(_ json: ToolJSON) throws {
        guard json["type"]?.string == "function", let top = json.members, Set(top.map(\.key)) == ["type","function"],
              let function = json["function"], let keys = function.members?.map(\.key), Set(keys).isSubset(of: ["name","description","parameters","strict"]),
              let name = function["name"]?.string, safeToolName(name) else { throw APIError(400, "Invalid function tool") }
        if let desc = function["description"] { guard let text = desc.string else { throw APIError(400, "Invalid tool description") }; try rejectNativeMarkers(text) }
        if let strict = function["strict"] { guard strict == .bool(false) else { throw APIError(400, "Strict schema mode is not implemented") } }
        self.name = name
        schema = try ToolSchema(function["parameters"] ?? .object([.init("type",.string("object")),.init("properties",.object([]))]))
        guard schema.types == ["object"] else { throw APIError(400, "Tool arguments must be an object") }
        self.json = json
    }
}
public struct FunctionCall: Sendable, Equatable {
    public let id: String
    public let name: String
    public let arguments: ToolJSON
    public init(id: String = "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""), name: String, arguments: ToolJSON) {
        self.id = id; self.name = name; self.arguments = arguments
    }
    public var wire: [String:Any] { ["id":id,"type":"function","function":["name":name,"arguments":arguments.encoded]] }
    var nativeJSON: ToolJSON { .object([.init("function",.object([.init("name",.string(name)),.init("arguments",arguments)]))]) }
}
public struct ToolMessage: Sendable {
    public let role: String
    public let content: String
    public let calls: [FunctionCall]
    public let resultID: String?
}
public struct ToolConversation: Sendable {
    public let tools: [FunctionTool]
    public let choice: String
    public let messages: [ToolMessage]
    public var allowsCalls: Bool { choice == "auto" && !tools.isEmpty }
    public init(_ root: ToolJSON) throws {
        if let raw = root["tools"] {
            guard let list = raw.array, !list.isEmpty, list.count <= 32 else { throw APIError(400, "Expected 1–32 tools") }
            tools = try list.map(FunctionTool.init)
        } else { tools = [] }
        guard Set(tools.map(\.name)).count == tools.count else { throw APIError(400, "Duplicate tool name") }
        choice = root["tool_choice"]?.string ?? "auto"
        guard root["tool_choice"] == nil || root["tool_choice"]?.string != nil, ["auto","none"].contains(choice), root["tool_choice"] == nil || !tools.isEmpty else { throw APIError(400, "Only tool_choice auto/none with definitions is supported") }
        guard let input = root["messages"]?.array, !input.isEmpty, input.count <= 128 else { throw APIError(400, "Expected 1–128 messages") }
        var result: [ToolMessage] = []; var pending: [String] = []; var allIDs = Set<String>(); var sawUser = false
        for (index, raw) in input.enumerated() {
            guard let fields = raw.members?.map(\.key), let role = raw["role"]?.string, ["system","user","assistant","tool"].contains(role) else { throw APIError(400, "Unsupported message role (developer is not supported by checkpoint)") }
            let allowed: Set<String> = role == "assistant" ? ["role","content","tool_calls"] : role == "tool" ? ["role","content","tool_call_id"] : ["role","content"]
            guard Set(fields).isSubset(of: allowed), role != "system" || index == 0 else { throw APIError(400, "Unsupported message fields/system placement") }
            let content: String
            if let text = raw["content"]?.string { content = text }
            else if role == "assistant", raw["content"] == .null || raw["content"] == nil { content = "" }
            else { throw APIError(400, "Only string message content supported") }
            if role == "assistant" {try rejectNativeMarkers(content)}
            if role == "tool" {
                guard let id = raw["tool_call_id"]?.string, pending.first == id else { throw APIError(400, "Unmatched, duplicate or out-of-order tool result") }
                try rejectNativeMarkers(content); pending.removeFirst()
                result.append(.init(role:role,content:content,calls:[],resultID:id)); continue
            }
            guard pending.isEmpty else { throw APIError(400, "Tool results required before next message") }
            if role == "user" { sawUser = true }
            var calls: [FunctionCall] = []
            if let rawCalls = raw["tool_calls"] {
                guard role == "assistant", sawUser, let list = rawCalls.array, !list.isEmpty, list.count <= 8 else { throw APIError(400, "Invalid assistant tool calls") }
                for item in list {
                    guard Set(item.members?.map(\.key) ?? []) == ["id","type","function"], item["type"]?.string == "function",
                          let id = item["id"]?.string, safeToolName(id), allIDs.insert(id).inserted,
                          let function = item["function"], Set(function.members?.map(\.key) ?? []) == ["name","arguments"],
                          let name = function["name"]?.string, let tool = tools.first(where: { $0.name == name }),
                          let argsText = function["arguments"]?.string else { throw APIError(400, "Invalid call identity, function or arguments") }
                    let args = try ToolJSON.parse(argsText, limit: 65_536)
                    guard args.members != nil, tool.schema.accepts(args) else { throw APIError(400, "History arguments violate tool schema") }
                    try rejectNativeMarkers(args.encoded)
                    calls.append(.init(id:id,name:name,arguments:args)); pending.append(id)
                }
            }
            result.append(.init(role:role,content:content,calls:calls,resultID:nil))
        }
        guard sawUser, pending.isEmpty, let last = result.last, last.role == "user" || last.role == "tool" else { throw APIError(400, "Expected user or completed tool results before generation") }
        messages = result
    }
}
