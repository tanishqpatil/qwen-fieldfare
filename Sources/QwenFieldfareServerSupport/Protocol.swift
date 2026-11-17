import Foundation
import QwenFieldfareCore

public struct APIError: Error, Sendable {
    public let status: Int
    public let message: String
    public init(_ status: Int, _ message: String) { self.status = status; self.message = message }
    public var data: Data { wire(["error":["message":message,"type":status < 500 ? "invalid_request_error":"server_error","code":"http_\(status)"]]) }
}
public func wire(_ value: [String:Any]) -> Data { (try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])) ?? Data() }
public func sse(_ value: [String:Any]) -> Data { Data("data: ".utf8) + wire(value) + Data("\n\n".utf8) }
public struct ChatRequest: Sendable {
    public let messages: [QwenChatTemplate.Message]
    public let stream: Bool
    public let includeUsage: Bool
    public let configuration: QwenGenerationConfiguration
    public init(data: Data, model: String) throws {
        guard data.count <= 1_048_576,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String:Any] else { throw APIError(400,"Invalid JSON body") }
        let allowed: Set<String> = ["model","messages","stream","stream_options","temperature","top_p","top_k","max_tokens","max_completion_tokens","presence_penalty","repetition_penalty","seed","n"]
        guard Set(json.keys).isSubset(of: allowed) else { throw APIError(400,"Unsupported request parameter") }
        guard json["model"] as? String == model else { throw APIError(404,"Unknown model") }
        guard let list = json["messages"] as? [[String:Any]], !list.isEmpty, list.count <= 128 else { throw APIError(400,"Expected 1–128 messages") }
        messages = try list.map { item in
            guard Set(item.keys) == ["role","content"], let role = item["role"] as? String,
                  let r = QwenChatTemplate.Message.Role(rawValue: role), let content = item["content"] as? String else { throw APIError(400,"Only system/user/assistant text messages are supported") }
            return .init(role:r,content:content)
        }
        guard messages.last?.role == .user else { throw APIError(400,"Final message must be user") }
        func number(_ key: String, _ fallback: Double) throws -> Double {
            guard let v = json[key] else { return fallback }
            guard let n = v as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite else { throw APIError(400,"Invalid numeric parameter") }
            return n.doubleValue
        }
        func integer(_ key: String, _ fallback: Int) throws -> Int {
            let n = try number(key,Double(fallback)); guard n.rounded() == n, n >= 0, n < Double(Int.max) else { throw APIError(400,"Invalid integer parameter") }; return Int(n)
        }
        if let v = json["stream"] { guard CFGetTypeID(v as CFTypeRef) == CFBooleanGetTypeID() else { throw APIError(400,"stream must be boolean") } }
        stream = json["stream"] as? Bool ?? false
        if let options = json["stream_options"] {
            guard stream, let o = options as? [String:Any], Set(o.keys).isSubset(of:["include_usage"]),
                  o["include_usage"] == nil || CFGetTypeID(o["include_usage"]! as CFTypeRef) == CFBooleanGetTypeID() else { throw APIError(400,"Invalid stream_options") }
            includeUsage = o["include_usage"] as? Bool ?? false
        } else { includeUsage = false }
        guard try integer("n",1) == 1, !(json["max_tokens"] != nil && json["max_completion_tokens"] != nil) else { throw APIError(400,"Only n=1 and one output limit are supported") }
        let max = try integer(json["max_tokens"] != nil ? "max_tokens":"max_completion_tokens",256)
        guard (1...4096).contains(max) else { throw APIError(400,"Output limit must be 1–4096") }
        configuration = .init(maxNewTokens:max,temperature:Float(try number("temperature",1)),topK:try integer("top_k",20),topP:Float(try number("top_p",0.95)),presencePenalty:Float(try number("presence_penalty",0)),repetitionPenalty:Float(try number("repetition_penalty",1)),seed:json["seed"] == nil ? nil : UInt64(try integer("seed",0)))
        do { try configuration.validate() } catch { throw APIError(400,"Invalid sampling configuration") }
    }
}
import CoreFoundation
