import Foundation
import Testing
import QwenFieldfareCore
@testable import QwenFieldfareServerSupport

struct PrefixBoundaryInvestigationTests {
    @Test func ordinaryHistoryChangesEarlierAssistantPreamble() {
        let first=QwenChatTemplate.format(messages:[],appendingUser:"first",enableThinking:false)
        let second=QwenChatTemplate.format(messages:[.init(role:.user,content:"first"),.init(role:.assistant,content:"hello")],appendingUser:"next",enableThinking:false)
        #expect(!second.hasPrefix(first))
        #expect(first.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        #expect(second.contains("<|im_start|>assistant\nhello<|im_end|>"))
    }
    @Test func canonicalSequentialToolsCanPreserveTheFramingPrefix() throws {
        let url=try #require(Bundle.module.url(forResource:"chat_template",withExtension:"jinja",subdirectory:"Fixtures"))
        let renderer=try ToolPrompt(source:String(contentsOf:url,encoding:.utf8))
        let tool:[String:Any]=["type":"function","function":["name":"audit_chunk","parameters":["type":"object","properties":["index":["type":"integer"]],"required":["index"]]]]
        let user:[String:Any]=["role":"user","content":"Read three chunks sequentially."]
        func render(_ messages:[[String:Any]]) throws -> String {
            try renderer.render(ToolConversation(ToolJSON.parse(String(decoding:wire(["messages":messages,"tools":[tool]]),as:UTF8.self))))
        }
        let first=try render([user])
        let native="<tool_call>\n<function=audit_chunk>\n<parameter=index>\n1\n</parameter>\n</function>\n</tool_call>"
        let assistant:[String:Any]=["role":"assistant","content":"","tool_calls":[["id":"call_one","type":"function","function":["name":"audit_chunk","arguments":"{\"index\":1}"]]]]
        let second=try render([user,assistant,["role":"tool","tool_call_id":"call_one","content":"chunk one"]])
        #expect(second.hasPrefix(first+native))
        // Byte equality is only a necessary condition: runtime must compare actual
        // token IDs, including BPE seams, and prove every represented state token.
        #expect(!second.hasPrefix(first+native+"\n"))
    }
}
