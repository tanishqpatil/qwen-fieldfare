import Foundation
import Testing
@testable import QwenFieldfareServerManagement

struct SettingsTests {
    @Test func validationReusesCLIAndPrivateFile() throws {
        var settings = ServerSettings(modelPath:"/missing")
        #expect(settings.validationMessage == nil)
        for port in [-1,65536] {settings.port = port;#expect(settings.validationMessage != nil)}
        settings.port = 0;settings.context = Int.max;#expect(settings.validationMessage != nil)
        settings.context = 8192
        let config = try configuration(settings,key:"unit-test-key-not-secret")
        #expect(config.port == 0 && config.apiKey == "unit-test-key-not-secret")
        #expect(throws:(any Error).self) {try configuration(settings,key:"bad\nkey")}
    }
}
