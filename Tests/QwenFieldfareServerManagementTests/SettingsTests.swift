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

extension SettingsTests {
    @Test func olderPersistedSettingsKeepValuesAndDefaultCacheOff() throws {
        let settings=try JSONDecoder().decode(ServerSettings.self,from:Data(#"{"modelPath":"/old/model","port":18094,"context":16384,"memoryGiB":4,"queueCapacity":1}"#.utf8))
        #expect(settings.port == 18094 && settings.context == 16384 && settings.memoryGiB == 4)
        #expect(settings.prefixCacheMode == .off)
        var exact=settings;exact.prefixCacheMode = .exactPrefix
        #expect(try configuration(exact,key:"settings-cache-test-key").prefixCacheMode == .exactPrefix)
        #expect(try JSONDecoder().decode(ServerSettings.self,from:JSONEncoder().encode(exact)) == exact)
    }
}
