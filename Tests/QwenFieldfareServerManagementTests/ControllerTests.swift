import Foundation
import Testing
import QwenFieldfareCore
import QwenFieldfareServerSupport
@testable import QwenFieldfareServerManagement

@MainActor private final class MemoryCredential: ServerCredentialStore {
    var key: String? = "test-credential-no-production-secret"
    func read() -> String? {key}
    func write(_ key: String) {self.key = key}
}
private struct ManagementMockBackend: GenerationBackend {
    func generate(_ request: ChatRequest,emit: @escaping @Sendable (String) -> Void) async throws -> Completion {
        emit("hello");return Completion(text:"hello",prompt:4,count:1,reason:"stop")
    }
}
private final class PrefixControlBackend: GenerationBackend, @unchecked Sendable {
    private let lock=NSLock();private var clears=0
    var count:Int {lock.withLock {clears}}
    func invalidatePrefixCache(_ reason:QwenPrefixInvalidation) {if reason == .explicitClear {lock.withLock {clears+=1}}}
    func generate(_ request:ChatRequest,emit:@escaping @Sendable (String)->Void) async throws -> Completion {
        Completion(text:"hello",prompt:2,count:1,reason:"stop")
    }
}
@MainActor @Suite(.serialized) struct ControllerTests {
    func defaults() -> UserDefaults {UserDefaults(suiteName:"qwen-controller-test-\(UUID())")!}
    func wait(_ controller: ServerController) async throws {
        for _ in 0..<1000 {
            if controller.state != .loading && controller.state != .stopping {return}
            try await Task.sleep(for:.milliseconds(10))
        }
        Issue.record("Controller did not settle")
    }
    func factory(_ config: ServerConfiguration) async throws -> any ManagedServerSession {
        let server = try LocalServer(backend:ManagementMockBackend(),key:config.apiKey,model:config.modelID)
        return InProcessServerSession(server:server,port:try await server.start(port:0))
    }
    @Test func startStopRestartPersistenceAndRedaction() async throws {
        let store = defaults();let key = MemoryCredential();var suspended = false
        let controller = ServerController(modelPath:"/test-model",defaults:store,credentials:key,factory:{config in
            let server = try LocalServer(backend:ManagementMockBackend(),key:config.apiKey,model:config.modelID)
            return InProcessServerSession(server:server,port:try await server.start(port:0))
        },prepareChat:{suspended = true},resumeChat:{suspended = false})
        controller.settings.port = 0;controller.start();try await wait(controller)
        #expect(controller.state == .ready && suspended && controller.blocksNativeChat)
        let port = try #require(controller.boundPort)
        let health = try await LoopbackServerObserver.health(port:port,key:key.key!)
        #expect(health.service == "QwenFieldfareServer")
        #expect(!controller.providerGuidance.contains(key.key!))
        #expect(!String(describing:store.dictionaryRepresentation()).contains(key.key!))
        await controller.restart();try await wait(controller);#expect(controller.state == .ready)
        #expect(await controller.shutdown());#expect(!suspended && !controller.blocksNativeChat)
        let reopened = ServerController(modelPath:"other",defaults:store,credentials:key)
        #expect(reopened.state == .stopped && reopened.settings.port == 0 && reopened.settings.modelPath == "/test-model")
    }
    @Test func invalidAuthAndMissingModelAreRecoverable() async throws {
        let key = MemoryCredential();key.key = "short"
        let controller = ServerController(modelPath:"/missing",defaults:defaults(),credentials:key)
        controller.start();try await wait(controller)
        #expect(controller.state == .failed && !controller.blocksNativeChat)
        #expect(!(controller.lastError ?? "").contains("short"))
        key.key = "test-valid-credential-only";controller.settings.port = 0
        controller.start();try await wait(controller)
        #expect(controller.state == .failed && !controller.blocksNativeChat)
        #expect(await controller.shutdown())
    }
    @Test func occupiedPortRejectsBeforeModelLoad() async throws {
        let key = MemoryCredential();let server = try LocalServer(backend:ManagementMockBackend(),key:key.key!,model:"qwen-fieldfare")
        let port = try await server.start(port:0)
        let controller = ServerController(modelPath:"/missing",defaults:defaults(),credentials:key)
        controller.settings.port = port;controller.start();try await wait(controller)
        #expect(controller.state == .failed && (controller.lastError ?? "").contains("port"))
        try await server.shutdown()
    }
    @Test func observingCLIHasNoStopAuthority() async throws {
        let key = MemoryCredential();let server = try LocalServer(backend:ManagementMockBackend(),key:key.key!,model:"qwen-fieldfare")
        let port = try await server.start(port:0)
        let controller = ServerController(modelPath:"/test-model",defaults:defaults(),credentials:key,pollInterval:.milliseconds(10))
        controller.settings.port = port;controller.observeExternal();try await wait(controller)
        #expect(controller.isExternal && controller.blocksNativeChat && controller.state == .ready)
        #expect(await controller.shutdown())
        #expect(try await LoopbackServerObserver.health(port:port,key:key.key!).status == "ready")
        try await server.shutdown()
    }
    @Test func ownedPrefixClearAndPersistedModeHaveNoExternalAuthority() async throws {
        let backend=PrefixControlBackend();let store=defaults();let key=MemoryCredential()
        let owner=ServerController(modelPath:"/test-model",defaults:store,credentials:key,factory:{config in
            #expect(config.prefixCacheMode == .exactPrefix)
            let server=try LocalServer(backend:backend,key:config.apiKey,model:config.modelID)
            return InProcessServerSession(server:server,port:try await server.start(port:0))
        })
        owner.settings.prefixCacheMode = .exactPrefix;owner.settings.port=0
        owner.start();try await wait(owner)
        #expect(owner.state == .ready)
        await owner.clearPrefixCache();#expect(backend.count == 1)
        let observer=ServerController(modelPath:"/test-model",defaults:defaults(),credentials:key)
        observer.settings.port=try #require(owner.boundPort);observer.observeExternal();try await wait(observer)
        #expect(observer.isExternal && observer.state == .ready)
        await observer.clearPrefixCache();#expect(backend.count == 1)
        #expect(await observer.shutdown());#expect(owner.state == .ready)
        #expect(await owner.shutdown())
        let reopened=ServerController(modelPath:"other",defaults:store,credentials:key)
        #expect(reopened.settings.prefixCacheMode == .exactPrefix && reopened.state == .stopped)
    }
    @Test func stoppingWhileLoadingCancelsAndResumesChat() async throws {
        let controller = ServerController(modelPath:"/test-model",defaults:defaults(),credentials:MemoryCredential(),factory:{_ in
            try await Task.sleep(for:.seconds(30));throw ManagementError.connection
        })
        controller.start();try await Task.sleep(for:.milliseconds(30));controller.stop()
        #expect(await controller.shutdown());#expect(controller.state == .stopped && !controller.blocksNativeChat)
    }
    @Test func privateKeyExportImportAndUnsafeFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("key-export-\(UUID())")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        defer {try? FileManager.default.removeItem(at:root)}
        let path = root.appendingPathComponent("key").path
        try exportPrivateKey("test-key-for-cli-interop",to:path)
        let controller = ServerController(modelPath:"/test-model",defaults:defaults(),credentials:MemoryCredential())
        try controller.importCLIKeyFile(path)
        #expect(try controller.credentialForExplicitAction() == "test-key-for-cli-interop")
        #expect(throws:(any Error).self) {try exportPrivateKey("another-key",to:path)}
        try FileManager.default.setAttributes([.posixPermissions:0o644],ofItemAtPath:path)
        #expect(throws:(any Error).self) {try controller.importCLIKeyFile(path)}
    }
}

private actor SessionCapture {
    var session: (any ManagedServerSession)?
    func set(_ value: any ManagedServerSession) {session = value}
}
extension ControllerTests {
    @Test func unexpectedRuntimeExitRecoversAndCanReload() async throws {
        let capture=SessionCapture()
        let controller=ServerController(modelPath:"/test-model",defaults:defaults(),credentials:MemoryCredential(),factory:{config in
            let server=try LocalServer(backend:ManagementMockBackend(),key:config.apiKey,model:config.modelID)
            let session=InProcessServerSession(server:server,port:try await server.start(port:0))
            await capture.set(session);return session
        },pollInterval:.milliseconds(10))
        controller.start();try await wait(controller)
        try await capture.session?.stop()
        for _ in 0..<100 where controller.state != .failed {try await Task.sleep(for:.milliseconds(10))}
        #expect(controller.state == .failed && !controller.blocksNativeChat)
        controller.start();try await wait(controller);#expect(controller.state == .ready)
        #expect(await controller.shutdown())
    }
    @Test func externalHealthAuthenticationAndResponseBound() async throws {
        let key="bounded-health-test-key"
        let server=try LocalServer(backend:ManagementMockBackend(),key:key,model:String(repeating:"m",count:70000))
        let port=try await server.start(port:0)
        await #expect(throws:(any Error).self) {try await LoopbackServerObserver.health(port:port,key:"wrong-key")}
        await #expect(throws:(any Error).self) {try await LoopbackServerObserver.health(port:port,key:key)}
        try await server.shutdown()
    }
}

extension ControllerTests {
    @Test func providerGuidanceUsesBoundedAgentAndEscapesObservedModelID() async throws {
        let controller=ServerController(modelPath:"/test-model",defaults:defaults(),credentials:MemoryCredential(),factory:{config in
            let server=try LocalServer(backend:ManagementMockBackend(),key:config.apiKey,model:"custom\"model")
            return InProcessServerSession(server:server,port:try await server.start(port:0))
        })
        controller.start();try await wait(controller)
        let data=Data(controller.providerConfigurationJSON.utf8)
        let config=try #require(try JSONSerialization.jsonObject(with:data) as? [String:Any])
        #expect(config["model"] as? String == "fieldfare/custom\"model")
        let agent=try #require((config["agent"] as? [String:Any])?["fieldfare"] as? [String:Any])
        let permissions=try #require(agent["permission"] as? [String:String])
        #expect(permissions["*"] == "deny" && permissions["edit"] == "ask" && permissions["external_directory"] == "deny")
        #expect(!controller.providerConfigurationJSON.contains("test-credential-no-production-secret"))
        if let output=ProcessInfo.processInfo.environment["QWENFIELDFARE_GUIDANCE_TEST_OUTPUT"] {
            try data.write(to:URL(fileURLWithPath:output))
        }
        #expect(await controller.shutdown())
    }
}
