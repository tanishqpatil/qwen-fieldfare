import Foundation
import Observation
import QwenFieldfareServerSupport

@MainActor @Observable public final class ServerController {
    public enum State: String {case stopped, loading, ready, generating, stopping, failed}
    public typealias Factory = @Sendable (ServerConfiguration) async throws -> any ManagedServerSession
    public var settings: ServerSettings {didSet {if let data = try? JSONEncoder().encode(settings) {defaults.set(data,forKey:Self.settingsKey)}}}
    public private(set) var state: State = .stopped
    public private(set) var snapshot: ServerHealth?
    public private(set) var lastError: String?
    public private(set) var isExternal = false
    public private(set) var blocksNativeChat = false
    public private(set) var boundPort: Int?
    public private(set) var authenticationStatus = "Stored in Keychain when configured"
    public static let settingsKey = "QwenFieldfare.server.settings.v1"
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let credentials: any ServerCredentialStore
    @ObservationIgnored private let factory: Factory
    @ObservationIgnored private let prepareChat: @MainActor () async throws -> Void
    @ObservationIgnored private let resumeChat: @MainActor () -> Void
    @ObservationIgnored private var session: (any ManagedServerSession)?
    @ObservationIgnored private var operation: Task<Void,Never>?
    @ObservationIgnored private var loading: Task<any ManagedServerSession,Error>?
    @ObservationIgnored private var polling: Task<Void,Never>?
    @ObservationIgnored private var observationKey: String?
    @ObservationIgnored private var stopRequested = false
    @ObservationIgnored private let pollInterval: Duration

    public init(modelPath: String,defaults: UserDefaults = .standard,credentials: (any ServerCredentialStore)? = nil,
                factory: @escaping Factory = InProcessServerSession.start,
                pollInterval: Duration = .seconds(1),
                prepareChat: @escaping @MainActor () async throws -> Void = {},resumeChat: @escaping @MainActor () -> Void = {}) {
        self.defaults = defaults;self.credentials = credentials ?? KeychainServerCredential();self.factory = factory
        self.prepareChat = prepareChat;self.resumeChat = resumeChat;self.pollInterval = pollInterval
        if let data = defaults.data(forKey:Self.settingsKey),let saved = try? JSONDecoder().decode(ServerSettings.self,from:data) {settings = saved}
        else {settings = ServerSettings(modelPath:modelPath)}
    }
    public var canConfigure: Bool {operation == nil && !blocksNativeChat}
    public var canStart: Bool {canConfigure && settings.validationMessage == nil}
    public var canStop: Bool {blocksNativeChat && state != .stopping}
    public var apiBaseURL: String? {
        let port = boundPort ?? settings.port
        return port > 0 && port <= 65535 ? "http://127.0.0.1:\(port)/v1" : nil
    }
    public func start() {
        guard canStart else {return}
        state = .loading;lastError = nil;stopRequested = false;blocksNativeChat = true
        operation = Task {await performStart()}
    }
    private func performStart() async {
        defer {operation = nil}
        do {
            let key = try credential()
            let config = try configuration(settings,key:key)
            try await prepareChat()
            if stopRequested {finishStopped();return}
            let factory = factory
            let task = Task.detached {try await factory(config)}
            loading = task
            session = try await task.value;loading = nil
            if stopRequested {await releaseOwned();return}
            snapshot = try await session!.health();boundPort = session!.port
            updateState();startPolling()
        } catch {
            loading = nil
            lastError = stopRequested ? nil : managementMessage(error)
            await releaseOwned(failure:lastError)
        }
    }
    /// Stop only this controller's server. External mode is observation only.
    public func stop() {
        guard blocksNativeChat || operation != nil else {return}
        stopRequested = true;state = .stopping;polling?.cancel();polling = nil;loading?.cancel()
        if operation != nil {return} // loading operation owns its cleanup
        operation = Task {await releaseOwned();operation = nil}
    }
    public func shutdown() async -> Bool {
        stop()
        while operation != nil {try? await Task.sleep(for:.milliseconds(10))}
        return !blocksNativeChat
    }
    public func clearPrefixCache() async {
        guard !isExternal,let session,state == .ready || state == .generating else {return}
        do {try await session.clearPrefixCache();snapshot=try await session.health()}
        catch {report(error)}
    }
    public func restart() async {
        guard !isExternal else {return}
        if await shutdown() {start()}
    }
    private func releaseOwned(failure: String? = nil) async {
        do {
            try await session?.stop()
            session = nil
            finishStopped()
            if let failure {lastError = failure;state = .failed}
        } catch {lastError = managementMessage(error);state = .failed} // retain owner, keep chat blocked, allow retry
    }
    private func finishStopped() {
        session = nil;snapshot = nil;boundPort = nil;observationKey = nil;isExternal = false
        blocksNativeChat = false;resumeChat();state = .stopped
    }
    public func observeExternal() {
        guard canStart,settings.port > 0 else {return}
        state = .loading;lastError = nil;stopRequested = false;blocksNativeChat = true
        operation = Task {
            defer {operation = nil}
            do {
                let key = try credential()
                let health = try await LoopbackServerObserver.health(port:settings.port,key:key)
                try await prepareChat()
                if stopRequested {finishStopped();return}
                isExternal = true;observationKey = key;boundPort = settings.port;snapshot = health
                updateState();startPolling()
            } catch {finishStopped();state = .failed;lastError = managementMessage(error)}
        }
    }
    private func updateState() {state = (snapshot?.active_generations ?? 0) > 0 ? .generating : .ready}
    private func startPolling() {
        polling?.cancel()
        polling = Task { [weak self] in
            while !Task.isCancelled {
                do {try await Task.sleep(for:self?.pollInterval ?? .seconds(1));try Task.checkCancellation()}
                catch {return}
                guard let self else {return}
                do {
                    let health: ServerHealth
                    if let session {health = try await session.health()}
                    else if let key = observationKey,let port = boundPort {health = try await LoopbackServerObserver.health(port:port,key:key)}
                    else {return}
                    guard !Task.isCancelled,self.state != .stopping else {return}
                    self.snapshot = health;self.updateState()
                } catch {
                    guard !Task.isCancelled else {return}
                    self.lastError = "Server diagnostics became unavailable. Stop or disconnect before retrying."
                    if self.isExternal {self.state = .failed} // never stop the independent process
                    else {
                        self.stopRequested = true;self.state = .stopping
                        self.operation = Task {await self.releaseOwned(failure:self.lastError);self.operation = nil}
                    }
                    return
                }
            }
        }
    }
    private func credential() throws -> String {
        let key: String
        if let existing = try credentials.read() {key = existing}
        else {key = try KeychainServerCredential.generate();try credentials.write(key)}
        _ = try configuration(ServerSettings(modelPath:"credential-validation"),key:key)
        authenticationStatus = "Configured in Keychain; bearer authentication required"
        return key
    }
    /// Call only after an explicit reveal/copy/export confirmation in the UI.
    public func credentialForExplicitAction() throws -> String {try credential()}
    public func importCLIKeyFile(_ path: String) throws {
        guard canConfigure else {throw ManagementError.invalid("Stop or disconnect before changing authentication.")}
        let config = try ServerSettings(modelPath:"credential-validation").configuration(keyFile:path)
        try credentials.write(config.apiKey);authenticationStatus = "CLI key imported into Keychain"
    }
    public func report(_ error: Error) {lastError = managementMessage(error)}
    public var providerConfigurationJSON: String {
        let base = apiBaseURL ?? "http://127.0.0.1:PORT/v1"
        let context = snapshot?.diagnostics.context ?? settings.context
        let model = snapshot?.model ?? "qwen-fieldfare"
        let permissions: [String:Any] = ["*":"deny","read":"allow","grep":"allow","edit":"ask","bash":"ask","external_directory":"deny"]
        let value: [String:Any] = [
            "provider":["fieldfare":["npm":"@ai-sdk/openai-compatible","options":["baseURL":base,"apiKey":"{env:QWENFIELDFARE_API_KEY}","timeout":600000],"models":[model:["limit":["context":context,"output":min(1024,max(1,context/2))],"tool_call":true]]]],
            "model":"fieldfare/\(model)","small_model":"fieldfare/\(model)",
            "agent":["fieldfare":["mode":"primary","prompt":"Work only in the current workspace. Use enabled tools one at a time. Never delegate.","temperature":0,"steps":16,"permission":permissions]],
            "autoupdate":false,"share":"disabled"
        ]
        guard let data = try? JSONSerialization.data(withJSONObject:value,options:[.sortedKeys,.withoutEscapingSlashes]),let json = String(data:data,encoding:.utf8) else {return "{}"}
        return json
    }
    public var providerGuidance: String {
        """
        OpenCode 1.19.5 (v1 config). Set QWENFIELDFARE_API_KEY privately to the matching key.
        Save this JSON as opencode.json in the intended workspace:
        \(providerConfigurationJSON)
        Run: opencode run --pure --agent fieldfare --dir /absolute/workspace 'YOUR TASK'
        Read/grep are permitted inside the workspace; edits/shell require client approval. No delegation or external-directory access. See AGENT_INTEGRATION.md for isolated XDG/instruction settings. No tool execution in this app/server.
        """
    }
}
