import AppKit
import SwiftUI
import QwenFieldfareCore
import QwenFieldfareServerManagement

struct ServerSettingsView: View {
    @Bindable var controller: ServerController
    @Environment(\.dismiss) private var dismiss
    @State private var confirmCredential = false
    @State private var credentialAction = "copy"
    @State private var revealedKey: String?

    var body: some View {
        VStack(spacing:0) {
            HStack {
                Label("Local Server",systemImage:"network").font(.title2.bold())
                Spacer()
                Button("Done") {dismiss()}.keyboardShortcut(.cancelAction)
            }.padding()
            Divider()
            Form {
                Section("Server mode") {
                    LabeledContent("Status") {
                        Label(controller.state.rawValue.capitalized,systemImage:controller.state == .ready ? "checkmark.circle.fill":"circle.fill")
                            .foregroundStyle(controller.state == .failed ? .red : .secondary)
                            .accessibilityIdentifier("server-status")
                    }
                    LabeledContent("Process",value:controller.isExternal ? "Independent CLI — observation only":"This application — no child process")
                    Text("Starting cancels and unloads native chat. Chat stays disabled until the server stops or observation disconnects.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button("Start Server") {controller.start()}.disabled(!controller.canStart).accessibilityIdentifier("server-start")
                        Button(controller.isExternal ? "Disconnect":"Stop Server") {controller.stop()}.disabled(!controller.canStop).accessibilityIdentifier("server-stop")
                        Button("Restart") {Task {await controller.restart()}}.disabled(controller.isExternal || controller.state != .ready)
                        Button("Observe CLI") {controller.observeExternal()}.disabled(!controller.canStart || controller.settings.port == 0)
                    }
                    if let error = controller.lastError {Text(error).foregroundStyle(.red).textSelection(.enabled)}
                }
                Section("Configuration — applied on next start") {
                    HStack {
                        TextField("Model directory",text:$controller.settings.modelPath)
                        Button("Choose…",action:chooseModel)
                    }
                    LabeledContent("Host",value:"127.0.0.1 (loopback only)")
                    TextField("Port (0 chooses an available port)",value:$controller.settings.port,format:.number.grouping(.never))
                    Picker("Context tokens",selection:$controller.settings.context) {
                        ForEach([4096,8192,16384,32768,65536],id:\.self) {context in
                            Text(context.formatted()).tag(context).disabled(!fits(context))
                        }
                    }
                    Picker("Active-streaming Metal target",selection:$controller.settings.memoryGiB) {
                        Text("3 GiB").tag(3);Text("4 GiB").tag(4);Text("5 GiB").tag(5)
                    }
                    Picker("Conversation state cache",selection:$controller.settings.prefixCacheMode) {
                        Text("Off — reset and replay").tag(QwenPrefixCacheMode.off)
                        Text("Exact prefix — verified in-place reuse").tag(QwenPrefixCacheMode.exactPrefix)
                    }
                    Text("Default off. Reuse requires the entire consumed token sequence to match; ordinary chat framing can force replay. No recurrent rewind or extra KV snapshot.").font(.caption).foregroundStyle(.secondary)
                    Stepper("Queue limit: \(controller.settings.queueCapacity)",value:$controller.settings.queueCapacity,in:0...8)
                    if let message = controller.settings.validationMessage {Text(message).foregroundStyle(.red)}
                    Text("Stop before changing settings. No automatic startup on reopening. Larger contexts are allocation choices, not occupied-history qualification.").font(.caption).foregroundStyle(.secondary)
                }.disabled(!controller.canConfigure)
                Section("Authentication and clients") {
                    Text(controller.authenticationStatus).font(.callout)
                    LabeledContent("API base URL",value:controller.apiBaseURL ?? "Assigned after start")
                    HStack {
                        Button("Copy API URL") {if let url = controller.apiBaseURL {copy(url)}}.disabled(controller.apiBaseURL == nil)
                        Button("Copy OpenCode Guidance") {copy(controller.providerGuidance)}
                    }
                    HStack {
                        Button("Import CLI Key…",action:importKey).disabled(!controller.canConfigure)
                        Button("Copy Key…") {credentialAction = "copy";confirmCredential = true}
                        Button("Reveal Key…") {credentialAction = "reveal";confirmCredential = true}
                        Button("Export Key File…") {credentialAction = "export";confirmCredential = true}
                    }
                    if let revealedKey {
                        Text(revealedKey).font(.system(.caption,design:.monospaced)).textSelection(.enabled)
                        Button("Hide Key") {self.revealedKey = nil}
                    }
                    Text("Keys grant local inference access. Guidance contains only an environment placeholder. OpenCode executes tools with its own permissions; this app/server executes none.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let health = controller.snapshot {
                    Section("Server diagnostics") {
                        LabeledContent("Model",value:health.model)
                        LabeledContent("Model readiness",value:health.status)
                        LabeledContent("Active generation / queued",value:"\(health.active_generations) / \(health.queued_requests) (limit \(health.queue_capacity))")
                        LabeledContent("Completed / rejected requests",value:"\(health.requests.completed) / \(health.requests.rejected)")
                        LabeledContent("Failed / cancelled admitted requests",value:"\(health.requests.failed) / \(health.requests.cancelled)")
                        LabeledContent("Current context",value:health.diagnostics.context.map {String($0)} ?? "Unavailable")
                        LabeledContent("Current Metal target",value:health.diagnostics.memoryBudgetGiB.map {"\($0) GiB"} ?? "Unavailable")
                        LabeledContent("Observed Metal allocation",value:health.diagnostics.metalAllocatedBytes.map {String(format:"%.3f GiB",Double($0)/1073741824)} ?? "Unavailable")
                        Text("Metal allocation is not total process memory or a measured peak. Counters reset when the server starts; rejected includes HTTP validation/authentication errors.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let cache = health.diagnostics.prefixCache {
                        Section("Conversation prefix cache") {
                            LabeledContent("Mode / committed state",value:"\(cache.mode.rawValue) / \(cache.valid ? "valid":"not reusable")")
                            LabeledContent("Committed consumed tokens",value:String(cache.consumedTokens))
                            LabeledContent("Context capacity",value:"\(cache.maximumTokens) tokens")
                            LabeledContent("Last reused / fresh prompt tokens",value:"\(cache.reusedPrefixTokens) / \(cache.processedPromptTokens)")
                            LabeledContent("Hits / misses / invalidations",value:"\(cache.hits) / \(cache.misses) / \(cache.invalidations)")
                            LabeledContent("Full replays",value:String(cache.fullReplays))
                            LabeledContent("Last invalidation",value:cache.lastInvalidationReason?.rawValue ?? "None")
                            LabeledContent("Token element capacity",value:"\(cache.tokenStorageBytes) bytes (no Metal snapshot)")
                            LabeledContent("Runner session",value:String(cache.session.prefix(8)))
                            Button("Clear Prefix Cache") {Task {await controller.clearPrefixCache()}}
                                .disabled(controller.isExternal || (controller.state != .ready && controller.state != .generating))
                            Text("Clears reusable metadata, not expert weights. In-flight work may finish but cannot commit the cleared prefix. External CLI observation has no control authority.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if let timing = health.requests.lastRequest {
                        Section("Last finished request — \(timing.outcome)") {
                            metric("Receipt through admission",timing.admissionMs)
                            metric("Admission queue wait",timing.queueWaitMs)
                            metric("Prompt formatting",timing.formattingMs)
                            metric("Tokenization",timing.tokenizationMs)
                            LabeledContent("Prompt / output tokens",value:"\(timing.promptTokens.map(String.init) ?? "—") / \(timing.outputTokens.map(String.init) ?? "—")")
                            LabeledContent("Actual fresh prefill tokens",value:timing.actualPrefillTokens.map(String.init) ?? "Unavailable")
                            metric("Prefill",timing.prefillMs)
                            metric("First generated token (from request)",timing.firstGeneratedTokenMs)
                            metric("First visible SSE content (flush)",timing.firstVisibleSSEMs)
                            metric("Validated tool buffering",timing.parserBufferMs)
                            metric("Decode after first token",timing.decodeAfterFirstTokenMs)
                            metric("Overall request",timing.overallMs)
                            Text("Unavailable values were not observed. SSE flush timing is server-side, not client receipt. Diagnostics retain one timing record and no conversation or tool arguments.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }.formStyle(.grouped)
        }
        .frame(width:760,height:760)
        .confirmationDialog("Bearer credentials allow local inference access. Copy, reveal or export only to a trusted client. Exported files remain your responsibility.",isPresented:$confirmCredential,titleVisibility:.visible) {
            Button("Continue") {credentialOperation()}
            Button("Cancel",role:.cancel) {}
        }
        .onDisappear {revealedKey = nil}
    }
    @ViewBuilder private func metric(_ title: String,_ value: Double?) -> some View {
        LabeledContent(title,value:value.map {String(format:"%.2f ms",$0)} ?? "Unavailable")
    }
    private func fits(_ context: Int) -> Bool {
        guard let budget = QwenActiveStreamingMemoryBudget(rawValue:controller.settings.memoryGiB) else {return false}
        return (try? QwenActiveStreamingMemoryConfiguration.make(budget:budget,contextLength:context)) != nil
    }
    private func chooseModel() {
        let panel = NSOpenPanel();panel.canChooseDirectories = true;panel.canChooseFiles = false;panel.allowsMultipleSelection = false
        if panel.runModal() == .OK,let url = panel.url {controller.settings.modelPath = url.path}
    }
    private func importKey() {
        let panel = NSOpenPanel();panel.title = "Import owned private CLI key file";panel.canChooseFiles = true;panel.canChooseDirectories = false;panel.allowsMultipleSelection = false
        if panel.runModal() == .OK,let url = panel.url {do {try controller.importCLIKeyFile(url.path)} catch {controller.report(error)}}
    }
    private func credentialOperation() {
        do {
            let key = try controller.credentialForExplicitAction()
            switch credentialAction {
            case "copy": copy(key)
            case "reveal": revealedKey = key
            default:
                let panel = NSSavePanel();panel.title = "Export private CLI key (0600)";panel.nameFieldStringValue = "qwenfieldfare-api-key"
                if panel.runModal() == .OK,let url = panel.url {try exportPrivateKey(key,to:url.path)}
            }
        } catch {controller.report(error)}
    }
    private func copy(_ text: String) {NSPasteboard.general.clearContents();NSPasteboard.general.setString(text,forType:.string)}
}
