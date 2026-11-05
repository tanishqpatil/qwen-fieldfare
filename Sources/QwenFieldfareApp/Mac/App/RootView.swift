import AppKit
import SwiftUI
import QwenFieldfareCore

struct RootView: View {
    let chatService: ChatService
    @State private var userMessage: String = ""
    @State private var showInspector: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            // Header / Navigation Bar
            headerView
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            // Main Split: Chat Canvas & Optional Inspector
            HStack(spacing: 0) {
                // Chat Canvas Area
                VStack(spacing: 0) {
                    // System Banners (Errors / Memory Warnings)
                    if let error = chatService.errorMessage {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                            Text(error)
                                .font(.callout)
                                .foregroundStyle(.red)
                            Spacer()
                            Button {
                                chatService.errorMessage = nil
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.1))
                    }

                    if let warning = chatService.memoryWarning {
                        HStack(spacing: 8) {
                            Image(systemName: "info.circle.fill")
                                .foregroundStyle(.orange)
                            Text(warning)
                                .font(.caption)
                                .foregroundStyle(.orange)
                            Spacer()
                            Button {
                                chatService.memoryWarning = nil
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 6)
                        .background(Color.orange.opacity(0.1))
                    }

                    // Multi-turn Chat Scroll Area
                    ChatScrollView(
                        messages: chatService.messages,
                        isGenerating: chatService.isGenerating,
                        modeDescription: modeDescription,
                        onSelectPrompt: { prompt in
                            userMessage = prompt
                            sendMessage()
                        }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(nsColor: .textBackgroundColor).opacity(0.35))

                    Divider()

                    // Enhanced Prompt Input Box
                    ChatPromptInputView(
                        text: $userMessage,
                        isGenerating: chatService.isGenerating,
                        onSend: sendMessage,
                        onStop: {
                            chatService.cancelGeneration()
                        }
                    )
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                // Collapsible Inspector Sidebar
                if showInspector {
                    Divider()

                    ChatInspectorView(
                        chatService: chatService,
                        onClearConversation: {
                            chatService.resetConversation()
                        },
                        onChooseModel: chooseModelDirectory
                    )
                    .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
        }
        .frame(minWidth: 960, minHeight: 600)
    }

    // MARK: - Header Bar

    private var headerView: some View {
        HStack(spacing: 12) {
            // Left: Model Branding & State
            HStack(spacing: 8) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.blue, Color.purple],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 26, height: 26)

                    Image(systemName: "sparkles")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text("Qwen 3.6 35B-A3B")
                            .font(.headline)

                        Text("Q4 Metal")
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1.5)
                            .background(Color.blue.opacity(0.15))
                            .foregroundStyle(.blue)
                            .cornerRadius(4)
                    }

                    HStack(spacing: 6) {
                        Circle()
                            .fill(stateColor)
                            .frame(width: 6, height: 6)

                        Text(chatService.modelState.rawValue)
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.secondary)

                        if chatService.isGenerating {
                            Text("•")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.1f tok/s", chatService.generationTokPerSec))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }

            Spacer(minLength: 16)

            // Center: Runtime Mode Picker
            Picker("", selection: Binding(
                get: { chatService.selectedMode },
                set: { chatService.setMode($0) }
            )) {
                Text("Active Expert Streaming").tag(QwenQ4ChatRunner.RuntimeMode.activeStreaming)
                Text("Fast Resident Q4").tag(QwenQ4ChatRunner.RuntimeMode.fastResident)
                Text("Memory Saver Q4").tag(QwenQ4ChatRunner.RuntimeMode.memorySaver)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 420)
            .disabled(chatService.isGenerating)

            Spacer(minLength: 16)

            // Right: Toolbar Actions
            HStack(spacing: 8) {
                // New Chat Button
                Button {
                    chatService.resetConversation()
                } label: {
                    Label("New Chat", systemImage: "square.and.pencil")
                        .font(.caption.weight(.medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(chatService.isGenerating || chatService.messages.isEmpty)
                .help("Start a new conversation")

                // Inspector Drawer Toggle
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showInspector.toggle()
                    }
                } label: {
                    Image(systemName: "sidebar.trailing")
                        .font(.system(size: 13, weight: showInspector ? .bold : .regular))
                        .foregroundStyle(showInspector ? Color.accentColor : Color.secondary)
                        .frame(width: 26, height: 26)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(showInspector ? Color.accentColor.opacity(0.12) : Color.clear)
                        )
                }
                .buttonStyle(.plain)
                .help(showInspector ? "Hide Inspector" : "Show Inspector")
            }
        }
    }

    // MARK: - Helpers

    private var modeDescription: String {
        switch chatService.selectedMode {
        case .activeStreaming:
            return "Active Expert Streaming — Fast Low Memory. \(chatService.activeStreamingMemoryBudget.displayName) plans \(chatService.expertCacheSlots.formatted()) expert-cache slots at \(chatService.contextLength / 1_024)K context. Changing the budget reloads the model."
        case .fastResident:
            return "Fast Resident Q4 is the high-memory diagnostic mode with all layers resident in RAM."
        case .memorySaver:
            return "Memory Saver Streamed Q4 is the low-throughput fallback mode for tight memory environments."
        }
    }

    private var stateColor: Color {
        switch chatService.modelState {
        case .unloaded: return .gray
        case .loading, .cancelling, .unloading: return .orange
        case .ready: return .green
        case .generating: return .blue
        case .error: return .red
        }
    }

    private func sendMessage() {
        let msg = userMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !msg.isEmpty, !chatService.isGenerating else { return }
        userMessage = ""
        chatService.send(message: msg)
    }

    private func chooseModelDirectory() {
        let panel = NSOpenPanel()
        panel.title = "Choose Qwen 3.6 35B-A3B Model Directory"
        panel.prompt = "Choose Model"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: chatService.modelPath, isDirectory: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try chatService.setModelPath(url.path)
        } catch {
            chatService.errorMessage = error.localizedDescription
        }
    }
}
