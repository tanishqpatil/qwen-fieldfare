import AppKit
import SwiftUI
import QwenFieldfareCore

struct ChatInspectorView: View {
    @Bindable var chatService: ChatService
    let onClearConversation: () -> Void
    let onChooseModel: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Section: Mode Guide
                modeGuideSection

                Divider()

                // Section: Parameters
                parametersSection

                Divider()

                // Section: Hardware Telemetry
                telemetrySection

                Divider()

                // Section: Cache & Latency
                cacheSection

                Divider()

                // Section: Management
                managementSection
            }
            .padding(16)
        }
        .frame(width: 300)
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.85))
    }

    // MARK: - Mode Guide Section

    private var modeGuideSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Runtime Modes", systemImage: "cpu")
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 8) {
                modeCard(
                    title: "Active Expert Streaming",
                    subtitle: "Fast · Low Memory (≤5 GiB Metal)",
                    description: "Dense Q4 weights stay resident in Metal RAM; routed MoE experts use a bounded \(chatService.expertCacheSlots.formatted())-slot cache at the selected context. Recommended for general chat.",
                    isSelected: chatService.selectedMode == .activeStreaming
                )

                modeCard(
                    title: "Fast Resident Q4",
                    subtitle: "Max Speed · Diagnostic (~18-20 GiB)",
                    description: "Loads all 40 layers and all experts fully into Unified Memory. Highest throughput, requires ~20 GiB free RAM.",
                    isSelected: chatService.selectedMode == .fastResident
                )

                modeCard(
                    title: "Memory Saver Q4",
                    subtitle: "Ultra Low Memory (~2-3 GiB)",
                    description: "Streams layer weights and experts with minimal cache footprint. Lower token generation rate.",
                    isSelected: chatService.selectedMode == .memorySaver
                )
            }
        }
    }

    private func modeCard(title: String, subtitle: String, description: String, isSelected: Bool) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
                Spacer()
                if isSelected {
                    Text("ACTIVE")
                        .font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1.5)
                        .background(Color.accentColor.opacity(0.15))
                        .foregroundStyle(Color.accentColor)
                        .cornerRadius(3)
                }
            }

            Text(subtitle)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(.secondary)

            Text(description)
                .font(.system(size: 10.5))
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 1)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.08) : Color(nsColor: .controlBackgroundColor).opacity(0.4))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(isSelected ? Color.accentColor.opacity(0.3) : Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
                )
        )
    }

    // MARK: - Parameters Section

    private var parametersSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Generation Parameters", systemImage: "slider.horizontal.3")
                .font(.headline)
                .foregroundStyle(.primary)

            Toggle("Reasoning (Thinking)", isOn: $chatService.enableThinking)
                .toggleStyle(.checkbox)
                .font(.subheadline)
                .disabled(chatService.isGenerating)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Context Length")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Picker("", selection: Binding(
                        get: { chatService.contextLength },
                        set: { chatService.setContextLength($0) }
                    )) {
                        ForEach(chatService.supportedContextLengths, id: \.self) { tokens in
                            Text("\(tokens / 1_024)K").tag(tokens)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 90)
                }
            }
            .disabled(chatService.isGenerating)

            if chatService.selectedMode == .activeStreaming {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Active Memory Budget")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Picker("", selection: Binding(
                            get: { chatService.activeStreamingMemoryBudget },
                            set: { chatService.setActiveStreamingMemoryBudget($0) }
                        )) {
                            ForEach(QwenActiveStreamingMemoryBudget.allCases, id: \.self) { budget in
                                Text(budget.displayName).tag(budget)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 90)
                    }
                    Text("Changes unload the model and reset the active runtime allocation.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .disabled(chatService.isGenerating)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Max Output")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Picker("", selection: $chatService.maxOutputTokens) {
                        ForEach([512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536]
                            .filter { $0 <= chatService.contextLength }, id: \.self) { tokens in
                            Text(tokens == 512 ? "512" : "\(tokens / 1_024)K").tag(tokens)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 90)
                }
            }
            .disabled(chatService.isGenerating)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Temperature")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.2f", chatService.temperature))
                        .font(.caption.monospacedDigit())
                }
                Slider(value: $chatService.temperature, in: 0...2, step: 0.05)
            }
            .disabled(chatService.isGenerating)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Top-K")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Stepper("\(chatService.topK)", value: $chatService.topK, in: 1...256)
                }
            }
            .disabled(chatService.isGenerating)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Top-P")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.2f", chatService.topP))
                        .font(.caption.monospacedDigit())
                }
                Slider(value: $chatService.topP, in: 0.01...1, step: 0.01)
            }
            .disabled(chatService.isGenerating)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Presence Penalty")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.1f", chatService.presencePenalty))
                        .font(.caption.monospacedDigit())
                }
                Slider(value: $chatService.presencePenalty, in: -2...2, step: 0.1)
            }
            .disabled(chatService.isGenerating)
        }
    }

    // MARK: - Hardware Telemetry Section

    private var telemetrySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Hardware & Memory", systemImage: "memorychip")
                .font(.headline)
                .foregroundStyle(.primary)

            telemetryRow(title: "Metal Allocated", value: String(format: "%.2f GiB", chatService.metalAllocatedGiB))
            telemetryRow(title: "System Wired RAM", value: String(format: "%.2f GiB", chatService.systemWiredGiB))
            telemetryRow(title: "KV Cache Size", value: String(format: "%.2f GiB", chatService.estimatedKVGiB))

            if chatService.selectedMode == .activeStreaming {
                telemetryRow(
                    title: "Active Runtime Plan",
                    value: String(
                        format: "%.2f / %@",
                        chatService.estimatedActiveRuntimeGiB,
                        chatService.activeStreamingMemoryBudget.displayName
                    )
                )
                telemetryRow(title: "Expert Cache Slots", value: chatService.expertCacheSlots.formatted())
            }

            HStack {
                Text("Memory Pressure")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(chatService.memoryPressure)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(pressureColor)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(pressureColor.opacity(0.12))
                    .cornerRadius(4)
            }
        }
    }

    // MARK: - Cache & Latency Section

    private var cacheSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Diagnostics & Performance", systemImage: "gauge.with.dots.needle.bottom.50percent")
                .font(.headline)
                .foregroundStyle(.primary)

            telemetryRow(title: "First Token Latency", value: chatService.metrics.firstTokenLatencyMs.map { String(format: "%.1f ms", $0) } ?? "N/A")
            telemetryRow(title: "Prompt Prefill", value: chatService.metrics.prefillDurationMs.map { String(format: "%.1f ms", $0) } ?? "N/A")
            telemetryRow(title: "p50 Token Latency", value: chatService.metrics.latency.p50Ms.map { String(format: "%.1f ms", $0) } ?? "N/A")
            telemetryRow(title: "p95 Token Latency", value: chatService.metrics.latency.p95Ms.map { String(format: "%.1f ms", $0) } ?? "N/A")
            telemetryRow(title: "Steady Rate", value: chatService.metrics.steadyTokensPerSecond.map { String(format: "%.2f tok/s", $0) } ?? "N/A")
            telemetryRow(title: "Average Rate", value: chatService.metrics.runAverageTokensPerSecond.map { String(format: "%.2f tok/s", $0) } ?? "N/A")
            telemetryRow(title: "Tokens Generated", value: "\(chatService.metrics.generatedTokenCount)")
            telemetryRow(title: "Cache Hit Rate", value: String(format: "%.1f%%", chatService.expertCacheHitRate))
            telemetryRow(title: "Cache Misses", value: String(format: "%.1f / tok", chatService.expertCacheMissesPerToken))
            telemetryRow(title: "Cache Occupancy", value: occupancyText)
        }
    }

    // MARK: - Management Section

    private var managementSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Model Management", systemImage: "gearshape")
                .font(.headline)
                .foregroundStyle(.primary)

            Toggle("Unload when idle (60s)", isOn: $chatService.unloadWhenIdle)
                .toggleStyle(.checkbox)
                .font(.caption)

            VStack(alignment: .leading, spacing: 3) {
                Text("Model Directory")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(chatService.modelPath)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                    .textSelection(.enabled)
                    .help(chatService.modelPath)
            }

            Button(action: onChooseModel) {
                Label("Choose Model Directory…", systemImage: "folder")
                    .font(.caption)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(chatService.isGenerating || chatService.modelState == .loading || chatService.modelState == .unloading)

            HStack(spacing: 8) {
                Button(role: .destructive) {
                    onClearConversation()
                } label: {
                    Label("Clear Chat", systemImage: "trash")
                        .font(.caption)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(chatService.isGenerating)

                Button {
                    chatService.isGenerating ? chatService.cancelGeneration() : chatService.unloadModel()
                } label: {
                    Text(chatService.isGenerating ? "Stop" : "Unload")
                        .font(.caption)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(chatService.modelState == .unloaded || chatService.modelState == .loading || chatService.modelState == .unloading)
            }
        }
    }

    // MARK: - Helpers

    private func telemetryRow(title: String, value: String) -> some View {
        HStack {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.primary)
        }
    }

    private var occupancyText: String {
        guard let cache = chatService.metrics.cache else { return "N/A" }
        return "\(cache.occupancy)/\(cache.capacity)"
    }

    private var pressureColor: Color {
        switch chatService.memoryPressure {
        case "Critical": return .red
        case "Warning": return .orange
        default: return .green
        }
    }
}
