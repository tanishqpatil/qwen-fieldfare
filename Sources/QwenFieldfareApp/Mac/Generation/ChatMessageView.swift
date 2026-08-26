import AppKit
import SwiftUI

struct ChatMessageView: View {
    let message: ChatMessage
    @State private var isCopied = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if message.role == .user {
                Spacer(minLength: 40)
                userMessageBody
            } else {
                assistantMessageBody
                Spacer(minLength: 40)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
    }

    // MARK: - User Message

    private var userMessageBody: some View {
        VStack(alignment: .trailing, spacing: 6) {
            HStack(spacing: 6) {
                Text("You")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(message.timestamp, format: .dateTime.hour().minute())
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            VStack(alignment: .leading, spacing: 0) {
                Text(message.content)
                    .font(.system(size: 14))
                    .lineSpacing(3)
                    .foregroundStyle(Color(nsColor: .labelColor))
                    .textSelection(.enabled)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color.accentColor.opacity(0.12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.accentColor.opacity(0.25), lineWidth: 1)
                    )
            )
        }
    }

    // MARK: - Assistant Message

    private var assistantMessageBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack(spacing: 8) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.blue.opacity(0.8), Color.purple.opacity(0.8)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 22, height: 22)
                    Image(systemName: "sparkles")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.white)
                }

                Text("Qwen 3.6 35B-A3B")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)

                if message.isStreaming {
                    HStack(spacing: 4) {
                        ProgressView()
                            .controlSize(.mini)
                        Text("Generating…")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                Text(message.timestamp, format: .dateTime.hour().minute())
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                if !message.content.isEmpty && !message.isStreaming {
                    copyButton
                }
            }

            // Error banner if any
            if let error = message.errorMessage {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.callout)
                        .foregroundStyle(.red)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.red.opacity(0.1))
                .cornerRadius(8)
            }

            // Body Content
            if message.content.isEmpty && message.isStreaming {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Processing prompt…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 8)
            } else {
                MarkdownContentView(content: message.content, isStreaming: message.isStreaming)
                    .padding(.vertical, 2)
            }

            // Metrics footer when finished
            if let metrics = message.metrics, !message.isStreaming {
                metricsFooter(metrics)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1)
                )
        )
    }

    // MARK: - Metrics Footer

    private func metricsFooter(_ m: ChatMessageMetrics) -> some View {
        HStack(spacing: 12) {
            if let rate = m.steadyTokensPerSecond ?? m.runAverageTokensPerSecond {
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 10))
                        .foregroundStyle(.yellow)
                    Text(String(format: "%.1f tok/s", rate))
                }
            }

            if let tokens = m.generatedTokenCount {
                HStack(spacing: 4) {
                    Image(systemName: "character.cursor.ibeam")
                        .font(.system(size: 10))
                    Text("\(tokens) tokens")
                }
            }

            if let ttft = m.firstTokenLatencyMs {
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                        .font(.system(size: 10))
                    Text(String(format: "TTFT: %.0f ms", ttft))
                }
            }

            if let hitRate = m.cacheHitRate {
                HStack(spacing: 4) {
                    Image(systemName: "memorychip")
                        .font(.system(size: 10))
                    Text(String(format: "Cache: %.1f%%", hitRate * 100))
                }
            }

            if let reason = m.finishReason {
                Text("(\(reason))")
                    .foregroundStyle(.tertiary)
            }
        }
        .font(.system(size: 11, design: .monospaced))
        .foregroundStyle(.secondary)
        .padding(.top, 4)
    }

    // MARK: - Copy Action

    private var copyButton: some View {
        Button {
            copyMessage()
        } label: {
            HStack(spacing: 3) {
                Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 11))
                if isCopied {
                    Text("Copied")
                        .font(.system(size: 10, weight: .medium))
                }
            }
            .foregroundStyle(isCopied ? .green : .secondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.primary.opacity(isCopied ? 0.08 : 0.03))
            )
        }
        .buttonStyle(.plain)
        .help("Copy full response")
    }

    private func copyMessage() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message.content, forType: .string)
        withAnimation(.spring(duration: 0.2)) {
            isCopied = true
        }
        Task {
            try? await Task.sleep(for: .seconds(2))
            withAnimation(.easeInOut(duration: 0.2)) {
                isCopied = false
            }
        }
    }
}
