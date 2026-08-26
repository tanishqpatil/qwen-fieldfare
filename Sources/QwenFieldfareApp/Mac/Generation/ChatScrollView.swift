import AppKit
import SwiftUI

struct ChatScrollView: View {
    let messages: [ChatMessage]
    let isGenerating: Bool
    let modeDescription: String
    let onSelectPrompt: (String) -> Void

    private let bottomAnchorID = "CHAT_SCROLL_BOTTOM_ANCHOR"

    var body: some View {
        ScrollViewReader { proxy in
            ZStack(alignment: .bottomTrailing) {
                ScrollView {
                    VStack(spacing: 14) {
                        if messages.isEmpty {
                            emptyStateHero
                                .padding(.top, 40)
                                .padding(.horizontal, 24)
                        } else {
                            ForEach(messages) { message in
                                ChatMessageView(message: message)
                                    .id(message.id)
                            }
                        }

                        // Bottom Spacer Anchor
                        Spacer(minLength: 16)
                            .id(bottomAnchorID)
                    }
                    .padding(.vertical, 16)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: messages.count) { _, _ in
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(bottomAnchorID, anchor: .bottom)
                    }
                }
                .onChange(of: messages.last?.content) { _, _ in
                    if isGenerating {
                        // Instant jump without animation for zero-lag token streaming
                        proxy.scrollTo(bottomAnchorID, anchor: .bottom)
                    }
                }
                .onChange(of: isGenerating) { _, generating in
                    if generating {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(bottomAnchorID, anchor: .bottom)
                        }
                    }
                }

                // Floating Scroll to Bottom button
                if !messages.isEmpty {
                    Button {
                        withAnimation(.easeOut(duration: 0.25)) {
                            proxy.scrollTo(bottomAnchorID, anchor: .bottom)
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "arrow.down")
                                .font(.system(size: 11, weight: .bold))
                            if isGenerating {
                                Text("Streaming…")
                                    .font(.caption2.weight(.medium))
                            }
                        }
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                        .overlay(Capsule().stroke(Color.primary.opacity(0.12), lineWidth: 1))
                        .shadow(color: .black.opacity(0.15), radius: 4, y: 2)
                    }
                    .buttonStyle(.plain)
                    .padding(20)
                }
            }
        }
    }

    // MARK: - Empty State Hero

    private var emptyStateHero: some View {
        VStack(spacing: 24) {
            VStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.blue.opacity(0.2), Color.purple.opacity(0.2)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: 64, height: 64)

                    Image(systemName: "sparkles")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [Color.blue, Color.purple],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                }

                VStack(spacing: 4) {
                    Text("Qwen 3.6 35B-A3B")
                        .font(.title2.weight(.bold))

                    Text("Native 4-bit affine quantized inference on Apple Silicon Metal")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                Text(modeDescription)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 540)
            }

            // Prompt Suggestions Grid
            VStack(alignment: .leading, spacing: 10) {
                Text("Try a prompt:")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 4)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 260), spacing: 12)], spacing: 12) {
                    PromptSuggestionCard(
                        icon: "memorychip",
                        title: "Unified Memory",
                        prompt: "Explain how Apple Silicon unified memory architecture benefits high-throughput LLM inference with zero-copy weights.",
                        onSelect: onSelectPrompt
                    )

                    PromptSuggestionCard(
                        icon: "swift",
                        title: "Swift AsyncStream",
                        prompt: "Write a complete Swift 6 example demonstrating how to stream LLM tokens using AsyncThrowingStream.",
                        onSelect: onSelectPrompt
                    )

                    PromptSuggestionCard(
                        icon: "waveform.path.ecg",
                        title: "DeltaNet vs Attention",
                        prompt: "Explain the architectural difference and computational complexity between Linear Attention (DeltaNet) and standard Full Attention.",
                        onSelect: onSelectPrompt
                    )

                    PromptSuggestionCard(
                        icon: "chart.bar.doc.horizontal",
                        title: "MoE Expert Routing",
                        prompt: "Describe how Mixture of Experts (MoE) dynamic routing selects top-8 experts and how streaming weight caches optimize memory bandwidth.",
                        onSelect: onSelectPrompt
                    )
                }
            }
            .frame(maxWidth: 680)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Prompt Suggestion Card

private struct PromptSuggestionCard: View {
    let icon: String
    let title: String
    let prompt: String
    let onSelect: (String) -> Void
    @State private var isHovered = false

    var body: some View {
        Button {
            onSelect(prompt)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: icon)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Color.accentColor)
                    Text(title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.primary)
                }

                Text(prompt)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
            }
            .padding(12)
            .frame(maxWidth: .infinity, minHeight: 74, alignment: .topLeading)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color(nsColor: .controlBackgroundColor).opacity(isHovered ? 0.9 : 0.5))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(
                                isHovered ? Color.accentColor.opacity(0.4) : Color(nsColor: .separatorColor).opacity(0.4),
                                lineWidth: 1
                            )
                    )
            )
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
    }
}
