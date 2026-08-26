import AppKit
import SwiftUI

struct ChatPromptInputView: View {
    @Binding var text: String
    let isGenerating: Bool
    let onSend: () -> Void
    let onStop: () -> Void
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 8) {
            // Main Input Container
            HStack(alignment: .bottom, spacing: 10) {
                // Text Editor with Placeholder
                ZStack(alignment: .topLeading) {
                    if text.isEmpty {
                        Text("Ask Qwen 3.6 35B-A3B anything…")
                            .font(.system(size: 14))
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 7)
                            .allowsHitTesting(false)
                    }

                    TextEditor(text: $text)
                        .font(.system(size: 14))
                        .lineSpacing(2)
                        .scrollContentBackground(.hidden)
                        .focused($isFocused)
                        .frame(minHeight: 36, maxHeight: 120)
                        .onKeyPress(.return, phases: [.down]) { keyPress in
                            if keyPress.modifiers.contains(.shift) {
                                return .ignored // Allow newline
                            } else if hasMarkedText {
                                return .ignored // Allow IME candidate confirmation
                            } else {
                                if canSend {
                                    onSend()
                                    return .handled
                                }
                                return .ignored
                            }
                        }
                }

                // Action Controls on the Right
                HStack(spacing: 8) {
                    if !text.isEmpty && !isGenerating {
                        Button {
                            text = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 16))
                                .foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                        .help("Clear text")
                    }

                    if isGenerating {
                        Button(action: onStop) {
                            ZStack {
                                Circle()
                                    .fill(Color.red.opacity(0.15))
                                    .frame(width: 32, height: 32)
                                Image(systemName: "stop.fill")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundStyle(.red)
                            }
                        }
                        .buttonStyle(.plain)
                        .help("Stop generating")
                    } else {
                        Button(action: onSend) {
                            ZStack {
                                Circle()
                                    .fill(canSend ? Color.accentColor : Color(nsColor: .separatorColor).opacity(0.3))
                                    .frame(width: 32, height: 32)
                                Image(systemName: "arrow.up")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundStyle(canSend ? .white : .secondary)
                            }
                        }
                        .buttonStyle(.plain)
                        .disabled(!canSend)
                        .help("Send message (Return)")
                    }
                }
                .padding(.bottom, 2)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(
                                isFocused ? Color.accentColor.opacity(0.6) : Color(nsColor: .separatorColor).opacity(0.5),
                                lineWidth: isFocused ? 1.5 : 1
                            )
                    )
            )

            // Hint footer
            HStack {
                Text("Return to send · Shift+Return for newline")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                Spacer()
            }
            .padding(.horizontal, 8)
        }
        .padding(.horizontal, 20)
        .padding(.top, 4)
        .padding(.bottom, 12)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var canSend: Bool {
        !isGenerating && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var hasMarkedText: Bool {
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() == true
    }
}
